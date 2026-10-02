"""Native TopoBench TRAWL backbone."""

import copy

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from topobench.data.utils.trawl.sampling import WalkSampler, walk_seed
from topobench.nn.backbones.general.trawl_blocks import make_layer

__all__ = ["TRAWL", "upload"]


def upload(array, device):
    """Copy a host array to ``device`` without making the host wait.

    Integer arrays become int64 on every platform. CUDA copies go through
    pinned memory with ``non_blocking`` so they never synchronize.
    """
    array = np.ascontiguousarray(array)
    if array.dtype.kind in "iub":
        array = array.astype(np.int64, copy=False)
    tensor = torch.from_numpy(array)
    if torch.device(device).type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


class NeighborhoodFusion(nn.Module):
    """Fuse available neighborhood embeddings, preserving empty-slot semantics."""

    def __init__(self, dim, count, mode="mean"):
        super().__init__()
        if mode not in {"mean", "concat", "learned", "attention"}:
            raise ValueError(f"Unknown neighborhood fusion {mode}")
        self.mode = mode
        self.weights = (
            nn.Parameter(torch.zeros(count)) if mode == "learned" else None
        )
        self.attention = nn.Linear(dim, 1) if mode == "attention" else None
        self.project = (
            nn.Linear(count * (dim + 1), dim) if mode == "concat" else None
        )

    def forward(self, values, available, fallback):
        if not bool(available.any()):
            return fallback
        if self.mode == "concat":
            slots = torch.cat(
                (values * available[:, None], available[:, None].to(values)),
                dim=1,
            )
            return self.project(slots.flatten())
        if self.mode == "mean":
            scores = values.new_zeros(len(values))
        elif self.mode == "learned":
            scores = self.weights
        else:
            scores = self.attention(values).squeeze(-1)
        weights = scores.masked_fill(~available, -torch.inf).softmax(dim=0)
        return (values * weights[:, None]).sum(dim=0)


class TRAWL(nn.Module):
    """Encode fixed-budget topological walks using configurable sequence layers.

    Inputs are produced by ``TRAWLTransform``. Outputs include contextual
    ``x_0`` for ordinary TopoBench readouts and graph embeddings for
    ``TRAWLReadout``. No dataset names or task-specific branches occur here.
    Call ``initialize`` before constructing optimizers when input widths are
    inferred (the standard runner does this automatically).
    """

    def __init__(
        self,
        hidden_dim=128,
        max_rank=2,
        in_channels=None,
        pe_dim=None,
        layers=None,
        architecture="hybrid",
        depth=5,
        layer_options=None,
        walks=None,
        walk_scope="union",
        num_neighborhoods=2,
        encoder_sharing="shared",
        fusion="mean",
        pooling="mean_max",
        rank_embedding=True,
        num_colors=0,
        move_embedding=True,
        dropout=0.0,
        seed=42,
        eval_views=1,
        checkpoint_layers=False,
        walk_refresh="train",
        occurrence_pooling="mean",
        graph_readout="walks",
        transition_cache_size=1024,
    ):
        super().__init__()
        if walk_scope not in {"union", "separate"}:
            raise ValueError("walk_scope must be union or separate")
        if encoder_sharing not in {"shared", "independent"}:
            raise ValueError("encoder_sharing must be shared or independent")
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("Unknown time pooling")
        if eval_views < 1 or depth < 1 or num_neighborhoods < 1:
            raise ValueError(
                "Depth, neighborhood count and eval_views must be positive"
            )
        self.hidden_dim, self.max_rank = hidden_dim, max_rank
        self.walk_sampler = WalkSampler(transition_cache_size)
        self.walk_scope, self.pooling = walk_scope, pooling
        self.num_neighborhoods = num_neighborhoods
        self.walks = dict(walks or {"k": 32, "length": 32})
        self.seed, self.eval_views = seed, eval_views
        self.checkpoint_layers = checkpoint_layers
        if walk_refresh not in {"train", "fixed"}:
            raise ValueError("walk_refresh must be train or fixed")
        self.walk_refresh = walk_refresh
        if occurrence_pooling not in {
            "mean",
            "attention",
        } or graph_readout not in {"walks", "cells"}:
            raise ValueError("Invalid occurrence pooling or graph readout")
        self.occurrence_pooling, self.graph_readout = (
            occurrence_pooling,
            graph_readout,
        )
        self.occurrence_attention = (
            nn.Linear(hidden_dim, 1)
            if occurrence_pooling == "attention"
            else None
        )
        self.register_buffer(
            "sampling_step", torch.zeros((), dtype=torch.long)
        )
        # Host mirror of ``sampling_step``; reading the buffer would wait for
        # the device on every forward pass.
        self._host_sampling_step = None
        self.features = nn.ModuleList(
            [
                nn.Linear(in_channels[r], hidden_dim)
                if in_channels is not None
                else nn.LazyLinear(hidden_dim)
                for r in range(max_rank + 1)
            ]
        )
        self.position = (
            nn.LazyLinear(hidden_dim, bias=False)
            if pe_dim is None
            else nn.Linear(pe_dim, hidden_dim, bias=False)
        )
        self.rank = (
            nn.Embedding(max_rank + 1, hidden_dim) if rank_embedding else None
        )
        self.color = (
            nn.Embedding(num_colors, hidden_dim) if num_colors else None
        )
        self.move = nn.Embedding(3, hidden_dim) if move_embedding else None
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        options = dict(layer_options or {})
        if layers is None:
            if architecture not in {
                "mamba",
                "sisa",
                "hybrid",
                "gru",
                "transformer",
                "mlp",
            }:
                raise ValueError(
                    "Custom architecture requires an explicit layer list"
                )
            kinds = [
                ("mamba" if i % 2 == 0 else "sisa")
                if architecture == "hybrid"
                else architecture
                for i in range(depth)
            ]
            layers = [
                {"kind": kind, **dict(options.get(kind, {}))} for kind in kinds
            ]
        if not layers:
            raise ValueError("Provide at least one sequence layer")
        encoder = nn.ModuleList(
            [make_layer(config, hidden_dim) for config in layers]
        )
        count = (
            num_neighborhoods
            if walk_scope == "separate" and encoder_sharing == "independent"
            else 1
        )
        self.encoders = nn.ModuleList(
            [encoder] + [copy.deepcopy(encoder) for _ in range(count - 1)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * (2 if pooling == "mean_max" else 1)
        self.fusion = NeighborhoodFusion(
            self.output_dim, num_neighborhoods, fusion
        )

    def initialize(self, data_list):
        """Materialize projections from training data, including absent ranks."""
        if not data_list:
            raise ValueError(
                "Cannot infer TRAWL input widths from an empty dataset"
            )
        with torch.no_grad():
            for rank, projection in enumerate(self.features):
                if not isinstance(projection, nn.LazyLinear):
                    continue
                key = f"trawl_signal_{rank}"
                widths = {
                    data[key].shape[1] for data in data_list if len(data[key])
                }
                if len(widths) > 1:
                    raise ValueError(
                        f"Inconsistent rank-{rank} feature widths: {widths}"
                    )
                width = next(iter(widths), data_list[0][key].shape[1])
                projection(
                    torch.zeros(1, width, device=projection.weight.device)
                )
            if isinstance(self.position, nn.LazyLinear):
                self.position(
                    torch.zeros(
                        1,
                        data_list[0].trawl_pe.shape[1],
                        device=self.position.weight.device,
                    )
                )

    # Walk metadata read on the host (see ``TBModel.on_before_batch_transfer``).
    host_fields = (
        "trawl_counts",
        "trawl_edges",
        "trawl_weights",
        "trawl_identity",
        "trawl_relations",
        "trawl_colors",
    )

    def _host(self, batch):
        """CPU copies of ``host_fields``, without waiting for the device."""
        host = getattr(batch, "trawl_host", None)
        if host is None:
            host = {
                key: batch[key].detach().cpu().numpy()
                for key in self.host_fields
                if key in batch
            }
        return host

    def next_sampling_step(self):
        """Return the training sampling step and advance it."""
        if self._host_sampling_step is None:
            self._host_sampling_step = int(self.sampling_step)
        step = self._host_sampling_step
        self._host_sampling_step += 1
        self.sampling_step.add_(1)
        return step

    def reset_sampling_step(self):
        """Restart walk refreshes, e.g. before supervised fitting."""
        self.sampling_step.zero_()
        self._host_sampling_step = 0

    def _load_from_state_dict(self, *args, **kwargs):
        # A restored buffer invalidates the host mirror.
        self._host_sampling_step = None
        super()._load_from_state_dict(*args, **kwargs)

    def _pool(self, x):
        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "max":
            return x.amax(dim=1)
        # Historical head order is max, then mean.
        return torch.cat((x.amax(dim=1), x.mean(dim=1)), dim=-1)

    def _encode(self, x, paths, ranks, relation):
        tokens = x[paths]
        if self.move is not None:
            path_ranks = ranks[paths]
            movement = torch.zeros_like(path_ranks)
            movement[:, 1:] = torch.sign(
                path_ranks[:, 1:] - path_ranks[:, :-1]
            )
            tokens = tokens + self.move(movement + 1)
        tokens = self.dropout(self.input_norm(tokens))
        encoder = self.encoders[0 if len(self.encoders) == 1 else relation]
        for layer in encoder:
            if self.checkpoint_layers and self.training:
                from torch.utils.checkpoint import checkpoint

                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        return self.output_norm(tokens)

    def forward(self, batch):
        host = self._host(batch)
        counts = host["trawl_counts"].reshape(-1, self.max_rank + 1)
        edge_slices = getattr(batch, "_slice_dict", {}).get("trawl_edges")
        node_offsets = np.zeros(self.max_rank + 1, dtype=int)
        state_offset = 0
        (
            graph_embeddings,
            nodes,
            node_batches,
            walk_embeddings,
            walk_batches,
        ) = [], [], [], [], []
        all_cells, all_ranks, walk_views = [], [], []
        rank_cells = [[] for _ in range(self.max_rank + 1)]
        rank_batches = [[] for _ in range(self.max_rank + 1)]
        views = 1 if self.training else self.eval_views
        step = self.next_sampling_step() if self.training else 0
        if self.walk_refresh != "train":
            step = 0
        for graph_id, sizes in enumerate(counts):
            if (
                int(host["trawl_relations"][graph_id])
                != self.num_neighborhoods
            ):
                raise ValueError("Transform/model neighborhood counts differ")
            signals = []
            for rank, size in enumerate(sizes):
                raw = batch[f"trawl_signal_{rank}"][
                    node_offsets[rank] : node_offsets[rank] + size
                ]
                signals.append(
                    self.features[rank](raw)
                    if size
                    else batch.trawl_pe.new_empty(0, self.hidden_dim)
                )
                node_offsets[rank] += size
            features = torch.cat(signals)
            total = int(sizes.sum())
            ranks = upload(
                np.repeat(np.arange(len(sizes)), sizes), features.device
            )
            x = features + self.position(
                batch.trawl_pe[state_offset : state_offset + total]
            )
            if self.rank is not None:
                x = x + self.rank(ranks)
            if self.color is not None:
                colors = batch.trawl_colors[
                    state_offset : state_offset + total
                ]
                host_colors = host["trawl_colors"][
                    state_offset : state_offset + total
                ]
                if (
                    host_colors.size
                    and int(host_colors.max()) >= self.color.num_embeddings
                ):
                    raise ValueError(
                        "num_colors is smaller than a transformed color ID"
                    )
                x = x + self.color(colors)
            if edge_slices is None:
                edges, weights = host["trawl_edges"], host["trawl_weights"]
            else:
                start, stop = (
                    int(edge_slices[graph_id]),
                    int(edge_slices[graph_id + 1]),
                )
                edges, weights = (
                    host["trawl_edges"][start:stop],
                    host["trawl_weights"][start:stop],
                )
            groups = (
                [None]
                if self.walk_scope == "union"
                else list(range(self.num_neighborhoods))
            )
            fallback = self._pool(features[None])[0]
            per_group, availability = [], []
            # Accumulate occurrences in full precision under autocast. Embedding
            # additions can promote encoded states independently of projections.
            accumulation_dtype = (
                torch.float32
                if features.dtype in {torch.float16, torch.bfloat16}
                else features.dtype
            )
            contextual = torch.zeros_like(features, dtype=accumulation_dtype)
            visits = contextual.new_zeros(total, 1)
            occurrence_values, occurrence_ids = [], []
            for group_id, relation in enumerate(groups):
                select = (
                    np.ones(len(edges), dtype=bool)
                    if relation is None
                    else edges[:, 2] == relation
                )
                chosen = edges[select]
                matrix = sp.csr_matrix(
                    (weights[select], (chosen[:, 0], chosen[:, 1])),
                    shape=(total, total),
                )
                pooled_views = []
                for view in range(views):
                    seed = walk_seed(
                        self.seed,
                        int(host["trawl_identity"][graph_id]),
                        step,
                        group_id,
                        view,
                    )
                    paths = self.walk_sampler(matrix, seed=seed, **self.walks)
                    if not len(paths):
                        continue
                    paths = upload(paths, x.device)
                    encoded = self._encode(x, paths, ranks, group_id)
                    pooled = self._pool(encoded)
                    pooled_views.append(pooled.mean(dim=0))
                    walk_embeddings.append(pooled)
                    walk_views.append(
                        torch.full(
                            (len(paths),),
                            view,
                            device=x.device,
                            dtype=torch.long,
                        )
                    )
                    walk_batches.append(
                        torch.full(
                            (len(paths),),
                            graph_id,
                            device=x.device,
                            dtype=torch.long,
                        )
                    )
                    contextual.index_add_(
                        0,
                        paths.flatten(),
                        encoded.flatten(0, 1).to(contextual),
                    )
                    if self.occurrence_attention is not None:
                        occurrence_values.append(encoded.flatten(0, 1))
                        occurrence_ids.append(paths.flatten())
                    visits.index_add_(
                        0,
                        paths.flatten(),
                        visits.new_ones(paths.numel(), 1),
                    )
                availability.append(bool(pooled_views))
                per_group.append(
                    torch.stack(pooled_views).mean(dim=0)
                    if pooled_views
                    else fallback * 0
                )
            if self.walk_scope == "union":
                graph_embeddings.append(
                    per_group[0] if availability[0] else fallback
                )
            else:
                graph_embeddings.append(
                    self.fusion(
                        torch.stack(per_group),
                        upload(np.array(availability), x.device).bool(),
                        fallback,
                    )
                )
            contextual = torch.where(
                visits > 0, contextual / visits.clamp_min(1), features
            )
            if occurrence_values:
                from torch_geometric.utils import softmax

                values, ids = (
                    torch.cat(occurrence_values),
                    torch.cat(occurrence_ids),
                )
                weights = softmax(
                    self.occurrence_attention(values), ids, num_nodes=total
                )
                attended = torch.zeros_like(contextual).index_add(
                    0, ids, (values * weights).to(contextual)
                )
                contextual = torch.where(visits > 0, attended, features)
            if self.graph_readout == "cells":
                graph_embeddings[-1] = self._pool(contextual[None])[0]
            rank_offset = 0
            for rank, size in enumerate(sizes):
                rank_cells[rank].append(
                    contextual[rank_offset : rank_offset + size]
                )
                rank_batches[rank].append(
                    torch.full(
                        (size,), graph_id, dtype=torch.long, device=x.device
                    )
                )
                rank_offset += size
            all_cells.append(contextual)
            all_ranks.append(ranks)
            nodes.append(contextual[: sizes[0]])
            node_batches.append(
                torch.full(
                    (sizes[0],), graph_id, dtype=torch.long, device=x.device
                )
            )
            state_offset += total
        result = {
            "x_0": torch.cat(nodes),
            "batch_0": torch.cat(node_batches),
            "cell_embeddings": torch.cat(all_cells),
            "cell_ranks": torch.cat(all_ranks),
            "graph_embedding": torch.stack(graph_embeddings),
            "labels": batch.get("y"),
            "walk_readout_compatible": self.walk_scope == "union"
            and self.graph_readout == "walks",
            "num_views": views,
        }
        result["walk_embedding"] = (
            torch.cat(walk_embeddings)
            if walk_embeddings
            else result["graph_embedding"].new_empty((0, self.output_dim))
        )
        result["walk_batch"] = (
            torch.cat(walk_batches)
            if walk_batches
            else result["batch_0"].new_empty(0)
        )
        result["walk_view"] = (
            torch.cat(walk_views)
            if walk_views
            else result["batch_0"].new_empty(0)
        )
        for rank in range(1, self.max_rank + 1):
            result[f"x_{rank}"] = torch.cat(rank_cells[rank])
            result[f"batch_{rank}"] = torch.cat(rank_batches[rank])
        return result
