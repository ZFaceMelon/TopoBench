"""Continuous TRAWL input encoding used by the historical hybrid recipes."""

import hashlib
import math
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from topobench.data.utils.trawl.sampling import (
    prepare_walk_rows,
    sample_neighbors,
    sample_walks,
    walk_seed,
)
from topobench.nn.backbones.general.trawl import TRAWL, upload

__all__ = ["ContinuousTRAWL"]


class ContinuousWalkInput(nn.Module):
    """Normalize state/PSE channels and inject absolute-difference/mean steps.

    BatchNorm sees all walks in the minibatch together, as in the original
    implementation. Synthetic steps are constructed after normalization.

    Parameters
    ----------
    signal_dim : int
        Width of the continuous state features.
    pe_dim : int
        Width of the positional/structural encodings.
    hidden_dim : int
        Output hidden dimension.
    embed_dim : int, optional
        Width of each per-channel embedding (default: 64).
    """

    def __init__(self, signal_dim, pe_dim, hidden_dim, embed_dim=64):
        super().__init__()
        self.node_norm = nn.BatchNorm1d(signal_dim)
        self.pe_norm = nn.BatchNorm1d(pe_dim)
        self.node_proj = nn.Linear(signal_dim, embed_dim)
        self.edge_proj = nn.Linear(signal_dim * 2, embed_dim)
        self.pe_proj = nn.Linear(pe_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim * 3)
        self.in_proj = nn.Linear(embed_dim * 3, hidden_dim)

    def forward(self, signals, pe):
        """Embed walk states, synthetic steps and encodings.

        Parameters
        ----------
        signals : torch.Tensor
            State features of shape ``[walk, time, signal_dim]``.
        pe : torch.Tensor
            Encodings of shape ``[walk, time, pe_dim]``.

        Returns
        -------
        torch.Tensor
            Walk tokens of shape ``[walk, time, hidden_dim]``.
        """
        signals = self.node_norm(signals.transpose(1, 2)).transpose(1, 2)
        pe = self.pe_norm(pe.transpose(1, 2)).transpose(1, 2)
        difference = (signals[:, 1:] - signals[:, :-1]).abs()
        average = (signals[:, 1:] + signals[:, :-1]) / 2
        steps = F.pad(torch.cat((difference, average), dim=-1), (0, 0, 1, 0))
        x = torch.cat(
            (self.node_proj(signals), self.edge_proj(steps), self.pe_proj(pe)),
            dim=-1,
        )
        return self.in_proj(self.input_norm(x))


class ContinuousTRAWL(TRAWL):
    """TRAWL variant for continuous state features and synthetic step inputs.

    Uses the same layer factory, optimizer, losses and runner as base TRAWL.
    Transform-provided ``trawl_continuous`` features must align with walk
    states. This variant processes the complete walk minibatch jointly.

    Parameters
    ----------
    signal_dim : int, optional
        Width of the continuous state features (default: 4).
    pe_dim : int, optional
        Width of the positional/structural encodings (default: 32).
    embed_dim : int, optional
        Width of each input channel embedding (default: 64).
    sampling_protocol : str, optional
        ``"stable"`` or ``"historical"`` walk sampling (default: "stable").
    branch_gate : dict, optional
        Settings (``prior``, ``hidden_dim``) of a learned gate weighting the
        two lifting branches; requires historical sampling (default: None).
    loss : dict, optional
        Auxiliary reconstruction loss settings (``weight``, ``decay_epochs``,
        ``mask_probability``, ``objective``) (default: None).
    cache_size : int, optional
        Maximum entries of the structure and evaluation walk caches
        (default: 65536).
    **kwargs : dict
        Additional arguments passed to ``TRAWL``.
    """

    def __init__(
        self,
        signal_dim=4,
        pe_dim=32,
        embed_dim=64,
        sampling_protocol="stable",
        branch_gate=None,
        loss=None,
        cache_size=65536,
        **kwargs,
    ):
        super().__init__(pe_dim=pe_dim, **kwargs)
        # Sampler inputs per graph (by content hash) and evaluation walks,
        # whose seeds do not depend on the epoch. Neither changes results.
        self.cache_size = int(cache_size)
        self._structure_cache = OrderedDict()
        self._eval_walk_cache = OrderedDict()
        if self.walk_scope != "union":
            raise ValueError(
                "ContinuousTRAWL currently requires union walks; use base TRAWL for neighborhood fusion"
            )
        if self.occurrence_pooling != "mean" or self.graph_readout != "walks":
            raise ValueError(
                "Historical inputs require mean occurrences and walk readout; use base TRAWL for other choices"
            )
        self.continuous_input = ContinuousWalkInput(
            signal_dim, pe_dim, self.hidden_dim, embed_dim
        )
        if sampling_protocol not in {"stable", "historical"}:
            raise ValueError("Unknown sampling protocol")
        self.sampling_protocol = sampling_protocol
        self.auxiliary = dict(loss or {})
        self.sampling_context = {
            "epoch": 1,
            "batch": 0,
            "rank": 0,
            "stage": "Training",
        }
        # These additive input modules are unused in this encoding strategy.
        self.features = nn.ModuleList()
        self.position = nn.Identity()
        self.rank = self.color = self.move = None
        self.input_norm = nn.Identity()
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(self.hidden_dim, 2 * signal_dim + pe_dim),
        )
        self.branch_gate = None
        if branch_gate is not None:
            if sampling_protocol != "historical":
                # Only historical sampling allocates walks per lifting branch.
                raise ValueError(
                    "branch_gate requires sampling_protocol=historical"
                )
            prior = float(branch_gate.get("prior", 0.7))
            if not 0 < prior < 1:
                raise ValueError("Branch prior must be in (0, 1)")
            self.branch_gate = nn.Sequential(
                nn.Linear(
                    2 * self.output_dim, branch_gate.get("hidden_dim", 64)
                ),
                nn.GELU(),
                nn.Linear(branch_gate.get("hidden_dim", 64), 1),
            )
            nn.init.zeros_(self.branch_gate[-1].weight)
            nn.init.constant_(
                self.branch_gate[-1].bias, math.log(prior / (1 - prior))
            )

    def initialize(self, data_list):
        """Check that data widths match the configured input layers.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Transformed training graphs.
        """
        for data in data_list:
            if (
                data.trawl_continuous.shape[1]
                != self.continuous_input.node_norm.num_features
            ):
                raise ValueError(
                    "Continuous signal width does not match model"
                )
            if (
                data.trawl_pe.shape[1]
                != self.continuous_input.pe_norm.num_features
            ):
                raise ValueError(
                    "Positional encoding width does not match model"
                )

    # Walk metadata read on the host. TBModel stashes CPU copies before the
    # batch moves to the GPU, so sampling never waits for device work.
    host_fields = (
        "trawl_counts",
        "trawl_identity",
        "trawl_branch_split",
        "trawl_edges",
        "trawl_weights",
        "trawl_members",
    )

    def _host(self, batch):
        """Return CPU copies of ``host_fields`` without waiting for the device.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch, optionally carrying precomputed ``trawl_host`` arrays.

        Returns
        -------
        dict
            Mapping from field name to NumPy array.
        """
        host = getattr(batch, "trawl_host", None)
        if host is None:
            host = {
                key: batch[key].detach().cpu().numpy()
                for key in self.host_fields
                if key in batch
            }
        return host

    def _graph_structure(self, edges, weights, count, first, split):
        """Return cached sampler inputs for one graph, keyed by its contents.

        Parameters
        ----------
        edges : numpy.ndarray
            Transition edges of shape ``[edges, 3]`` (source, target,
            relation).
        weights : numpy.ndarray
            Transition weights of shape ``[edges]``.
        count : int
            Number of walk states in the graph.
        first : int
            Number of rank-0 states, excluded from historical sampling.
        split : int
            Lifting branch split index (0 when unused).

        Returns
        -------
        key : bytes
            Content hash of the graph structure.
        structure : tuple or scipy.sparse.csr_matrix
            Neighbor lists for historical sampling, otherwise a sparse
            transition matrix.
        """
        digest = hashlib.blake2b(digest_size=16)
        for part in (edges, weights, np.array([count, first, split])):
            digest.update(np.ascontiguousarray(part).tobytes())
        key = digest.digest()
        cached = self._structure_cache.get(key)
        if cached is not None:
            self._structure_cache.move_to_end(key)
            return key, cached
        if self.sampling_protocol == "historical":
            neighbors = [[] for _ in range(count - first)]
            probabilities = [[] for _ in range(count - first)]
            for (source, target, _), weight in zip(
                edges, weights, strict=True
            ):
                neighbors[source - first].append(int(target - first))
                probabilities[source - first].append(float(weight))
            structure = (
                neighbors,
                probabilities,
                prepare_walk_rows(neighbors, probabilities),
            )
        else:
            structure = sp.csr_matrix(
                (weights, (edges[:, 0], edges[:, 1])), shape=(count, count)
            )
        self._structure_cache[key] = structure
        if len(self._structure_cache) > self.cache_size:
            self._structure_cache.popitem(last=False)
        return key, structure

    def _sample(self, key, structure, seed, first, split):
        """Sample one view; evaluation draws are cached (their seeds are fixed).

        Parameters
        ----------
        key : bytes
            Content hash of the graph structure.
        structure : tuple or scipy.sparse.csr_matrix
            Sampler inputs returned by ``_graph_structure``.
        seed : int
            Sampling seed.
        first : int
            Number of rank-0 states, offset added to historical paths.
        split : int
            Lifting branch split index (0 when unused).

        Returns
        -------
        numpy.ndarray
            Walk state indices of shape ``[walk, time]``.
        """
        cache_key = (key, seed)
        if not self.training:
            cached = self._eval_walk_cache.get(cache_key)
            if cached is not None:
                return cached
        if self.sampling_protocol == "historical":
            neighbors, probabilities, rows = structure
            settings = {
                name: value
                for name, value in self.walks.items()
                if name != "guidance"
            }
            if split > 0:
                settings["branch_split"] = split
            paths = (
                sample_neighbors(
                    neighbors, probabilities, seed=seed, rows=rows, **settings
                )
                + first
            )
        else:
            paths = sample_walks(structure, seed=seed, **self.walks)
        if not self.training:
            self._eval_walk_cache[cache_key] = paths
            if len(self._eval_walk_cache) > self.cache_size:
                self._eval_walk_cache.popitem(last=False)
        return paths

    def _walk_inputs(self, batch, views=None):
        """Sample walks and gather their state features.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by the TRAWL transform.
        views : int, optional
            Number of walk views per graph. If None, one view is used in
            training and ``eval_views`` otherwise (default: None).

        Returns
        -------
        signals : torch.Tensor
            State features along walks, shape ``[walk, time, signal_dim]``.
        positions : torch.Tensor
            Encodings along walks, shape ``[walk, time, pe_dim]``.
        graph_ids : torch.Tensor
            Graph index of each walk.
        paths : torch.Tensor
            Batch-level state indices of shape ``[walk, time]``.
        info : dict
            Host-side bookkeeping (paths and per-graph/per-view walk counts).
        """
        if views is None:
            views = 1 if self.training else self.eval_views
        host = self._host(batch)
        sizes = host["trawl_counts"]
        counts = sizes.sum(axis=1).tolist()
        identities = host["trawl_identity"].tolist()
        splits = (
            host["trawl_branch_split"].tolist()
            if "trawl_branch_split" in host
            else [0] * len(counts)
        )
        slices = getattr(batch, "_slice_dict", {}).get(
            "trawl_edges", [0, len(host["trawl_edges"])]
        )
        step = self.next_sampling_step() if self.training else 0
        if self.walk_refresh != "train":
            step = 0
        host_paths, view_lengths = [], []
        offset = 0
        for graph_id, count in enumerate(counts):
            start, stop = int(slices[graph_id]), int(slices[graph_id + 1])
            first, split = int(sizes[graph_id, 0]), int(splits[graph_id])
            key, structure = self._graph_structure(
                host["trawl_edges"][start:stop],
                host["trawl_weights"][start:stop],
                count,
                first,
                split,
            )
            lengths = []
            for view in range(views):
                seed = walk_seed(self.seed, identities[graph_id], step, view)
                if self.sampling_protocol == "historical":
                    context = self.sampling_context
                    base = self.seed
                    if self.training:
                        base += (
                            context["epoch"] * 1000003
                            + context["batch"] * 997
                            + context.get("microbatch", 0) * 131
                            + context.get("seed_offset", 0)
                            + context["rank"] * 17
                        )
                    else:
                        base += 10 if context["stage"] == "Validation" else 20
                    seed = int(
                        base
                        + identities[graph_id] * 1000003
                        + graph_id * 9176
                        + view * 97
                    )
                paths = self._sample(key, structure, seed, first, split)
                if not len(paths):
                    # Historical inputs reject empty cell graphs at preprocessing.
                    raise ValueError(
                        "Continuous walk input has no active states"
                    )
                host_paths.append(np.asarray(paths, dtype=np.int64) + offset)
                lengths.append(len(paths))
            view_lengths.append(lengths)
            offset += count
        host_paths = np.concatenate(host_paths)
        walk_counts = [sum(lengths) for lengths in view_lengths]
        host_graph_ids = np.repeat(np.arange(len(counts)), walk_counts)
        device = batch.trawl_pe.device
        paths = upload(host_paths, device)
        graph_ids = upload(host_graph_ids, device)
        info = {
            "paths": host_paths,
            "walk_counts": walk_counts,
            "view_lengths": view_lengths,
            "sizes": sizes,
            "total": int(sizes.sum()),
        }
        return (
            batch.trawl_continuous[paths],
            batch.trawl_pe[paths],
            graph_ids,
            paths,
            info,
        )

    def encode_walks(self, signals, positions, mask_ratio=0.0):
        """Encode walks with the sequence layers.

        Parameters
        ----------
        signals : torch.Tensor
            State features of shape ``[walk, time, signal_dim]``.
        positions : torch.Tensor
            Encodings of shape ``[walk, time, pe_dim]``.
        mask_ratio : float, optional
            Fraction of walk steps masked during training (default: 0.0).

        Returns
        -------
        tokens : torch.Tensor
            Encoded states of shape ``[walk, time, hidden_dim]``.
        pooled : torch.Tensor
            Pooled walk embeddings of shape ``[walk, output_dim]``.
        """
        if mask_ratio and self.training:
            keep = (
                torch.rand(*signals.shape[:2], 1, device=signals.device)
                >= mask_ratio
            ).to(signals)
            signals, positions = signals * keep, positions * keep
        tokens = self.continuous_input(signals, positions)
        for layer in self.encoders[0]:
            if self.checkpoint_layers and self.training:
                from torch.utils.checkpoint import checkpoint

                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        tokens = self.output_norm(tokens)
        return tokens, self.dropout(self._pool(tokens))

    def reconstruction_loss(self, batch, mask_ratio=0.15):
        """Compute a masked walk-summary reconstruction loss.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by the TRAWL transform.
        mask_ratio : float, optional
            Fraction of walk steps masked before encoding (default: 0.15).

        Returns
        -------
        torch.Tensor
            Mean squared reconstruction error.
        """
        # Historical SSL validation scored one walk view, independent of TTA.
        signals, positions, _, _, _ = self._walk_inputs(batch, views=1)
        _, pooled = self.encode_walks(
            signals, positions, mask_ratio=mask_ratio
        )
        delta = (
            (signals[:, 1:] - signals[:, :-1]).abs().mean(dim=1)
            if signals.shape[1] > 1
            else torch.zeros_like(signals[:, 0])
        )
        targets = torch.cat(
            (signals.mean(dim=1), positions.mean(dim=1), delta), dim=-1
        )
        return F.mse_loss(self.reconstruction_decoder(pooled), targets)

    def forward(self, batch):
        """Sample and encode walks for every graph in the batch.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by the TRAWL transform.

        Returns
        -------
        dict
            Node features ``x_0`` and ``batch_0``, graph and walk embeddings
            with their indices, labels, readout metadata and, when enabled,
            branch walk weights and the auxiliary reconstruction loss.
        """
        signals, positions, graph_ids, paths, info = self._walk_inputs(batch)
        tokens, pooled = self.encode_walks(signals, positions)
        sizes, device = info["sizes"], pooled.device
        count, total = len(sizes), info["total"]
        graphs = pooled.new_zeros(count, self.output_dim).index_add(
            0, graph_ids, pooled
        )
        counts = upload(np.asarray(info["walk_counts"]), device).to(pooled)
        graphs = graphs / counts[:, None].clamp_min(1)
        cells = tokens.new_zeros(total, self.hidden_dim).index_add(
            0, paths.flatten(), tokens.flatten(0, 1)
        )
        visits = upload(
            np.bincount(info["paths"].ravel(), minlength=total), device
        ).to(tokens)[:, None]
        cells = cells / visits.clamp_min(1)
        # Rank-0 outputs are contextual features of their incident walk states.
        nodes, node_batches = [], []
        host_members = self._host(batch)["trawl_members"]
        member_slices = getattr(batch, "_slice_dict", {}).get(
            "trawl_members", [0, len(host_members)]
        )
        offset = 0
        for graph_id, graph_sizes in enumerate(sizes.tolist()):
            start, stop = (
                int(member_slices[graph_id]),
                int(member_slices[graph_id + 1]),
            )
            members = batch.trawl_members[start:stop]
            node = cells.new_zeros(graph_sizes[0], self.hidden_dim)
            if stop > start:
                node.index_add_(
                    0, members[:, 0], cells[members[:, 1] + offset]
                )
                occurrences = np.bincount(
                    host_members[start:stop, 0], minlength=graph_sizes[0]
                )
                node = node / upload(occurrences, device).to(node)[
                    :, None
                ].clamp_min(1)
            nodes.append(node)
            node_batches.append(
                torch.full(
                    (graph_sizes[0],),
                    graph_id,
                    dtype=torch.long,
                    device=device,
                )
            )
            offset += sum(graph_sizes)
        output = {
            "x_0": torch.cat(nodes),
            "batch_0": torch.cat(node_batches),
            "graph_embedding": graphs,
            "walk_embedding": pooled,
            "walk_batch": graph_ids,
            "walk_counts": info["walk_counts"],
            "labels": batch.get("y"),
            "num_views": 1 if self.training else self.eval_views,
            "walk_readout_compatible": True,
        }
        views = output["num_views"]
        output["walk_view"] = upload(
            np.concatenate(
                [
                    np.repeat(np.arange(views), walks // views)
                    for walks in info["walk_counts"]
                ]
            ),
            device,
        )
        if self.branch_gate is not None:
            walk_weights = pooled.new_zeros(len(pooled))
            walks_per_view = self.walks.get("k", 32) * (
                2 if self.walks.get("reverse", False) else 1
            )
            splits = self._host(batch)["trawl_branch_split"].tolist()
            offset = start = 0
            for graph_id, graph_sizes in enumerate(sizes.tolist()):
                split = int(splits[graph_id])
                if split < 1:
                    raise ValueError(
                        "Branch gating requires a gated historical lifting"
                    )
                for view in range(views):
                    selected = np.arange(
                        start + view * walks_per_view,
                        min(
                            start + (view + 1) * walks_per_view,
                            start + info["walk_counts"][graph_id],
                        ),
                    )
                    champion = (
                        info["paths"][selected, 0]
                        < offset + graph_sizes[0] + split
                    )
                    first = upload(selected[champion], device)
                    second = upload(selected[~champion], device)
                    if not len(first) or not len(second):
                        raise ValueError(
                            "Branch gating needs walks from both lifting branches"
                        )
                    gate = (
                        self.branch_gate(
                            torch.cat(
                                (pooled[first].mean(0), pooled[second].mean(0))
                            )
                        )
                        .sigmoid()
                        .squeeze()
                        .to(walk_weights)
                    )
                    walk_weights[first] = gate / len(first) / views
                    walk_weights[second] = (1 - gate) / len(second) / views
                start += info["walk_counts"][graph_id]
                offset += sum(graph_sizes)
            output["walk_weights"] = walk_weights
            output["graph_embedding"] = torch.zeros_like(graphs).index_add(
                0, graph_ids, pooled * walk_weights[:, None]
            )
        epoch = self.sampling_context["epoch"]
        if (
            self.training
            and self.auxiliary.get("weight", 0) > 0
            and epoch <= self.auxiliary.get("decay_epochs", 20)
        ):
            # Reuse the supervised paths, including their sampling RNG policy.
            reconstructed = pooled
            if self.auxiliary.get("mask_probability", 0):
                _, reconstructed = self.encode_walks(
                    signals, positions, self.auxiliary["mask_probability"]
                )
            raw = batch.trawl_continuous[paths]
            delta = (
                (raw[:, 1:] - raw[:, :-1]).abs().mean(1)
                if raw.shape[1] > 1
                else torch.zeros_like(raw[:, 0])
            )
            target = torch.cat((raw.mean(1), positions.mean(1), delta), dim=-1)
            objective = (
                F.smooth_l1_loss
                if self.auxiliary.get("objective", "smooth_l1") == "smooth_l1"
                else F.mse_loss
            )
            output["reconstruction_loss"] = objective(
                self.reconstruction_decoder(reconstructed), target
            )
            output["epoch"] = epoch
        return output
