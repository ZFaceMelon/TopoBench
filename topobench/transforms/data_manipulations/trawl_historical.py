"""Explicit historical bond/ring preprocessing profile for recipe experiments."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

from topobench.data.utils.trawl import legacy_features


class HistoricalCellTransform(BaseTransform):
    """Original weighted overlap graph, four state signals and optional rich PSE.

    This is an opt-in compatibility lifting, never the base TRAWL default.
    Split-dependent RWSE seeds are applied after the dataset is partitioned.
    Source node features/edges are retained so the split hook can rebuild PSE.

    Parameters
    ----------
    lift : bool, optional
        If True, lift bonds to higher cells such as rings; otherwise only
        bonds are states (default: True).
    seed : int, optional
        Base seed for the empirical RWSE (default: 42).
    rwse_samples : int, optional
        Number of sampled walks for the empirical RWSE (default: 8).
    rw_steps : int, optional
        Number of RWSE steps (default: 8).
    rich_features : bool, optional
        If True, append rich molecular structural encodings (default: False).
    attr_dim : int, optional
        Atom attribute dimension passed to the feature helpers (default: 0).
    split_seeds : bool, optional
        If True, defer building to ``prepare_split`` with split-dependent
        seeds; otherwise build in ``forward`` (default: True).
    guidance : dict, optional
        Keyword arguments for ``laplacian_guided_neigh_probs`` (default:
        None, meaning ``{"gamma": 0.2, "diffusion_t": 0.1}``).
    spectral : dict, optional
        Keyword arguments for ``hasse_spectral_pse`` (default: None).
    profile : str, optional
        Feature implementation, ``"classic"`` or ``"extended"``
        (default: "classic").
    lifting : str, optional
        Cell lifting: ``"champion"``, ``"star"``, ``"short_cycles"`` or
        ``"multiscale"``, or with the extended profile also
        ``"cycle18_adj3"``, ``"merged"`` or ``"gated"`` (default: "champion").
    cycle_mix : float, optional
        Cycle mixing weight for the ``"merged"`` lifting (default: 0.35).
    **kwargs : dict
        Ignored extra options.
    """

    def __init__(
        self,
        lift=True,
        seed=42,
        rwse_samples=8,
        rw_steps=8,
        rich_features=False,
        attr_dim=0,
        split_seeds=True,
        guidance=None,
        spectral=None,
        profile="classic",
        lifting="champion",
        cycle_mix=0.35,
        **kwargs,
    ):
        self.lift, self.seed = lift, int(seed)
        self.rwse_samples, self.rw_steps = rwse_samples, rw_steps
        self.rich_features, self.attr_dim = rich_features, attr_dim
        self.split_seeds = split_seeds
        self.guidance = dict(guidance or {"gamma": 0.2, "diffusion_t": 0.1})
        self.spectral = dict(spectral or {})
        if profile not in {"classic", "extended"}:
            raise ValueError("Historical profile must be classic or extended")
        if profile == "classic" and lifting != "champion":
            raise ValueError(
                "Alternative historical liftings require the extended profile"
            )
        self.profile, self.lifting, self.cycle_mix = (
            profile,
            lifting,
            cycle_mix,
        )

    def forward(self, data):
        """Build features now, or defer them to ``prepare_split``.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Molecular graph.

        Returns
        -------
        torch_geometric.data.Data
            The unchanged input when ``split_seeds`` is set, otherwise the
            built graph.
        """
        return data if self.split_seeds else self.build(data, self.seed)

    def prepare_split(self, dataset, split_offset, cache_dir=None):
        """Build split-seeded features, reusing an exact on-disk copy.

        The cache key covers the transform settings, the split, every input
        graph's tensors, the feature implementation and numerical library
        versions, so a cached split is the identical result of rebuilding.

        Parameters
        ----------
        dataset : Any
            Split dataset whose ``data_lst`` is replaced in place; ignored
            when None.
        split_offset : int
            Offset added to ``seed`` for this split.
        cache_dir : str or pathlib.Path, optional
            Directory for cached splits; caching is disabled when None
            (default: None).
        """
        if not self.split_seeds or dataset is None:
            return
        path = (
            self._split_cache_path(dataset.data_lst, split_offset, cache_dir)
            if cache_dir is not None
            else None
        )
        if path is not None and path.is_file():
            dataset.data_lst = torch.load(path, weights_only=False)
            return
        dataset.data_lst = [
            self.build(data, self.seed + split_offset + i, identity=i)
            for i, data in enumerate(dataset.data_lst)
        ]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Unique per process: concurrent runs may build the same split.
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(dataset.data_lst, temporary)
            temporary.replace(path)

    def _split_cache_path(self, data_list, split_offset, cache_dir):
        """Return the cache file path for a split, keyed by its content hash.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Input graphs of the split.
        split_offset : int
            Offset added to ``seed`` for this split.
        cache_dir : str or pathlib.Path
            Root cache directory.

        Returns
        -------
        pathlib.Path
            Path of the cached split file.
        """
        import networkx
        import scipy

        from topobench.data.utils import trawl as trawl_utils

        digest = hashlib.sha256()
        settings = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_")
        }
        digest.update(
            json.dumps(
                {
                    "settings": settings,
                    "split": split_offset,
                    "versions": [
                        np.__version__,
                        scipy.__version__,
                        networkx.__version__,
                        torch.__version__,
                    ],
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
        sources = [Path(__file__)] + sorted(
            Path(trawl_utils.__file__).parent.glob("*.py")
        )
        for source in sources:
            digest.update(source.read_bytes())
        for data in data_list:
            for key in sorted(data.keys()):
                value = data[key]
                if torch.is_tensor(value):
                    digest.update(key.encode())
                    digest.update(
                        str((value.dtype, tuple(value.shape))).encode()
                    )
                    digest.update(
                        value.detach().cpu().contiguous().numpy().tobytes()
                    )
        return (
            Path(cache_dir) / "trawl_historical" / f"{digest.hexdigest()}.pt"
        )

    def build(self, data, seed, identity=0):
        """Build the historical cell states, signals and encodings for a graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Molecular graph; it is cloned, not modified.
        seed : int
            Seed for the empirical RWSE.
        identity : int, optional
            Value stored in ``trawl_identity`` (default: 0).

        Returns
        -------
        torch_geometric.data.Data
            Copy of the input with ``trawl_*`` fields added.
        """
        features = legacy_features
        if self.profile == "extended":
            from topobench.data.utils.trawl import (
                extended_features as features,
            )

        data = data.clone()
        if data.get("x") is None and data.get("x_0") is not None:
            data.x = data.x_0
        endpoints, weights, bond_types = features.unique_undirected_bonds(data)
        branch_split = None
        if self.lift and self.lifting in {"cycle18_adj3", "merged", "gated"}:
            if self.lifting == "cycle18_adj3":
                result = features.build_cycle18_adj3(data)
            elif self.lifting == "merged":
                result = features.build_dual_champion_cycle18(
                    data, cycle_mix=self.cycle_mix
                )
            else:
                result = features.build_dual_learnable_gate(data)
                branch_split = result[-1]
            endpoints, weights, bond_types, neighbors, probabilities = result[
                :5
            ]
        elif self.lift:
            if self.lifting not in {
                "champion",
                "star",
                "short_cycles",
                "multiscale",
            }:
                raise ValueError(f"Unknown historical lifting {self.lifting}")
            if self.lifting in {"short_cycles", "multiscale"}:
                endpoints, weights, bond_types = features.lift_short_cycles(
                    data, endpoints, weights, bond_types
                )
            else:
                endpoints, weights, bond_types = features.lift_rings(
                    data, endpoints, weights, bond_types
                )
            if self.lifting in {"star", "multiscale"}:
                endpoints, weights, bond_types = features.lift_stars(
                    data, endpoints, weights, bond_types
                )
            neighbors, probabilities = features.build_sparse_dual(
                endpoints, weights
            )
        else:
            neighbors, probabilities = features.build_sparse_dual(
                endpoints, weights
            )
        if not endpoints:
            raise ValueError(
                "Historical bond/ring profile cannot represent an edgeless graph; use base TRAWL"
            )
        local = features.local_topological_pe(
            endpoints, weights, bond_types, neighbors
        )
        rwse = features.empirical_rwse(
            neighbors,
            probabilities,
            k_rwse=self.rw_steps,
            n_samples=self.rwse_samples,
            seed=seed,
        )
        parts = (
            [(0, len(endpoints))]
            if branch_split is None
            else [(0, branch_split), (branch_split, len(endpoints))]
        )
        spectral_parts, guided = [], []
        for lo, hi in parts:
            part_neighbors = [
                [j - lo for j in row] for row in neighbors[lo:hi]
            ]
            spectral_parts.append(
                features.hasse_spectral_pse(
                    part_neighbors,
                    probabilities[lo:hi],
                    q_vec=local[lo:hi, 1],
                    **self.spectral,
                )
            )
            guided.extend(
                features.laplacian_guided_neigh_probs(
                    part_neighbors, probabilities[lo:hi], **self.guidance
                )
            )
        spectral = np.concatenate(spectral_parts, axis=0)
        atom_types = features.extract_atom_types(data, attr_dim=self.attr_dim)
        atoms, atom_mask = features.pack_hyperedge_atoms(endpoints, atom_types)
        tokens = np.asarray(
            [
                features.hyperedge_token_from_type(len(cell), token)
                for cell, token in zip(endpoints, bond_types, strict=True)
            ]
        )
        num_atoms = atom_mask.sum(-1, keepdims=True).clip(min=1)
        signal = np.concatenate(
            (
                (atoms * atom_mask).sum(-1, keepdims=True) / num_atoms / 63.0,
                num_atoms / features.MAX_HYPEREDGE_ATOMS,
                tokens[:, None] / (features.VOCAB_SIZE - 1),
                atom_mask.mean(-1, keepdims=True),
            ),
            axis=1,
        ).astype(np.float32)
        pe = np.concatenate((local, rwse, spectral), axis=1)
        if self.rich_features:
            pe = np.concatenate(
                (
                    pe,
                    features.rich_molecular_structural_pe(
                        data, endpoints, atom_types, attr_dim=self.attr_dim
                    ),
                ),
                axis=1,
            )
        n = int(data.num_nodes)
        # Keep original neighbor order and double precision for replay sampling.
        edge_rows, edge_weights = [], []
        for source, (targets, probs) in enumerate(
            zip(neighbors, guided, strict=True)
        ):
            for target, probability in zip(targets, probs, strict=True):
                edge_rows.append((source + n, target + n, 0))
                edge_weights.append(probability)
        data.trawl_edges = torch.tensor(edge_rows, dtype=torch.long)
        data.trawl_weights = torch.tensor(edge_weights, dtype=torch.float64)
        data.trawl_counts = torch.tensor([[n, len(endpoints)]])
        data.trawl_relations = torch.tensor([1])
        data.trawl_identity = torch.tensor([identity])
        data.trawl_branch_split = torch.tensor(
            [-1 if branch_split is None else branch_split]
        )
        data.trawl_continuous = torch.from_numpy(
            np.concatenate((np.zeros((n, 4), dtype=np.float32), signal))
        )
        data.trawl_pe = torch.from_numpy(
            np.concatenate((np.zeros((n, pe.shape[1]), dtype=np.float32), pe))
        )
        data.trawl_members = torch.tensor(
            [
                (vertex, cell + n)
                for cell, vertices in enumerate(endpoints)
                for vertex in vertices
            ],
            dtype=torch.long,
        )
        data.trawl_tokens = torch.from_numpy(
            np.concatenate((np.zeros(n, dtype=np.int64), tokens))
        )
        data.trawl_atoms = torch.from_numpy(
            np.concatenate(
                (
                    np.zeros(
                        (n, features.MAX_HYPEREDGE_ATOMS), dtype=np.int64
                    ),
                    atoms,
                )
            )
        )
        data.trawl_atom_mask = torch.from_numpy(
            np.concatenate(
                (
                    np.zeros(
                        (n, features.MAX_HYPEREDGE_ATOMS), dtype=np.float32
                    ),
                    atom_mask,
                )
            )
        )
        return data
