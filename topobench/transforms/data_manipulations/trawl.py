"""Adapt lifting outputs to relation-labelled walk graphs without losing ranks.

All connectivity stored by this transform uses graph-local indices. PyG must
concatenate, not increment, these fields; their names deliberately avoid
``index`` and ``x_`` (TopoBench's collator assigns meaning to those strings).
"""

import hashlib
import re

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from topobench.transforms.data_manipulations.combined_feature_encodings import (
    FE_ENCODINGS,
    CombinedFEs,
)
from topobench.transforms.data_manipulations.combined_positional_and_structural_encodings import (
    CombinedPSEs,
)


def hopse_encodings(matrix, local=True, features=None, **parameters):
    """HOPSE encodings of one relation, computed on its active states.

    Edge weights and direction are dropped: the HOPSE encodings read an
    unweighted ``edge_index``. Inactive states receive zeros.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix of one relation.
    local : bool, optional
        Whether to include log-degree and neighbor-count channels
        (default: True).
    features : np.ndarray, optional
        State input features ``(n, f)``; required by the HOPSE feature
        encodings (``KHopFE``, ``HKFE``, ``PPRFE``) (default: None).
    **parameters : dict
        Keyword arguments per HOPSE encoding, keyed by its name (``LapPE``,
        ``RWSE``, ``ElectrostaticPE``, ``HKdiagSE``, ``KHopFE``, ``HKFE``,
        ``PPRFE``), in output order.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(n, d)``.
    """
    matrix = matrix.tocsr()
    n = matrix.shape[0]
    active = np.flatnonzero(
        np.asarray((matrix + matrix.T).sum(axis=1)).ravel() > 0
    )
    parts = []
    if local:
        degree = np.asarray(matrix.sum(axis=1)).ravel()
        parts.append(
            np.stack(
                (np.log1p(degree), matrix.getnnz(axis=1).astype(float)), axis=1
            )
        )
    if parameters:
        options = {
            name: {**dict(value or {}), "concat_to_x": False}
            for name, value in parameters.items()
        }
        fes = [name for name in parameters if name in FE_ENCODINGS]
        pses = [name for name in parameters if name not in FE_ENCODINGS]
        if fes and features is None:
            raise ValueError("HOPSE feature encodings need state features")

        def encode(edge_index, num_nodes, x):
            """Run the requested PSEs and FEs on one edge index.

            Parameters
            ----------
            edge_index : torch.Tensor
                Edge index of shape ``[2, edges]``.
            num_nodes : int
                Number of states.
            x : torch.Tensor or None
                State features, used by the FEs.

            Returns
            -------
            np.ndarray
                Encodings in ``parameters`` order, ``(num_nodes, d)``.
            """
            data = Data(edge_index=edge_index, num_nodes=num_nodes, x=x)
            if pses:
                data = CombinedPSEs(
                    encodings=pses, parameters={k: options[k] for k in pses}
                )(data)
            if fes:
                data = CombinedFEs(
                    encodings=fes, parameters={k: options[k] for k in fes}
                )(data)
            return torch.cat(
                [data[name].double().cpu() for name in parameters], dim=1
            ).numpy()

        def state_x(rows):
            """Features of the given states, if the transform has them.

            Parameters
            ----------
            rows : int or np.ndarray
                State indices, or a count of all-ones rows for the probe.

            Returns
            -------
            torch.Tensor or None
                Float features, or ``None`` without input features.
            """
            if features is None:
                return None
            if isinstance(rows, int):
                return torch.ones(rows, features.shape[1])
            return torch.as_tensor(features[rows], dtype=torch.float32)

        if len(active):
            sub = matrix[active][:, active].tocoo()
            encoded = encode(
                torch.tensor(np.stack((sub.row, sub.col)), dtype=torch.long),
                len(active),
                state_x(active),
            )
        else:
            # Some encodings return a different width on empty graphs; take
            # the regular width from a two-state probe graph instead.
            probe = encode(torch.tensor([[0, 1], [1, 0]]), 2, state_x(2))
            encoded = np.zeros((0, probe.shape[1]))
        full = np.zeros((n, encoded.shape[1]))
        full[active] = encoded
        parts.append(full)
    result = np.concatenate(parts, axis=1) if parts else np.zeros((n, 0))
    result[np.setdiff1d(np.arange(n), active)] = 0
    return np.nan_to_num(result).astype(np.float32)


def as_csr(value):
    """Convert dense or any PyTorch sparse layout to unsigned CPU CSR.

    Parameters
    ----------
    value : torch.Tensor or array_like or scipy.sparse.spmatrix
        Dense or sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Float64 CSR matrix of absolute values.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.layout != torch.strided:
            value = value.to_sparse_coo().coalesce()
            row, col = value.indices().numpy()
            return sp.csr_matrix(
                (np.abs(value.values().numpy()), (row, col)), value.shape
            )
        value = value.numpy()
    return abs(sp.csr_matrix(value, dtype=np.float64))


def signed_csr(value):
    """Convert dense or sparse PyTorch/array input to signed CPU CSR.

    Parameters
    ----------
    value : torch.Tensor or array_like or scipy.sparse.spmatrix
        Dense or sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Float64 CSR matrix keeping the entries' signs.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.layout != torch.strided:
            value = value.to_sparse_coo().coalesce()
            row, col = value.indices().numpy()
            return sp.csr_matrix(
                (value.values().numpy().astype(np.float64), (row, col)),
                value.shape,
            )
        value = value.numpy()
    return sp.csr_matrix(value, dtype=np.float64)


def canonical_signs(incidence, faces, cells):
    """Boundary signs from the sorted-vertex orientation of each simplex.

    The coefficient of face ``i`` (the face omitting the ``i``-th smallest
    vertex) in the boundary of a simplex is ``(-1) ** i``.

    Parameters
    ----------
    incidence : scipy.sparse.spmatrix
        Unsigned ``(faces, cells)`` incidence pattern.
    faces : scipy.sparse.spmatrix
        Vertex membership of the faces, ``(vertices, faces)``.
    cells : scipy.sparse.spmatrix
        Vertex membership of the cells, ``(vertices, cells)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Signed incidence with the same pattern.
    """
    coo = incidence.tocoo()
    faces, cells = faces.tocsc(), cells.tocsc()
    values = np.empty(coo.nnz)
    for entry, (face, cell) in enumerate(zip(coo.row, coo.col, strict=True)):
        vertices = np.sort(cells.indices[cells.indptr[cell] : cells.indptr[cell + 1]])
        kept = faces.indices[faces.indptr[face] : faces.indptr[face + 1]]
        missing = np.flatnonzero(~np.isin(vertices, kept))
        if len(missing) != 1:
            raise ValueError("Signed incidences need simplicial faces")
        values[entry] = (-1.0) ** int(missing[0])
    return sp.csr_matrix((values, (coo.row, coo.col)), shape=incidence.shape)


def relation_matrix(name, counts, incidences, data):
    """Resolve TopoTune hop/direction/relation/rank syntax.

    Adjacency means shared upper cells; coadjacency means shared lower cells.
    Multi-hop incidence is the product along consecutive ranks. Explicit
    lifting-provided relation matrices take precedence over derived ones.

    Parameters
    ----------
    name : str
        Neighborhood name such as ``"up_incidence-0"`` or
        ``"2-down_adjacency-2"``.
    counts : list of int
        Number of cells per rank.
    incidences : dict
        CSR incidence matrix per rank ``r >= 1``, shaped
        ``(counts[r - 1], counts[r])``.
    data : torch_geometric.data.Data
        Lifted data, checked for an explicit matrix stored under ``name``.

    Returns
    -------
    tuple
        ``(rank, target, matrix)``: source rank, target rank and the CSR
        relation matrix of shape ``(counts[rank], counts[target])``.
    """
    match = re.fullmatch(
        r"(?:(\d+)-)?(up|down)_(adjacency|incidence)-(\d+)", name
    )
    if match is None:
        raise ValueError(f"Invalid TRAWL neighborhood: {name!r}")
    hops, direction, kind, rank = match.groups()
    hops, rank = int(hops or 1), int(rank)
    if hops < 1 or rank >= len(counts):
        raise ValueError(f"Neighborhood outside configured ranks: {name}")
    other = rank + (hops if direction == "up" else -hops)
    target = other if kind == "incidence" else rank
    shape = (counts[rank], counts[target] if 0 <= target < len(counts) else 0)
    if not 0 <= other < len(counts):
        return rank, target, sp.csr_matrix(shape)
    if name in data:
        matrix = as_csr(data[name])
        # TopoBench stores incidence as target x source, unlike adjacency.
        if kind == "incidence":
            matrix = matrix.T.tocsr()
    else:
        matrix = sp.eye(counts[min(rank, other)], format="csr")
        for level in range(min(rank, other) + 1, max(rank, other) + 1):
            matrix = matrix @ incidences[level]
        if direction == "down":
            matrix = matrix.T.tocsr()
        if kind == "adjacency":
            matrix = (matrix @ matrix.T).tocsr()
            matrix.setdiag(0)
            matrix.eliminate_zeros()
    if matrix.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {matrix.shape}")
    return rank, target, matrix


class TRAWLTransform(BaseTransform):
    """Prepare any rank/incidence lifting for TRAWL.

    Plain graphs become rank 0 states; ``incidence_hyperedges`` is adapted to
    rank 1. Missing higher ranks are empty, not fabricated. Featureless cells
    receive a constant feature, while existing lifted features are preserved.

    Parameters
    ----------
    max_rank : int, optional
        Highest cell rank to include (default: 2).
    graph : str, optional
        Walk graph: ``"hasse"``, ``"augmented_hasse"`` or ``"cell_overlap"``
        (default: "augmented_hasse").
    neighborhoods : list of str, optional
        Relation names resolved by ``relation_matrix``; defaults to
        ``up_incidence-r`` for every rank below ``max_rank`` (default: None).
    bidirectional : bool, optional
        If True, symmetrize every relation (default: True).
    overlap_ranks : list of int, optional
        Ranks included in the ``"cell_overlap"`` graph; defaults to
        ``1..max_rank`` (default: None).
    encodings : dict, optional
        Keyword arguments for ``hopse_encodings`` (default: None, meaning
        local channels only).
    encoding_scope : str, optional
        Compute encodings per relation (``"separate"``) or on their union
        (``"union"``) (default: "separate").
    color_key : str, optional
        Data attribute holding one nonnegative color ID per cell; defaults
        to colouring cells by rank (default: None).
    color_refinement : bool, optional
        If True, split encoding graphs by unordered color pair
        (default: False).
    num_colors : int, optional
        Number of color IDs; defaults to ``max_rank + 1`` (default: None).
    signs : bool, optional
        If True, store ``trawl_signs``: the signed boundary coefficient
        (+1/-1) of every walk edge that is an immediate incidence; for two
        same-rank cells sharing exactly one face (down adjacency), their
        relative orientation ``-s(f, a) * s(f, b)``; 0 otherwise. Signs come from the lifted incidences if they are signed,
        else from the sorted-vertex orientation (default: False).
    pivots : bool, optional
        If True, store ``trawl_pivots``: for every walk edge of an adjacency
        relation, the graph-local state index of the cell it pivots across
        (the shared face for ``down_adjacency``, the shared coface for
        ``up_adjacency``); -1 for incidence edges or ambiguous pairs
        (default: False).
    cell_vertices : bool, optional
        If True, store ``trawl_cell_vertices``: for every state, the
        graph-local indices of its vertices (rank 0 states list themselves),
        padded with -1 to ``cell_vertex_width`` columns. Unsigned structure
        only (default: False).
    rich_features : bool, optional
        If True, append to ``trawl_pe`` the 40 old-TRAWL molecular context
        features of every state's vertex set (degree statistics and bins,
        neighbor atom-type histogram, ring-size membership, clustering and
        k-core, atom-type context, graph size/degree/density; see
        ``trawl_graph_features``). Opt-in, for graph datasets whose old
        TRAWL recipes used them (NCI1) (default: False).
    attr_dim : int, optional
        Leading continuous node-attribute columns of ``x`` (e.g. 1 for
        PROTEINS with ``use_node_attr``); skipped when reading atom types
        and summarized as 6 extra rich features (default: 0).
    attr_scaling : str, optional
        Rescale the first ``attr_dim`` (continuous attribute) columns of the
        state inputs: ``"signed_log1p"`` maps a vertex attribute ``a`` to
        ``sign(a) * log1p(|a|)`` and rebuilds higher-rank columns as the
        ProjectionSum of the rescaled vertex values (edges sum their
        vertices, cells sum their edges). Dataset-free, so no split
        statistics leak. None leaves inputs unchanged (default: None).
    cell_vertex_width : int, optional
        Columns of ``trawl_cell_vertices``: the largest number of vertices of
        any cell. None means ``max_rank + 1`` (simplices); cell complexes
        need the largest cell size, e.g. ``max_cell_length`` of a cycle
        lifting. A larger cell raises an error (default: None).
    **kwargs : dict
        Ignored extra options.
    """

    def __init__(
        self,
        max_rank=2,
        graph="augmented_hasse",
        neighborhoods=None,
        bidirectional=True,
        overlap_ranks=None,
        encodings=None,
        encoding_scope="separate",
        color_key=None,
        color_refinement=False,
        num_colors=None,
        signs=False,
        pivots=False,
        cell_vertices=False,
        cell_vertex_width=None,
        rich_features=False,
        attr_dim=0,
        attr_scaling=None,
        **kwargs,
    ):
        if graph not in {"hasse", "augmented_hasse", "cell_overlap"}:
            raise ValueError(f"Unknown walk graph {graph}")
        if encoding_scope not in {"separate", "union"}:
            raise ValueError("encoding_scope must be separate or union")
        self.max_rank = int(max_rank)
        if self.max_rank < 0:
            raise ValueError("max_rank must be nonnegative")
        self.graph = graph
        self.neighborhoods = list(
            neighborhoods
            or [f"up_incidence-{r}" for r in range(self.max_rank)]
            or ["up_adjacency-0"]
        )
        self.bidirectional = bidirectional
        self.overlap_ranks = list(overlap_ranks or range(1, self.max_rank + 1))
        self.encodings = dict(encodings or {"local": True})
        self.encoding_scope = encoding_scope
        self.color_key = color_key
        self.color_refinement = color_refinement
        self.num_colors = int(num_colors or (self.max_rank + 1))
        self.signs = bool(signs)
        self.pivots = bool(pivots)
        self.cell_vertices = bool(cell_vertices)
        self.cell_vertex_width = (
            self.max_rank + 1
            if cell_vertex_width is None
            else int(cell_vertex_width)
        )
        if self.cell_vertex_width < 1:
            raise ValueError("cell_vertex_width must be positive")
        self.rich_features = bool(rich_features)
        self.attr_dim = int(attr_dim)
        if attr_scaling not in {None, "signed_log1p"}:
            raise ValueError(f"Unknown attr_scaling {attr_scaling}")
        self.attr_scaling = attr_scaling

    def finalize_dataset(self, data_list):
        """Give absent ranks the same empty feature shape as populated ranks.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Transformed graphs, updated in place.

        Returns
        -------
        list of torch_geometric.data.Data
            The same graphs with consistent per-rank feature widths.
        """
        for rank in range(self.max_rank + 1):
            key = f"trawl_signal_{rank}"
            widths = {
                data[key].shape[1] for data in data_list if len(data[key])
            }
            if len(widths) > 1:
                raise ValueError(
                    f"Rank-{rank} feature widths vary across graphs: {widths}"
                )
            if widths:
                width = next(iter(widths))
                for data in data_list:
                    if not len(data[key]):
                        data[key] = data[key].new_empty(0, width)
                        if f"x_{rank}" in data and not len(data[f"x_{rank}"]):
                            data[f"x_{rank}"] = data[key].clone()
        return data_list

    def _expanded(self, name, counts, incidences, data, offsets, total):
        """One relation as a ``[total, total]`` matrix over all states.

        Parameters
        ----------
        name : str
            Relation name resolved by ``relation_matrix``.
        counts : list of int
            Number of cells per rank.
        incidences : dict
            CSR incidence matrix per rank.
        data : torch_geometric.data.Data
            Lifted data.
        offsets : np.ndarray
            First state index of each rank.
        total : int
            Number of states.

        Returns
        -------
        scipy.sparse.csr_matrix
            The relation, symmetrized if ``bidirectional``.
        """
        if self.graph == "hasse" and ("adjacency" in name or name[0].isdigit()):
            raise ValueError(
                "Strict Hasse graphs only accept immediate incidence neighborhoods"
            )
        src, dst, matrix = relation_matrix(name, counts, incidences, data)
        # Native graphs have no rank-1 incidence: use their edges.
        if (
            name == "up_adjacency-0"
            and counts[1:2] in ([], [0])
            and "edge_index" in data
        ):
            edges = data.edge_index.cpu().numpy()
            matrix = sp.csr_matrix(
                (np.ones(edges.shape[1]), edges),
                shape=(counts[0], counts[0]),
            )
        coo = matrix.tocoo()
        if coo.nnz:
            expanded = sp.csr_matrix(
                (
                    coo.data,
                    (coo.row + offsets[src], coo.col + offsets[dst]),
                ),
                shape=(total, total),
            )
        else:
            expanded = sp.csr_matrix((total, total))
        if self.bidirectional:
            expanded = expanded.maximum(expanded.T)
        return expanded

    def _rich_features(self, data, counts, memberships):
        """Old-TRAWL molecular context features of every state.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Lifted graph with node features ``x`` (or ``x_0``).
        counts : list of int
            Number of cells per rank.
        memberships : dict
            Vertex membership per rank, ``(vertices, cells)``.

        Returns
        -------
        np.ndarray
            Float32 ``[states, 40 (+6 with attr_dim)]`` features.
        """
        from torch_geometric.data import Data as Graph

        from topobench.transforms.data_manipulations import (
            trawl_graph_features as features,
        )

        x = data.get("x")
        if x is None:
            x = data.get("x_0")
        if len(counts) > 1 and counts[1]:
            bonds = memberships[1].tocsc()
            pairs = np.asarray(
                [
                    bonds.indices[bonds.indptr[e] : bonds.indptr[e + 1]]
                    for e in range(counts[1])
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            edge_index = torch.from_numpy(
                np.concatenate((pairs, pairs[:, ::-1])).T.copy()
            )
        else:
            edge_index = data.get("edge_index")
            if edge_index is None:
                edge_index = torch.empty(2, 0, dtype=torch.long)
        graph = Graph(x=x, edge_index=edge_index, num_nodes=counts[0])
        endpoints = []
        for rank, count in enumerate(counts):
            members = memberships[rank].tocsc()
            endpoints.extend(
                tuple(
                    int(v)
                    for v in np.sort(
                        members.indices[
                            members.indptr[cell] : members.indptr[cell + 1]
                        ]
                    )
                )
                for cell in range(count)
            )
        atom_types = features.extract_atom_types(graph, self.attr_dim)
        return features.rich_molecular_structural_pe(
            graph, endpoints, atom_types, attr_dim=self.attr_dim
        )

    def _edge_pivots(self, edges, counts, offsets, incidences):
        """Pivot cell of each adjacency walk edge.

        Parameters
        ----------
        edges : np.ndarray
            ``(source, target, relation)`` rows of ``trawl_edges``.
        counts : list of int
            Number of cells per rank.
        offsets : np.ndarray
            First state index of each rank.
        incidences : dict
            Unsigned CSR incidence per rank ``(counts[r-1], counts[r])``.

        Returns
        -------
        np.ndarray
            Int64 graph-local pivot state per edge, -1 if none.
        """
        out = np.full(len(edges), -1, dtype=np.int64)
        for relation, name in enumerate(self.neighborhoods):
            match = re.fullmatch(r"(up|down)_adjacency-(\d+)", name)
            if match is None:
                continue
            direction, rank = match.group(1), int(match.group(2))
            level = rank if direction == "down" else rank + 1
            if not 1 <= level <= self.max_rank or not incidences[level].nnz:
                continue
            # down: faces (rank-1) of rank cells; up: cofaces (rank+1).
            matrix = incidences[level].tocsr()
            if direction == "up":
                members, pivot_rank = matrix.tocsr(), rank + 1
                members = members.T.tocsr()  # rows: cofaces, cols: cells
            else:
                members, pivot_rank = matrix, rank - 1  # rows: faces
            table = {}
            for pivot in range(members.shape[0]):
                cells = members.indices[
                    members.indptr[pivot] : members.indptr[pivot + 1]
                ]
                for a in cells:
                    for b in cells:
                        if a != b:
                            key = (int(a), int(b))
                            table[key] = -2 if key in table else pivot
            rows = np.flatnonzero(edges[:, 2] == relation)
            base = offsets[rank]
            for row in rows:
                pivot = table.get(
                    (int(edges[row, 0] - base), int(edges[row, 1] - base)), -1
                )
                if pivot >= 0:
                    out[row] = pivot + offsets[pivot_rank]
        return out

    def _edge_signs(
        self, edges, counts, offsets, incidences, raw, memberships
    ):
        """Signed boundary coefficient of each walk edge.

        Parameters
        ----------
        edges : np.ndarray
            ``(source, target, relation)`` rows of ``trawl_edges``.
        counts : list of int
            Number of cells per rank.
        offsets : np.ndarray
            First state index of each rank.
        incidences : dict
            Unsigned CSR incidence per rank.
        raw : dict
            Signed (as lifted) CSR incidence per rank, where available.
        memberships : dict
            Vertex membership per rank, ``(vertices, cells)``.

        Returns
        -------
        np.ndarray
            Int8 array with one sign per edge, 0 for non-incidence edges.
        """
        total = int(offsets[-1])
        table = sp.lil_matrix((total, total))
        for rank in range(1, self.max_rank + 1):
            pattern = incidences[rank]
            if not pattern.nnz:
                continue
            signed = raw.get(rank)
            if signed is None or not (signed.data < 0).any():
                signed = canonical_signs(
                    pattern, memberships[rank - 1], memberships[rank]
                )
            coo = signed.tocoo()
            keep = coo.data != 0
            rows = coo.row[keep] + offsets[rank - 1]
            cols = coo.col[keep] + offsets[rank]
            values = np.sign(coo.data[keep])
            table[rows, cols] = values
            table[cols, rows] = values
            # Down adjacency: cells sharing exactly one face are oppositely
            # induced on it iff their orientations agree.
            unit = signed.copy()
            unit.data = np.sign(unit.data)
            shared = (abs(unit).T @ abs(unit)).tocoo()
            product = (unit.T @ unit).tocsr()
            once = (shared.row != shared.col) & (shared.data == 1)
            rows, cols = shared.row[once], shared.col[once]
            relative = -np.asarray(product[rows, cols]).ravel()
            table[rows + offsets[rank], cols + offsets[rank]] = relative
        if not len(edges):
            return np.zeros(0, dtype=np.int8)
        table = table.tocsr()
        return (
            np.asarray(table[edges[:, 0], edges[:, 1]]).ravel().astype(np.int8)
        )

    def forward(self, data):
        """Build the TRAWL walk graph, features, colors and encodings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Lifted (or plain) graph.

        Returns
        -------
        torch_geometric.data.Data
            The input data with ``trawl_*`` fields added.
        """
        counts = [int(data.num_nodes)] + [0] * self.max_rank
        if counts[0] <= 0:
            raise ValueError("TRAWL requires at least one rank-0 cell")
        incidences, raw_incidences = {}, {}
        for rank in range(1, self.max_rank + 1):
            key = f"incidence_{rank}"
            if (
                rank == 1
                and key not in data
                and "incidence_hyperedges" in data
            ):
                key = "incidence_hyperedges"
            if key in data:
                matrix = as_csr(data[key])
                if self.signs:
                    raw_incidences[rank] = signed_csr(data[key])
                if matrix.shape[0] != counts[rank - 1]:
                    raise ValueError(f"{key} has inconsistent lower-rank size")
                counts[rank] = matrix.shape[1]
            else:
                features = data.get(f"x_{rank}")
                counts[rank] = 0 if features is None else len(features)
                matrix = sp.csr_matrix((counts[rank - 1], counts[rank]))
            incidences[rank] = matrix
        offsets = np.cumsum([0] + counts)
        total = int(offsets[-1])
        memberships = {0: sp.eye(counts[0], format="csr")}
        for rank in range(1, self.max_rank + 1):
            memberships[rank] = memberships[rank - 1] @ incidences[rank]
            memberships[rank].data[:] = 1
        for rank, count in enumerate(counts):
            features = data.get(f"x_{rank}")
            if features is None and rank == 0:
                features = data.get("x")
            if features is None and rank == 1:
                features = data.get("x_hyperedges")
            if features is None:
                features = torch.ones(count, 1)
            if features.ndim == 1:
                features = features[:, None]
            if len(features) != count:
                raise ValueError(f"Rank {rank} feature/count mismatch")
            data[f"trawl_signal_{rank}"] = features.float()
        if self.attr_scaling is not None and self.attr_dim > 0:
            raw = data["trawl_signal_0"][:, : self.attr_dim].double().numpy()
            values = np.sign(raw) * np.log1p(np.abs(raw))
            for rank in range(len(counts)):
                if rank:
                    values = incidences[rank].T @ values
                signal = data[f"trawl_signal_{rank}"]
                if counts[rank]:
                    signal = signal.clone()
                    signal[:, : self.attr_dim] = torch.from_numpy(
                        np.asarray(values)
                    ).to(signal.dtype)
                    data[f"trawl_signal_{rank}"] = signal
        colors = np.repeat(np.arange(len(counts)), counts)
        if self.color_key is not None:
            colors = np.asarray(data[self.color_key].cpu()).reshape(-1)
            if len(colors) != total or np.any(colors < 0):
                raise ValueError(
                    "Colors must contain one nonnegative ID per cell"
                )
        matrices = []
        if self.graph == "cell_overlap":
            selected = [
                memberships[r]
                if r in self.overlap_ranks
                else sp.csr_matrix((counts[0], counts[r]))
                for r in range(len(counts))
            ]
            membership = sp.hstack(selected, format="csr")
            matrix = (membership.T @ membership).tocsr()
            matrix.setdiag(0)
            matrix.eliminate_zeros()
            matrices.append(matrix)
        else:
            for joined in self.neighborhoods:
                # "a+b" walks the union of relations a and b as one family
                # (e.g. up_incidence-0+up_incidence-1: rank-switching Hasse
                # walks over vertices, edges and cells).
                parts = [
                    self._expanded(
                        name, counts, incidences, data, offsets, total
                    )
                    for name in joined.split("+")
                ]
                matrices.append(
                    parts[0] if len(parts) == 1 else sum(parts[1:], parts[0]).tocsr()
                )
        entries, weights = [], []
        for relation, matrix in enumerate(matrices):
            coo = matrix.tocoo()
            entries.extend(
                zip(coo.row, coo.col, np.full(coo.nnz, relation), strict=True)
            )
            weights.extend(coo.data)
        data.trawl_edges = torch.tensor(
            np.asarray(entries).reshape(-1, 3), dtype=torch.long
        )
        data.trawl_weights = torch.tensor(weights, dtype=torch.float32)
        if self.cell_vertices:
            table = np.full(
                (total, self.cell_vertex_width), -1, dtype=np.int64
            )
            for rank, count in enumerate(counts):
                if not count:
                    continue
                members = memberships[rank].tocsc()
                for cell in range(count):
                    verts = np.sort(
                        members.indices[
                            members.indptr[cell] : members.indptr[cell + 1]
                        ]
                    )
                    if len(verts) > self.cell_vertex_width:
                        raise ValueError(
                            f"A rank-{rank} cell has {len(verts)} vertices; "
                            f"raise cell_vertex_width "
                            f"({self.cell_vertex_width})"
                        )
                    table[offsets[rank] + cell, : len(verts)] = verts
            data.trawl_cell_vertices = torch.from_numpy(table)
        if self.pivots:
            if self.graph == "cell_overlap":
                raise ValueError("pivots need named adjacency relations")
            data.trawl_pivots = torch.from_numpy(
                self._edge_pivots(
                    data.trawl_edges.numpy(), counts, offsets, incidences
                )
            )
        if self.signs:
            data.trawl_signs = torch.from_numpy(
                self._edge_signs(
                    data.trawl_edges.numpy(),
                    counts,
                    offsets,
                    incidences,
                    raw_incidences,
                    memberships,
                )
            )
        data.trawl_counts = torch.tensor([counts], dtype=torch.long)
        data.trawl_colors = torch.tensor(colors, dtype=torch.long)
        data.trawl_relations = torch.tensor([len(matrices)])
        union = sum(matrices, sp.csr_matrix((total, total)))
        enc_matrices = (
            matrices if self.encoding_scope == "separate" else [union]
        )
        if self.color_refinement:
            if np.any(colors >= self.num_colors):
                raise ValueError("num_colors must cover every color ID")
            refined = []
            # Fixed slots for each unordered color pair, including same-color
            # relations. Refine PSE graphs without changing walk connectivity.
            for matrix in enc_matrices:
                coo = matrix.tocoo()
                for a in range(self.num_colors):
                    for b in range(a, self.num_colors):
                        keep = (
                            (colors[coo.row] == a) & (colors[coo.col] == b)
                        ) | ((colors[coo.row] == b) & (colors[coo.col] == a))
                        refined.append(
                            sp.csr_matrix(
                                (
                                    coo.data[keep],
                                    (coo.row[keep], coo.col[keep]),
                                ),
                                shape=matrix.shape,
                            )
                        )
            enc_matrices = refined
        # HOPSE feature encodings diffuse the state inputs; they need one
        # shared input width across ranks.
        widths = {
            data[f"trawl_signal_{rank}"].shape[1]
            for rank, count in enumerate(counts)
            if count
        }
        state_features = (
            np.concatenate(
                [
                    data[f"trawl_signal_{rank}"].cpu().numpy()
                    for rank, count in enumerate(counts)
                    if count
                ]
            )
            if len(widths) == 1
            else None
        )
        data.trawl_pe = torch.from_numpy(
            np.concatenate(
                [
                    hopse_encodings(
                        matrix, features=state_features, **self.encodings
                    )
                    for matrix in enc_matrices
                ],
                axis=1,
            )
        )
        if self.rich_features:
            data.trawl_pe = torch.cat(
                (
                    data.trawl_pe,
                    torch.from_numpy(
                        self._rich_features(data, counts, memberships)
                    ).to(data.trawl_pe.dtype),
                ),
                dim=1,
            )
        digest = hashlib.sha256()
        for tensor in [
            data.trawl_counts,
            data.trawl_edges,
            data.trawl_weights,
        ]:
            digest.update(tensor.numpy().tobytes())
        data.trawl_identity = torch.tensor(
            [int.from_bytes(digest.digest()[:7], "little")]
        )
        return data
