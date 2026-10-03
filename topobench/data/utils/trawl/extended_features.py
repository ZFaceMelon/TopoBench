"""Extended token vocabulary and complementary cycle/star liftings.

These definitions are from the later rich hybrid/gated snapshot. Their token
normalization and constituent width differ from the frozen classic profile.
"""

from collections import defaultdict

import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

from topobench.data.utils.trawl.legacy_features import (
    build_sparse_dual,
    unique_undirected_bonds,
)
from topobench.data.utils.trawl.legacy_features import (
    empirical_rwse as empirical_rwse,
)
from topobench.data.utils.trawl.legacy_features import (
    hasse_spectral_pse as hasse_spectral_pse,
)
from topobench.data.utils.trawl.legacy_features import (
    laplacian_guided_neigh_probs as laplacian_guided_neigh_probs,
)
from topobench.data.utils.trawl.legacy_features import (
    rich_molecular_structural_pe as rich_molecular_structural_pe,
)

TOKEN_RING3, TOKEN_RING4, TOKEN_RING5, TOKEN_RING6 = 4, 5, 6, 7
TOKEN_STAR, TOKEN_NODE, TOKEN_CYCLE_LONG = 8, 9, 10
VOCAB_SIZE, MAX_HYPEREDGE_ATOMS, NUM_ATOM_TYPES, LOCAL_PE_DIM = 11, 18, 64, 8


def extract_atom_types(data, attr_dim=0) -> np.ndarray:
    """Integer atom-type IDs from PyG data.x (categorical or one-hot).

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph whose ``x`` holds categorical or one-hot atom labels.
    attr_dim : int, optional
        Number of leading continuous attribute columns to skip (default: 0).

    Returns
    -------
    np.ndarray
        Atom-type IDs of shape ``(num_nodes,)``, clipped to
        ``[0, NUM_ATOM_TYPES - 1]``; zeros when no labels are available.
    """
    n = int(data.num_nodes)
    if data.x is None:
        return np.zeros(n, dtype=np.int64)
    x = data.x.detach().cpu().numpy()
    attr_dim = max(int(attr_dim), 0)
    if x.ndim > 1 and attr_dim:
        x = x[:, attr_dim:]
        if x.shape[1] == 0:
            return np.zeros(n, dtype=np.int64)
    if x.ndim == 1:
        ids = x.astype(np.int64)
    elif x.shape[1] == 1:
        ids = x[:, 0].astype(np.int64)
    else:
        ids = np.argmax(x, axis=1).astype(np.int64)
    return np.clip(ids, 0, NUM_ATOM_TYPES - 1)


def pack_hyperedge_atoms(
    endpoints: list[tuple[int, ...]],
    atom_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-hyperedge atom IDs + mask, padded to MAX_HYPEREDGE_ATOMS.

    Parameters
    ----------
    endpoints : list[tuple[int, ...]]
        Node indices of each hyperedge.
    atom_types : np.ndarray
        Atom-type ID of each node.

    Returns
    -------
    atoms : np.ndarray
        Atom-type IDs of shape ``(m, MAX_HYPEREDGE_ATOMS)``, zero-padded.
    mask : np.ndarray
        Float mask of the same shape, 1.0 at filled positions.
    """
    m = len(endpoints)
    atoms = np.zeros((m, MAX_HYPEREDGE_ATOMS), dtype=np.int64)
    mask = np.zeros((m, MAX_HYPEREDGE_ATOMS), dtype=np.float32)
    for i, nodes in enumerate(endpoints):
        for j, v in enumerate(nodes[:MAX_HYPEREDGE_ATOMS]):
            atoms[i, j] = int(atom_types[int(v)])
            mask[i, j] = 1.0
    return atoms, mask


def lift_rings(
    data,
    endpoints: list[tuple[int, ...]],
    weights: list[float],
    bond_types: list[int],
) -> tuple[list[tuple[int, ...]], list[float], list[int]]:
    """Append 5-/6-cycles as hyperedges (cellular lifting).

    Cycles are taken from the networkx cycle basis; the input lists are
    extended in place.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Molecular graph.
    endpoints : list[tuple[int, ...]]
        Node indices of the existing hyperedges.
    weights : list[float]
        Weights of the existing hyperedges.
    bond_types : list[int]
        Type tokens of the existing hyperedges.

    Returns
    -------
    endpoints : list[tuple[int, ...]]
        Input endpoints with ring hyperedges appended.
    weights : list[float]
        Input weights with weight 3.0 appended for each ring.
    bond_types : list[int]
        Input types with ``TOKEN_RING5``/``TOKEN_RING6`` appended.
    """
    G = to_networkx(data, to_undirected=True)
    for cycle in nx.cycle_basis(G):
        if len(cycle) in (5, 6):
            endpoints.append(tuple(int(v) for v in cycle))
            weights.append(3.0)
            bond_types.append(TOKEN_RING5 if len(cycle) == 5 else TOKEN_RING6)
    return endpoints, weights, bond_types


def lift_short_cycles(
    data,
    endpoints: list[tuple[int, ...]],
    weights: list[float],
    bond_types: list[int],
) -> tuple[list[tuple[int, ...]], list[float], list[int]]:
    """Append unique triangle/4-cycle cells plus the champion's 5/6-cycle basis.

    Cells whose node set already exists are skipped; the input lists are
    extended in place.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Molecular graph.
    endpoints : list[tuple[int, ...]]
        Node indices of the existing hyperedges.
    weights : list[float]
        Weights of the existing hyperedges.
    bond_types : list[int]
        Type tokens of the existing hyperedges.

    Returns
    -------
    endpoints : list[tuple[int, ...]]
        Input endpoints with the new cycle cells appended.
    weights : list[float]
        Input weights with 2.5 (triangle), 2.75 (4-cycle) or 3.0 (5/6-cycle)
        appended per new cell.
    bond_types : list[int]
        Input types with the matching ring tokens appended.
    """
    G = to_networkx(data, to_undirected=True)
    seen = {frozenset(nodes) for nodes in endpoints}

    # Every 3-clique is a triangle cell.
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) > 3:
            break
        if len(clique) == 3:
            key = frozenset(int(v) for v in clique)
            if key not in seen:
                endpoints.append(tuple(sorted(key)))
                weights.append(2.5)
                bond_types.append(TOKEN_RING3)
                seen.add(key)

    # Two nodes with two common neighbors define a 4-cycle. We store the cell
    # as a node set because the downstream dual graph uses incidence only.
    nodes = sorted(int(v) for v in G.nodes())
    for ai, u in enumerate(nodes):
        nu = set(G.neighbors(u))
        for v in nodes[ai + 1 :]:
            common = sorted(int(x) for x in nu.intersection(G.neighbors(v)))
            for i in range(len(common)):
                for j in range(i + 1, len(common)):
                    key = frozenset((u, v, common[i], common[j]))
                    if len(key) == 4 and key not in seen:
                        endpoints.append(tuple(sorted(key)))
                        weights.append(2.75)
                        bond_types.append(TOKEN_RING4)
                        seen.add(key)

    for cycle in nx.cycle_basis(G):
        if len(cycle) in (5, 6):
            key = frozenset(int(v) for v in cycle)
            if key not in seen:
                endpoints.append(tuple(int(v) for v in cycle))
                weights.append(3.0)
                bond_types.append(
                    TOKEN_RING5 if len(cycle) == 5 else TOKEN_RING6
                )
                seen.add(key)
    return endpoints, weights, bond_types


def lift_stars(
    data,
    endpoints: list[tuple[int, ...]],
    weights: list[float],
    bond_types: list[int],
) -> tuple[list[tuple[int, ...]], list[float], list[int]]:
    """Append each nontrivial closed 1-hop neighborhood as an ego/star cell.

    Neighborhoods with fewer than three nodes or an existing node set are
    skipped; the input lists are extended in place.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Molecular graph.
    endpoints : list[tuple[int, ...]]
        Node indices of the existing hyperedges.
    weights : list[float]
        Weights of the existing hyperedges.
    bond_types : list[int]
        Type tokens of the existing hyperedges.

    Returns
    -------
    endpoints : list[tuple[int, ...]]
        Input endpoints with star cells appended.
    weights : list[float]
        Input weights with weight 2.0 appended for each star.
    bond_types : list[int]
        Input types with ``TOKEN_STAR`` appended for each star.
    """
    G = to_networkx(data, to_undirected=True)
    seen = {frozenset(nodes) for nodes in endpoints}
    for center in G.nodes():
        key = frozenset([int(center), *(int(v) for v in G.neighbors(center))])
        if len(key) < 3 or key in seen:
            continue
        endpoints.append(tuple(sorted(key)))
        weights.append(2.0)
        bond_types.append(TOKEN_STAR)
        seen.add(key)
    return endpoints, weights, bond_types


def build_cycle18_adj3(
    data,
) -> tuple[
    list[tuple[int, ...]],
    list[float],
    list[int],
    list[list[int]],
    list[list[float]],
]:
    """TopoBench-like cycle lift with the six same-rank Adj-3 relations.

    Cells are rank-0 nodes, rank-1 graph edges, and rank-2 cycle-basis cycles
    of length at most 18. Adjacency is the union of A01/A02, A10/A12, and
    A20/A21. This is a single-graph TRAWL analogue of HOPSE's separate Adj-3
    neighborhood ensemble.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Molecular graph.

    Returns
    -------
    endpoints : list[tuple[int, ...]]
        Node indices of the node, edge and cycle cells, in that order.
    weights : list[float]
        Cell weights (1.0 for nodes, bond weights for edges, 3.0 for cycles).
    bond_types : list[int]
        Cell tokens (``TOKEN_NODE``, bond types, ring/long-cycle tokens).
    neighbors : list[list[int]]
        Sorted Adj-3 neighbors of each cell (self-loop when isolated).
    probs : list[list[float]]
        Uniform transition probabilities over each cell's neighbors.
    """
    G = to_networkx(data, to_undirected=True)
    edge_cells, edge_weights, edge_types = unique_undirected_bonds(data)
    cycle_cells: list[tuple[int, ...]] = []
    cycle_types: list[int] = []
    seen_cycles: set[frozenset[int]] = set()
    for cycle in nx.cycle_basis(G):
        if not 3 <= len(cycle) <= 18:
            continue
        key = frozenset(int(v) for v in cycle)
        if key in seen_cycles:
            continue
        seen_cycles.add(key)
        cycle_cells.append(tuple(int(v) for v in cycle))
        cycle_types.append(
            TOKEN_RING3
            if len(cycle) == 3
            else TOKEN_RING4
            if len(cycle) == 4
            else TOKEN_RING5
            if len(cycle) == 5
            else TOKEN_RING6
            if len(cycle) == 6
            else TOKEN_CYCLE_LONG
        )

    node_cells = [(int(v),) for v in sorted(G.nodes())]
    endpoints = node_cells + edge_cells + cycle_cells
    weights = [1.0] * len(node_cells) + edge_weights + [3.0] * len(cycle_cells)
    bond_types = [TOKEN_NODE] * len(node_cells) + edge_types + cycle_types
    n0, n1 = len(node_cells), len(edge_cells)
    neighbors_sets: list[set[int]] = [set() for _ in endpoints]

    def connect_group(indices: list[int]) -> None:
        """Connect every pair of the given cells in ``neighbors_sets``.

        Parameters
        ----------
        indices : list[int]
            Cell indices to connect pairwise.
        """
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                neighbors_sets[a].add(b)
                neighbors_sets[b].add(a)

    # A0,1 and A1,0: node adjacency via edges; edge adjacency via nodes.
    node_to_edges: dict[int, list[int]] = defaultdict(list)
    edge_lookup: dict[frozenset[int], int] = {}
    for e, cell in enumerate(edge_cells):
        idx = n0 + e
        edge_lookup[frozenset(cell)] = idx
        for v in cell:
            node_to_edges[int(v)].append(idx)
        connect_group([int(cell[0]), int(cell[1])])
    for incident in node_to_edges.values():
        connect_group(incident)

    # A0,2/A2,0 and A1,2/A2,1.
    node_to_cycles: dict[int, list[int]] = defaultdict(list)
    edge_to_cycles: dict[int, list[int]] = defaultdict(list)
    for c, cycle in enumerate(cycle_cells):
        cidx = n0 + n1 + c
        cycle_nodes = [int(v) for v in cycle]
        connect_group(cycle_nodes)
        for v in cycle_nodes:
            node_to_cycles[v].append(cidx)
        cycle_edge_indices: list[int] = []
        for i, u in enumerate(cycle_nodes):
            v = cycle_nodes[(i + 1) % len(cycle_nodes)]
            eidx = edge_lookup.get(frozenset((u, v)))
            if eidx is not None:
                cycle_edge_indices.append(eidx)
                edge_to_cycles[eidx].append(cidx)
        connect_group(cycle_edge_indices)
    for incident in node_to_cycles.values():
        connect_group(incident)
    for incident in edge_to_cycles.values():
        connect_group(incident)

    neighbors: list[list[int]] = []
    probs: list[list[float]] = []
    for i, nbr_set in enumerate(neighbors_sets):
        nbrs = sorted(nbr_set) if nbr_set else [i]
        neighbors.append(nbrs)
        probs.append([1.0 / len(nbrs)] * len(nbrs))
    return endpoints, weights, bond_types, neighbors, probs


def build_dual_champion_cycle18(
    data,
    *,
    cycle_mix: float = 0.35,
) -> tuple[
    list[tuple[int, ...]],
    list[float],
    list[int],
    list[list[int]],
    list[list[float]],
]:
    """Fuse champion and cycle18/Adj-3 liftings without discarding either.

    The champion edge+5/6-ring dual graph supplies the primary transition
    channel.  Cycle18/Adj-3 contributes node cells, longer cycle cells and its
    same-rank relations. Duplicate cells are shared between channels. The
    conservative cycle_mix keeps the new topology complementary instead of
    letting its denser adjacency dominate the random walk.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Molecular graph.
    cycle_mix : float, optional
        Weight of the cycle18/Adj-3 transitions relative to the champion
        transitions before row normalization (default: 0.35).

    Returns
    -------
    endpoints : list[tuple[int, ...]]
        Node indices of the deduplicated cells.
    weights : list[float]
        Weight of each cell (from its first occurrence).
    bond_types : list[int]
        Token of each cell (from its first occurrence).
    neighbors : list[list[int]]
        Sorted neighbors of each cell in the fused transition.
    probs : list[list[float]]
        Row-normalized fused transition probabilities.
    """
    champ_ep, champ_w, champ_t = unique_undirected_bonds(data)
    champ_ep, champ_w, champ_t = lift_rings(data, champ_ep, champ_w, champ_t)
    cyc_ep, cyc_w, cyc_t, cyc_nbr, cyc_prob = build_cycle18_adj3(data)

    endpoints: list[tuple[int, ...]] = []
    weights: list[float] = []
    bond_types: list[int] = []
    lookup: dict[tuple[int, ...], int] = {}

    def add_cell(cell: tuple[int, ...], weight: float, token: int) -> int:
        """Register a cell once by its sorted node set and return its index.

        Parameters
        ----------
        cell : tuple[int, ...]
            Node indices of the cell.
        weight : float
            Cell weight, stored only for a new cell.
        token : int
            Cell token, stored only for a new cell.

        Returns
        -------
        int
            Index of the (new or existing) cell in ``endpoints``.
        """
        key = tuple(sorted(int(v) for v in cell))
        idx = lookup.get(key)
        if idx is None:
            idx = len(endpoints)
            lookup[key] = idx
            endpoints.append(tuple(int(v) for v in cell))
            weights.append(float(weight))
            bond_types.append(int(token))
        return idx

    champ_map = [
        add_cell(e, w, t)
        for e, w, t in zip(champ_ep, champ_w, champ_t, strict=True)
    ]
    cyc_map = [
        add_cell(e, w, t) for e, w, t in zip(cyc_ep, cyc_w, cyc_t, strict=True)
    ]
    champ_nbr, champ_prob = build_sparse_dual(champ_ep, champ_w)
    transition: list[dict[int, float]] = [
        defaultdict(float) for _ in endpoints
    ]

    for old_i, nbrs in enumerate(champ_nbr):
        i = champ_map[old_i]
        for old_j, prob in zip(nbrs, champ_prob[old_i], strict=True):
            transition[i][champ_map[old_j]] += float(prob)
    for old_i, nbrs in enumerate(cyc_nbr):
        i = cyc_map[old_i]
        for old_j, prob in zip(nbrs, cyc_prob[old_i], strict=True):
            transition[i][cyc_map[old_j]] += float(cycle_mix) * float(prob)

    neighbors: list[list[int]] = []
    probs: list[list[float]] = []
    for i, row in enumerate(transition):
        if not row:
            neighbors.append([i])
            probs.append([1.0])
            continue
        nbrs = sorted(row)
        vals = [row[j] for j in nbrs]
        total = sum(vals)
        neighbors.append(nbrs)
        probs.append([v / total for v in vals])
    return endpoints, weights, bond_types, neighbors, probs


def build_dual_learnable_gate(
    data,
) -> tuple[
    list[tuple[int, ...]],
    list[float],
    list[int],
    list[list[int]],
    list[list[float]],
    int,
]:
    """Return independent champion and cycle18 components for gated fusion.

    Keeping the components disconnected makes the first half of sampled walks
    champion-only and the second half cycle18-only. The neural model can then
    learn a per-graph mixture without changing the total walk budget.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Molecular graph.

    Returns
    -------
    endpoints : list[tuple[int, ...]]
        Champion cells followed by cycle18 cells.
    weights : list[float]
        Weight of each cell.
    bond_types : list[int]
        Token of each cell.
    neighbors : list[list[int]]
        Neighbors of each cell; cycle18 indices are offset by ``split``.
    probs : list[list[float]]
        Transition probabilities aligned with ``neighbors``.
    split : int
        Number of champion cells (index of the first cycle18 cell).
    """
    champ_ep, champ_w, champ_t = unique_undirected_bonds(data)
    champ_ep, champ_w, champ_t = lift_rings(data, champ_ep, champ_w, champ_t)
    champ_nbr, champ_prob = build_sparse_dual(champ_ep, champ_w)
    cyc_ep, cyc_w, cyc_t, cyc_nbr, cyc_prob = build_cycle18_adj3(data)
    split = len(champ_ep)
    neighbors = [list(row) for row in champ_nbr]
    neighbors.extend([[split + int(j) for j in row] for row in cyc_nbr])
    probs = [list(map(float, row)) for row in champ_prob]
    probs.extend([list(map(float, row)) for row in cyc_prob])
    return (
        champ_ep + cyc_ep,
        champ_w + cyc_w,
        champ_t + cyc_t,
        neighbors,
        probs,
        split,
    )


def hyperedge_token_from_type(size: int, bond_type: int) -> int:
    """Vocabulary token of a hyperedge from its stored type.

    Parameters
    ----------
    size : int
        Number of nodes in the hyperedge (unused; kept for API parity).
    bond_type : int
        Stored type token of the hyperedge.

    Returns
    -------
    int
        Type token clipped to ``[0, VOCAB_SIZE - 1]``.
    """
    return int(np.clip(bond_type, 0, VOCAB_SIZE - 1))


def local_topological_pe(
    endpoints: list[tuple[int, ...]],
    weights: list[float],
    bond_types: list[int],
    neighbors: list[list[int]],
) -> np.ndarray:
    """Local topological positional encoding of each hyperedge.

    Channels: normalized dual degree, log weight, size, bond flag,
    normalized token, cycle flag, star flag and a constant bias.

    Parameters
    ----------
    endpoints : list[tuple[int, ...]]
        Node indices of each hyperedge.
    weights : list[float]
        Weight of each hyperedge.
    bond_types : list[int]
        Type token of each hyperedge.
    neighbors : list[list[int]]
        Dual-graph neighbors of each hyperedge.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(m, LOCAL_PE_DIM)``.
    """
    m = len(endpoints)
    max_deg = max((len(neighbors[i]) for i in range(m)), default=1)
    pe = np.zeros((m, LOCAL_PE_DIM), dtype=np.float32)
    for i, (nodes, w, btype) in enumerate(
        zip(endpoints, weights, bond_types, strict=True)
    ):
        size = len(nodes)
        tok = hyperedge_token_from_type(size, btype)
        pe[i, 0] = len(neighbors[i]) / max_deg
        pe[i, 1] = np.log1p(float(w))
        pe[i, 2] = min(size, 6) / 6.0
        pe[i, 3] = 1.0 if tok <= 3 else 0.0
        pe[i, 4] = float(tok) / float(VOCAB_SIZE - 1)
        pe[i, 5] = (
            1.0
            if tok
            in (
                TOKEN_RING3,
                TOKEN_RING4,
                TOKEN_RING5,
                TOKEN_RING6,
                TOKEN_CYCLE_LONG,
            )
            else 0.0
        )
        pe[i, 6] = 1.0 if tok == TOKEN_STAR else 0.0
        pe[i, 7] = 1.0
    return pe
