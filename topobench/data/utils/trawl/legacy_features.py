"""Historical bond/ring feature definitions, isolated from training scripts.

The frozen PROTEINS 74.70 snapshot supplies topology/local/spectral features;
the rich hybrid snapshot supplies the optional 40/46 structural channels.
See docs/trawl.md for provenance and numerical limitations.
"""

import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

from topobench.data.utils.trawl.sampling import (
    prepare_walk_rows,
    simulate_nbrw_sparse,
)

TOKEN_RING5 = 4
TOKEN_RING6 = 5
VOCAB_SIZE = 6
MAX_HYPEREDGE_ATOMS = 6
NUM_ATOM_TYPES = 64
LOCAL_PE_DIM = 8


def extract_atom_types(data, attr_dim=0) -> np.ndarray:
    """Integer atom-type IDs from PyG data.x (categorical or one-hot).

    The first ``attr_dim`` columns are continuous node attributes, as with
    the historical ``TRAWL_NODE_ATTR_DIM`` setting, and are skipped.

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


def _bond_type(raw) -> int:
    """Integer bond type from a scalar, a one-column or a one-hot edge row.

    Parameters
    ----------
    raw : array_like
        Edge attribute value or row.

    Returns
    -------
    int
        Active column of a one-hot row, otherwise the leading value.
    """
    values = np.ravel(np.asarray(raw))
    if values.size == 1:
        return int(values[0])
    # Multi-column rows (e.g. PyG one-hot bond labels) identify the type by
    # their active column; other widths fall back to the leading column.
    if np.isin(values, (0, 1)).all() and values.sum() == 1:
        return int(values.argmax())
    return int(values[0])


def unique_undirected_bonds(
    data,
) -> tuple[list[tuple[int, ...]], list[float], list[int]]:
    """Size-2 hyperedges from PyG edge_index + integer/float edge_attr.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with ``edge_index`` and optional ``edge_attr``.

    Returns
    -------
    endpoints : list[tuple[int, ...]]
        Sorted ``(u, v)`` pair of each unique undirected bond.
    weights : list[float]
        Bond weight ``1 + bond_type`` (1.0 without edge attributes).
    bond_types : list[int]
        Bond type clipped to ``[0, 3]`` (0 without edge attributes).
    """
    ei = data.edge_index.cpu().numpy()
    ea = None if data.edge_attr is None else data.edge_attr.cpu().numpy()
    seen: dict[tuple[int, int], int] = {}
    endpoints: list[tuple[int, ...]] = []
    weights: list[float] = []
    bond_types: list[int] = []

    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen[key] = len(endpoints)
        endpoints.append(key)
        if ea is None:
            btype, w = 0, 1.0
        else:
            btype = int(np.clip(_bond_type(ea[e]), 0, 3))
            w = 1.0 + float(btype)  # single < double < ...
        bond_types.append(btype)
        weights.append(w)
    return endpoints, weights, bond_types


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


def hyperedge_token_from_type(size: int, bond_type: int) -> int:
    """Vocabulary token of a hyperedge from its size and bond type.

    Parameters
    ----------
    size : int
        Number of nodes in the hyperedge.
    bond_type : int
        Bond type of the hyperedge.

    Returns
    -------
    int
        ``TOKEN_RING5`` or ``TOKEN_RING6`` for 5-/6-node hyperedges,
        otherwise the bond type clipped to ``[0, 3]``.
    """
    if size == 5:
        return TOKEN_RING5
    if size == 6:
        return TOKEN_RING6
    return int(np.clip(bond_type, 0, 3))


def build_sparse_dual(
    endpoints: list[tuple[int, ...]],
    weights: list[float],
) -> tuple[list[list[int]], list[list[float]]]:
    """Build the sparse hyperedge dual graph with row-normalized weights.

    Two hyperedges are adjacent when they share nodes; the edge weight is
    the product of their weights times the number of shared nodes.
    Isolated hyperedges get a self-loop.

    Parameters
    ----------
    endpoints : list[tuple[int, ...]]
        Node indices of each hyperedge.
    weights : list[float]
        Weight of each hyperedge.

    Returns
    -------
    neighbors : list[list[int]]
        Dual-graph neighbors of each hyperedge.
    neigh_weights : list[list[float]]
        Transition probabilities to each neighbor (rows sum to 1).
    """
    m = len(endpoints)
    node_to_edges: dict[int, list[int]] = defaultdict(list)
    for e, nodes in enumerate(endpoints):
        for v in nodes:
            node_to_edges[int(v)].append(e)

    pair_shared: dict[tuple[int, int], int] = defaultdict(int)
    for edges in node_to_edges.values():
        for a_i in range(len(edges)):
            for b_i in range(a_i + 1, len(edges)):
                a, b = edges[a_i], edges[b_i]
                if a > b:
                    a, b = b, a
                pair_shared[(a, b)] += 1

    neighbors: list[list[int]] = [[] for _ in range(m)]
    neigh_weights: list[list[float]] = [[] for _ in range(m)]
    for (a, b), shared in pair_shared.items():
        w = float(weights[a]) * float(weights[b]) * shared
        neighbors[a].append(b)
        neigh_weights[a].append(w)
        neighbors[b].append(a)
        neigh_weights[b].append(w)

    for i in range(m):
        if not neighbors[i]:
            neighbors[i] = [i]
            neigh_weights[i] = [1.0]
        total = sum(neigh_weights[i])
        neigh_weights[i] = [w / total for w in neigh_weights[i]]
    return neighbors, neigh_weights


def local_topological_pe(
    endpoints: list[tuple[int, ...]],
    weights: list[float],
    bond_types: list[int],
    neighbors: list[list[int]],
) -> np.ndarray:
    """Local topological positional encoding of each hyperedge.

    Channels: normalized dual degree, log weight, size, bond flag,
    token, 5-ring flag, 6-ring flag and a constant bias.

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
        pe[i, 4] = float(tok) / 5.0
        pe[i, 5] = 1.0 if tok == TOKEN_RING5 else 0.0
        pe[i, 6] = 1.0 if tok == TOKEN_RING6 else 0.0
        pe[i, 7] = 1.0
    return pe


def laplacian_guided_neigh_probs(
    neighbors: list[list[int]],
    neigh_probs: list[list[float]],
    *,
    gamma: float = 1.0,
    diffusion_t: float = 0.1,
) -> list[list[float]]:
    """Reweight dual-graph transitions with a Chung-Laplacian heat-kernel prior.

    Matches the HOPSE-style guidance used in trawl_zinc_sisa: transitions
    are multiplied by ``|exp(-t L)| ** gamma`` and renormalized. Rows whose
    guided mass vanishes fall back to the original probabilities.

    Parameters
    ----------
    neighbors : list[list[int]]
        Dual-graph neighbors of each hyperedge.
    neigh_probs : list[list[float]]
        Transition probabilities to each neighbor.
    gamma : float, optional
        Exponent applied to the heat-kernel magnitude (default: 1.0).
    diffusion_t : float, optional
        Heat-kernel diffusion time (default: 0.1).

    Returns
    -------
    list[list[float]]
        Guided transition probabilities aligned with ``neighbors``.
    """
    import scipy.linalg
    import scipy.sparse as sp

    from . import legacy_laplacian as mutag_math

    m = len(neighbors)
    if m == 0:
        return []
    P = np.zeros((m, m), dtype=np.float64)
    for i, (nbrs, probs) in enumerate(
        zip(neighbors, neigh_probs, strict=True)
    ):
        for j, p in zip(nbrs, probs, strict=True):
            P[i, int(j)] += float(p)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    P = P / row_sums

    P_sp = sp.csr_matrix(P)
    pi = mutag_math.compute_stationary_distribution(P_sp)
    L = np.asarray(mutag_math.compute_chung_laplacian(P_sp, pi).todense())
    evals, evecs = scipy.linalg.eigh(L)
    H_t = evecs @ np.diag(np.exp(-float(diffusion_t) * evals)) @ evecs.T
    P_guided = P * (np.abs(H_t) ** float(gamma))
    g_sums = P_guided.sum(axis=1, keepdims=True)
    g_sums[g_sums == 0.0] = 1.0
    P_guided = P_guided / g_sums

    out: list[list[float]] = []
    for i, nbrs in enumerate(neighbors):
        if not nbrs:
            out.append([])
            continue
        raw = np.asarray([P_guided[i, int(j)] for j in nbrs], dtype=np.float64)
        s = float(raw.sum())
        if s <= 0.0:
            raw = np.asarray(neigh_probs[i], dtype=np.float64)
            s = float(raw.sum()) or 1.0
        out.append((raw / s).tolist())
    return out


def hasse_spectral_pse(
    neighbors: list[list[int]],
    neigh_probs: list[list[float]],
    *,
    q_vec: np.ndarray | None = None,
    heat_times=(0.05, 0.1, 0.2, 0.5, 1.0),
    electrostatic_betas=(0.1, 0.5, 1.0),
    laplacian_dim=8,
) -> np.ndarray:
    """Compute extra node-wise PSEs on the hyperedge Hasse dual graph.

    The encodings are:
      1) HKdiagSE: diagonal entries of exp(-t L_H)
      2) Electrostatic potentials: p_beta = (L_H + beta I)^{-1} q
      3) LapPE: low-frequency eigenvectors (Laplacian Eigenmaps)

    Uses Chung Laplacian on the Markov transition defined by neigh_probs.
    We use a truncated eigendecomposition for speed; features are approximate.

    Parameters
    ----------
    neighbors : list[list[int]]
        Dual-graph neighbors of each hyperedge.
    neigh_probs : list[list[float]]
        Transition probabilities to each neighbor.
    q_vec : np.ndarray or None, optional
        Charge vector for the electrostatic potentials; a constant charge is
        used when None or mismatched in length (default: None).
    heat_times : sequence of float, optional
        Diffusion times for HKdiagSE.
    electrostatic_betas : sequence of float, optional
        Regularization shifts for the electrostatic potentials.
    laplacian_dim : int, optional
        Number of Laplacian eigenvector channels (default: 8).

    Returns
    -------
    np.ndarray
        Float32 array of shape
        ``(m, len(heat_times) + len(electrostatic_betas) + laplacian_dim)``.
    """
    import scipy.linalg
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    from . import legacy_laplacian as mutag_math

    m = len(neighbors)
    if m == 0:
        return np.zeros(
            (0, len(heat_times) + len(electrostatic_betas) + laplacian_dim),
            dtype=np.float32,
        )

    if q_vec is None:
        # Default charge: favor higher-degree / weight hyperedges using local_topological_pe proxy.
        q_vec = np.ones(m, dtype=np.float64)
    q_vec = np.asarray(q_vec, dtype=np.float64).reshape(-1)
    if q_vec.shape[0] != m:
        q_vec = np.ones(m, dtype=np.float64)
    # Center & normalize for numerical stability.
    q_vec = q_vec - float(q_vec.mean())
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm <= 1e-12:
        # Fall back to a constant charge, then recompute normalization.
        q_vec = np.ones(m, dtype=np.float64)
        q_norm = float(np.linalg.norm(q_vec))

    # Build transition matrix P as sparse rows.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i, nbrs in enumerate(neighbors):
        for j, p in zip(nbrs, neigh_probs[i], strict=True):
            rows.append(i)
            cols.append(int(j))
            vals.append(float(p))

    if len(rows) == 0:
        return np.zeros(
            (m, len(heat_times) + len(electrostatic_betas) + laplacian_dim),
            dtype=np.float32,
        )

    P = sp.csr_matrix(
        (np.array(vals, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(m, m),
    )
    row_sums = np.asarray(P.sum(axis=1)).reshape(-1)
    row_sums[row_sums == 0.0] = 1.0
    # Row-normalize to ensure a proper transition matrix.
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ P

    pi = mutag_math.compute_stationary_distribution(P)
    L = mutag_math.compute_chung_laplacian(P, pi)

    # Truncated eigen-decomposition: smallest-magnitude eigenpairs.
    # We need enough eigenvectors for HKdiagSE (uses exp(-t*lambda)) and for LapPE.
    n_eigs = int(max(laplacian_dim + 2, len(heat_times) + 2))
    n_eigs = min(max(n_eigs, 6), m - 1) if m > 2 else max(min(m, 2), 1)

    if m <= n_eigs + 1:
        L_dense = np.asarray(L.todense(), dtype=np.float64)
        evals, evecs = scipy.linalg.eigh(L_dense)
    else:
        # eigsh returns unsorted eigenvalues; sort them.
        try:
            evals, evecs = spla.eigsh(
                L, k=n_eigs, which="SM", v0=mutag_math.arpack_start(m)
            )
        except spla.ArpackNoConvergence:
            # Near-disconnected gated graphs can stall the smallest-magnitude
            # iteration. A negative shift avoids the Laplacian's zero mode
            # while targeting the same low-frequency eigenspace, sparsely.
            warnings.warn(
                "Historical spectral encoding did not converge; retrying "
                "with sparse shift-invert iteration.",
                RuntimeWarning,
                stacklevel=2,
            )
            evals, evecs = spla.eigsh(
                L,
                k=n_eigs,
                sigma=-1e-6,
                which="LM",
                v0=mutag_math.arpack_start(m),
            )
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

    # Ensure float64 for exp/inv stability.
    evals = np.asarray(evals, dtype=np.float64).reshape(-1)
    evecs = np.asarray(evecs, dtype=np.float64)

    # 1) HKdiagSE
    hk_feats: list[np.ndarray] = []
    for t in heat_times:
        coeff = np.exp(-float(t) * evals)  # (k,)
        diag_t = (evecs**2) @ coeff  # sum_k u_k(i)^2 * exp(-t*lambda_k)
        hk_feats.append(diag_t.astype(np.float32).reshape(-1, 1))

    hk = (
        np.concatenate(hk_feats, axis=1)
        if hk_feats
        else np.zeros((m, 0), dtype=np.float32)
    )

    # 2) Electrostatic potentials: p_beta = sum_k (u_k^T q) * u_k / (lambda_k + beta)
    # Compute alpha_k = u_k^T q once.
    alpha = (evecs.T @ (q_vec / q_norm)).astype(np.float64)  # (k,)
    electro_feats: list[np.ndarray] = []
    for beta in electrostatic_betas:
        denom = evals + float(beta)
        coeff = alpha / denom
        p_beta = evecs @ coeff  # (m,)
        electro_feats.append(p_beta.astype(np.float32).reshape(-1, 1))
    electro = (
        np.concatenate(electro_feats, axis=1)
        if electro_feats
        else np.zeros((m, 0), dtype=np.float32)
    )

    # 3) Laplacian Eigenmaps: low-frequency eigenvectors after the trivial component.
    # Skip the first eigenpair if it's near-constant (typically lambda~0).
    # Then take the next laplacian_dim vectors.
    # Identify non-trivial indices where lambda is above a threshold.
    nontrivial = np.where(evals > 1e-6)[0]
    if nontrivial.size == 0:
        lape = np.zeros((m, laplacian_dim), dtype=np.float32)
    else:
        start = int(nontrivial[0])
        vecs = evecs[:, start : start + laplacian_dim]
        if vecs.shape[1] < laplacian_dim:
            pad = np.zeros(
                (m, laplacian_dim - vecs.shape[1]), dtype=np.float64
            )
            vecs = np.concatenate([vecs, pad], axis=1)
        lape = vecs.astype(np.float32)

    pse = np.concatenate([hk, electro, lape], axis=1).astype(np.float32)
    # Defensive clamp: numerical spectral routines can occasionally produce
    # NaN/Inf on near-degenerate tiny components; zero them instead of
    # poisoning the entire cache / training run.
    pse = np.nan_to_num(pse, nan=0.0, posinf=0.0, neginf=0.0)
    return pse


def empirical_rwse(
    neighbors: list[list[int]],
    neigh_probs: list[list[float]],
    k_rwse: int = 8,
    n_samples: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Monte Carlo random-walk return probabilities of each hyperedge.

    Parameters
    ----------
    neighbors : list[list[int]]
        Dual-graph neighbors of each hyperedge.
    neigh_probs : list[list[float]]
        Transition probabilities to each neighbor.
    k_rwse : int, optional
        Number of walk steps (default: 8).
    n_samples : int, optional
        Number of sampled walks per start hyperedge (default: 8).
    seed : int, optional
        Random seed (default: 0).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(m, k_rwse)``; entry ``t - 1`` is the
        fraction of walks at their start after ``t`` steps.
    """
    m = len(neighbors)
    rwse = np.zeros((m, k_rwse), dtype=np.float32)
    rng = np.random.default_rng(seed)
    rows = prepare_walk_rows(neighbors, neigh_probs)
    for i in range(m):
        hits = np.zeros(k_rwse, dtype=np.float64)
        for _ in range(n_samples):
            path = simulate_nbrw_sparse(
                neighbors, neigh_probs, i, k_rwse + 1, rng, rows=rows
            )
            for t in range(1, k_rwse + 1):
                if path[t] == i:
                    hits[t - 1] += 1.0
        rwse[i] = (hits / n_samples).astype(np.float32)
    return rwse


def rich_molecular_structural_pe(
    data,
    endpoints: list[tuple[int, ...]],
    atom_types: np.ndarray,
    attr_dim: int = 0,
) -> np.ndarray:
    """Target-free local molecular context for NCI1-style node labels.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Molecular graph.
    endpoints : list[tuple[int, ...]]
        Node indices of each hyperedge.
    atom_types : np.ndarray
        Atom-type ID of each node.
    attr_dim : int, optional
        Number of leading continuous attribute columns in ``data.x``; when
        positive, six attribute summary channels are added (default: 0).

    Returns
    -------
    np.ndarray
        Float32 array with one row of degree, neighbor-label, cycle, local,
        atom and global context features per hyperedge.
    """
    graph = to_networkx(data, to_undirected=True)
    n = max(int(data.num_nodes), 1)
    degrees = np.asarray([graph.degree(v) for v in range(n)], dtype=np.float32)
    max_degree = max(float(degrees.max(initial=1.0)), 1.0)

    neighbor_hist = np.zeros((n, 16), dtype=np.float32)
    for v in range(n):
        nbrs = list(graph.neighbors(v))
        for u in nbrs:
            neighbor_hist[v, int(atom_types[int(u)]) % 16] += 1.0
        if nbrs:
            neighbor_hist[v] /= float(len(nbrs))

    cycle_flags = np.zeros((n, 4), dtype=np.float32)
    for cycle in nx.cycle_basis(graph):
        bucket = (
            0
            if len(cycle) == 3
            else 1
            if len(cycle) == 4
            else 2
            if len(cycle) == 5
            else 3
        )
        for v in cycle:
            cycle_flags[int(v), bucket] = 1.0

    clustering_map = nx.clustering(graph)
    clustering = np.asarray(
        [clustering_map.get(v, 0.0) for v in range(n)], dtype=np.float32
    )
    try:
        core_map = nx.core_number(graph)
    except nx.NetworkXError:
        core_map = {v: 0 for v in range(n)}
    cores = np.asarray(
        [core_map.get(v, 0) for v in range(n)], dtype=np.float32
    )
    max_core = max(float(cores.max(initial=1.0)), 1.0)
    global_context = np.asarray(
        [
            np.log1p(n) / np.log(128.0),
            float(degrees.mean()) / 8.0,
            float(nx.density(graph)) if n > 1 else 0.0,
        ],
        dtype=np.float32,
    )

    attr_dim = max(int(attr_dim), 0)
    transformed_attrs = np.zeros((n, attr_dim), dtype=np.float32)
    neighbor_attrs = np.zeros((n, attr_dim), dtype=np.float32)
    if attr_dim and data.x is not None:
        raw_x = data.x.detach().cpu().numpy()
        if raw_x.ndim == 1:
            raw_x = raw_x[:, None]
        available = min(attr_dim, raw_x.shape[1])
        raw_attrs = raw_x[:, :available].astype(np.float32)
        transformed_attrs[:, :available] = (
            np.sign(raw_attrs) * np.log1p(np.abs(raw_attrs)) / np.log1p(1024.0)
        )
        for v in range(n):
            nbrs = list(graph.neighbors(v))
            if nbrs:
                neighbor_attrs[v] = transformed_attrs[
                    np.asarray(nbrs, dtype=np.int64)
                ].mean(axis=0)

    rows: list[np.ndarray] = []
    for nodes_tuple in endpoints:
        nodes = np.asarray(nodes_tuple, dtype=np.int64)
        node_degrees = degrees[nodes]
        degree_stats = np.asarray(
            [
                node_degrees.mean() / max_degree,
                node_degrees.max(initial=0.0) / max_degree,
                node_degrees.min(initial=0.0) / max_degree,
                node_degrees.std() / max_degree,
            ],
            dtype=np.float32,
        )
        degree_bins = np.zeros(5, dtype=np.float32)
        for degree in node_degrees:
            degree_bins[min(int(degree), 4)] += 1.0
        degree_bins /= float(max(len(nodes), 1))

        labels = atom_types[nodes].astype(np.float32)
        atom_context = np.asarray(
            [
                labels.mean() / float(NUM_ATOM_TYPES - 1),
                labels.std() / float(NUM_ATOM_TYPES - 1),
                len(np.unique(labels)) / float(max(len(labels), 1)),
                float(np.all(labels == labels[0])) if len(labels) else 0.0,
            ],
            dtype=np.float32,
        )
        local_context = np.asarray(
            [
                clustering[nodes].mean(),
                clustering[nodes].max(initial=0.0),
                cores[nodes].mean() / max_core,
                cores[nodes].max(initial=0.0) / max_core,
            ],
            dtype=np.float32,
        )
        continuous_context = np.empty(0, dtype=np.float32)
        if attr_dim:
            own = transformed_attrs[nodes].reshape(-1)
            nbr = neighbor_attrs[nodes].reshape(-1)
            continuous_context = np.asarray(
                [
                    own.mean(),
                    own.std(),
                    own.min(),
                    own.max(),
                    nbr.mean(),
                    nbr.std(),
                ],
                dtype=np.float32,
            )
        rows.append(
            np.concatenate(
                [
                    degree_stats,
                    degree_bins,
                    neighbor_hist[nodes].mean(axis=0),
                    cycle_flags[nodes].mean(axis=0),
                    local_context,
                    atom_context,
                    global_context,
                    continuous_context,
                ]
            )
        )
    return np.asarray(rows, dtype=np.float32)
