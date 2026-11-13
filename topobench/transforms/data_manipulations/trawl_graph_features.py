"""Old-TRAWL molecular context features for graph datasets (opt-in).

Ported verbatim from the old TRAWL benchmark source (v12,
``topobench/data/utils/trawl/legacy_features.py``) so new TRAWL can match the
inputs of the old NCI1 recipes ("rich features"). These are precomputed
structural descriptors (degree statistics, neighbor atom-type histograms,
ring-size membership, clustering, k-core, density): use them only where the
old recipes did, behind ``TRAWLTransform(rich_features=True)``.
"""

import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

NUM_ATOM_TYPES = 64
RICH_FEATURE_DIM = 40


def extract_atom_types(data, attr_dim=0) -> np.ndarray:
    """Integer atom-type IDs from PyG data.x (categorical or one-hot).

    The first ``attr_dim`` columns are continuous node attributes, as with
    the historical ``TRAWL_NODE_ATTR_DIM`` setting, and are skipped.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with node features ``x``.
    attr_dim : int, optional
        Leading continuous attribute columns to skip (default: 0).

    Returns
    -------
    np.ndarray
        Atom-type ID per node.
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
        Graph with ``edge_index`` and node features ``x``.
    endpoints : list of tuple of int
        Vertex set of every cell to describe.
    atom_types : np.ndarray
        Atom-type ID per node.
    attr_dim : int, optional
        Leading continuous attribute columns of ``x`` (default: 0).

    Returns
    -------
    np.ndarray
        ``[cells, 40 + 6 * (attr_dim > 0)]`` features.
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
