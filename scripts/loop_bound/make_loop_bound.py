"""Generate the LoopBound diagnostic: does a marked edge cycle bound?

For every MANTRA 2-manifold of a chosen type (torus, Klein bottle, RP^2), sample
simple edge cycles and label each by whether it is a boundary over Z/2 (the
cycle vector lies in the column space of the edge-triangle incidence mod 2).
Each kept complex contributes one bounding and one non-bounding cycle of the
SAME length, so counts, orientability, surface type and cycle length carry no
label information: the label depends only on how the marked edges relate to
the triangles.

The label is computed here with linear algebra for dataset construction only;
models never see it as an input.

Usage: python make_loop_bound.py 2_manifolds.json out.json [seed]
"""

import json
import sys
from collections import deque

import numpy as np

TYPES = {"T^2", "Klein bottle", "RP^2"}


def rank_mod2(matrix):
    """Rank over Z/2 by Gaussian elimination.

    Parameters
    ----------
    matrix : np.ndarray
        0/1 matrix.

    Returns
    -------
    int
        Rank over Z/2.
    """
    m = matrix.copy() % 2
    rank = 0
    rows, cols = m.shape
    for c in range(cols):
        pivot = next((r for r in range(rank, rows) if m[r, c]), None)
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(rows):
            if r != rank and m[r, c]:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def random_cycle(adj, edge_id, rng, mode):
    """Sample one simple cycle as a sorted tuple of edge ids.

    Parameters
    ----------
    adj : list of list of int
        Vertex adjacency lists.
    edge_id : dict
        Map from sorted vertex pair to edge id.
    rng : np.random.Generator
        Random generator.
    mode : str
        ``"bfs"`` (shortest path closing a random edge) or ``"walk"``
        (loop-erased random walk closing a random edge).

    Returns
    -------
    tuple of int or None
        Edge ids of the cycle, or None.
    """
    edges = list(edge_id)
    u, v = edges[rng.integers(len(edges))]
    if mode == "bfs":
        prev = {v: None}
        queue = deque([v])
        while queue:
            a = queue.popleft()
            if a == u:
                break
            nbrs = list(adj[a])
            rng.shuffle(nbrs)
            for b in nbrs:
                if b in prev or (a == v and b == u):
                    continue
                prev[b] = a
                queue.append(b)
        if u not in prev:
            return None
        path = [u]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
    else:
        # Loop-erased random walk from v to u, not using edge (v, u) first.
        path = [v]
        steps = 0
        while path[-1] != u and steps < 200:
            a = path[-1]
            nbrs = [b for b in adj[a] if not (a == v and b == u and len(path) == 1)]
            b = nbrs[rng.integers(len(nbrs))]
            if b in path:
                path = path[: path.index(b) + 1]
            else:
                path.append(b)
            steps += 1
        if path[-1] != u:
            return None
    if len(path) < 3:
        return None
    cyc = [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
    cyc.append(tuple(sorted((u, v))))
    if len(set(cyc)) != len(cyc):
        return None
    return tuple(sorted(edge_id[e] for e in cyc))


def main():
    """Write the LoopBound json."""
    src, out = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = np.random.default_rng(seed)
    manifolds = [m for m in json.load(open(src)) if m["name"] in TYPES]
    records, skipped = [], 0
    for cid, m in enumerate(manifolds):
        tris = [tuple(sorted(t)) for t in m["triangulation"]]
        verts = sorted({v for t in tris for v in t})
        index = {v: i for i, v in enumerate(verts)}
        tris = [tuple(index[v] for v in t) for t in tris]
        edges = sorted({e for a, b, c in tris for e in ((a, b), (a, c), (b, c))})
        edge_id = {e: i for i, e in enumerate(edges)}
        adj = [[] for _ in verts]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        b2 = np.zeros((len(edges), len(tris)), dtype=np.uint8)
        for j, (a, b, c) in enumerate(tris):
            for e in ((a, b), (a, c), (b, c)):
                b2[edge_id[e], j] = 1
        base = rank_mod2(b2)
        found = {}
        for attempt in range(400):
            cyc = random_cycle(adj, edge_id, rng, "bfs" if attempt % 2 else "walk")
            if cyc is None or cyc in found:
                continue
            vec = np.zeros((len(edges), 1), dtype=np.uint8)
            vec[list(cyc), 0] = 1
            found[cyc] = int(rank_mod2(np.hstack((b2, vec))) == base)
        by_len = {}
        for cyc, lab in found.items():
            by_len.setdefault(len(cyc), {0: [], 1: []})[lab].append(cyc)
        lengths = [n for n, d in by_len.items() if d[0] and d[1]]
        if not lengths:
            skipped += 1
            continue
        n = lengths[rng.integers(len(lengths))]
        for lab in (0, 1):
            options = by_len[n][lab]
            cyc = options[rng.integers(len(options))]
            records.append(
                {
                    "complex_id": cid,
                    "name": m["name"],
                    "n_vertices": len(verts),
                    "triangulation": [list(t) for t in tris],
                    "cycle": [list(edges[i]) for i in cyc],
                    "length": n,
                    "bounds": lab,
                }
            )
    json.dump(records, open(out, "w"))
    lens = np.array([r["length"] for r in records])
    print(f"complexes {len(manifolds)}, kept {len(manifolds) - skipped}, "
          f"records {len(records)}, cycle length min/median/max "
          f"{lens.min()}/{int(np.median(lens))}/{lens.max()}")
    for name in sorted(TYPES):
        sel = [r for r in records if r["name"] == name]
        print(name, len(sel), "bounding frac",
              np.mean([r["bounds"] for r in sel]) if sel else None)


if __name__ == "__main__":
    main()
