"""Lifting stats and anonymous-id cap usage of the graph_u8_lin walks."""
import sys
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from topobench.nn.backbones.combinatorial.trawl import WalkSampler, walk_seed

path = sys.argv[1]
out = fs.torch_load(path)
ds = InMemoryDataset.__new__(InMemoryDataset); InMemoryDataset.__init__(ds, None)
ds.data = out[-1].from_dict(out[0]) if isinstance(out[0], dict) else out[0]; ds.slices = out[1]
sampler = WalkSampler(0)
cells, nodes, fallback, sizes, distinct = [], [], 0, [], []
for g in range(len(ds)):
    d = ds[g]
    counts = d.trawl_counts[0].numpy()
    nodes.append(counts[0]); cells.append(counts[2])
    table = d.trawl_cell_vertices.numpy()
    top = table[counts[0] + counts[1]:]
    sizes.extend((top >= 0).sum(1).tolist())
    edges = d.trawl_edges.numpy(); w = d.trawl_weights.numpy()
    sel = edges[:, 2] == 0
    if not sel.any():
        fallback += 1; sel = edges[:, 2] == 1
    if not sel.any():
        continue
    total = int(counts.sum())
    m = sp.csr_matrix((w[sel], (edges[sel, 0], edges[sel, 1])), shape=(total, total))
    paths = sampler(m, seed=walk_seed(0, g, 0, 0, 0), k=32, length=32)
    verts = table[paths].reshape(len(paths), -1)
    for row in verts:
        distinct.append(len(np.unique(row[row >= 0])))
cells, nodes, sizes, distinct = map(np.array, (cells, nodes, sizes, distinct))
print(path)
print("graphs %d | nodes mean %.1f max %d | >32 nodes %.1f%%" % (len(cells), nodes.mean(), nodes.max(), 100 * (nodes > 32).mean()))
print("2-cells/graph mean %.2f median %d max %d | zero 2-cells %.1f%% | fallback to edge walks (no edge-adjacent 2-cells) %.1f%%" % (cells.mean(), np.median(cells), cells.max(), 100 * (cells == 0).mean(), 100 * fallback / len(cells)))
print("cell sizes max %d hist %s" % (sizes.max(), dict(zip(*map(list, np.unique(sizes, return_counts=True))))))
for cap in (32, 64, 128):
    print("walks with >= %d distinct vertices (ids clipped at cap %d): %.2f%%" % (cap, cap, 100 * (distinct >= cap).mean()))
print("distinct vertices per walk: mean %.1f p50 %d p90 %d p99 %d max %d" % (distinct.mean(), *np.percentile(distinct, [50, 90, 99]).astype(int), distinct.max()))
