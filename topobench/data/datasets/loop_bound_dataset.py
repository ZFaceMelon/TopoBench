"""LoopBound: does a marked edge cycle of a triangulated surface bound?

A synthetic diagnostic built by ``scripts/loop_bound/make_loop_bound.py`` from
MANTRA 2-manifolds. Each graph is a surface with one marked simple edge cycle
(edge feature ``x_1 = 1`` on the cycle); the label is 1 if the cycle is a
boundary over Z/2. Every complex appears with one bounding and one
non-bounding cycle of equal length, so the label depends only on how the
marked edges relate to the triangles.
"""

import json
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


def sparse(rows, cols, values, shape):
    """Build a coalesced sparse COO float tensor.

    Parameters
    ----------
    rows, cols : array_like
        Indices.
    values : array_like
        Values.
    shape : tuple
        Matrix shape.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor.
    """
    return torch.sparse_coo_tensor(
        torch.tensor(np.stack((rows, cols)), dtype=torch.long),
        torch.tensor(values, dtype=torch.float),
        shape,
    ).coalesce()


def record_to_data(record):
    """Convert one LoopBound record to a simplicial ``Data`` object.

    Signed incidences follow the sorted-vertex convention (the coefficient of
    the face omitting the i-th vertex is (-1)^i), as toponetx does.

    Parameters
    ----------
    record : dict
        Record with ``triangulation``, ``cycle`` and ``bounds``.

    Returns
    -------
    torch_geometric.data.Data
        Graph with ``x_0``, ``x_1`` (cycle mark), ``x_2``, signed
        ``incidence_1``/``incidence_2`` and label ``y``.
    """
    tris = sorted(tuple(sorted(t)) for t in record["triangulation"])
    n = 1 + max(v for t in tris for v in t)
    edges = sorted({e for a, b, c in tris for e in ((a, b), (a, c), (b, c))})
    eid = {e: i for i, e in enumerate(edges)}
    r1, c1, v1 = [], [], []
    for j, (a, b) in enumerate(edges):
        r1 += [a, b]
        c1 += [j, j]
        v1 += [-1.0, 1.0]
    r2, c2, v2 = [], [], []
    for j, (a, b, c) in enumerate(tris):
        for face, sign in (((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0)):
            r2.append(eid[face])
            c2.append(j)
            v2.append(sign)
    mark = torch.zeros(len(edges), 1)
    for a, b in record["cycle"]:
        mark[eid[tuple(sorted((a, b)))], 0] = 1.0
    return Data(
        x=torch.ones(n, 1),
        x_0=torch.ones(n, 1),
        x_1=mark,
        x_2=torch.ones(len(tris), 1),
        incidence_1=sparse(r1, c1, v1, (n, len(edges))),
        incidence_2=sparse(r2, c2, v2, (len(edges), len(tris))),
        num_nodes=n,
        y=torch.tensor([int(record["bounds"])], dtype=torch.long).squeeze(),
    )


class LoopBoundDataset(InMemoryDataset):
    """LoopBound graphs read from ``raw/loop_bound.json``.

    Parameters
    ----------
    root : str
        Dataset root (``<data_dir>/<data_name>``).
    """

    def __init__(self, root):
        super().__init__(root)
        out = fs.torch_load(self.processed_paths[0])
        data, self.slices, self.sizes, data_cls = out
        self.data = data_cls.from_dict(data)

    @property
    def raw_file_names(self):
        """Return the raw file name.

        Returns
        -------
        list of str
            ``["loop_bound.json"]``.
        """
        return ["loop_bound.json"]

    @property
    def processed_file_names(self):
        """Return the processed file name.

        Returns
        -------
        str
            ``"data.pt"``.
        """
        return "data.pt"

    def download(self):
        """Fail: generate the raw file with make_loop_bound.py.

        Raises
        ------
        FileNotFoundError
            Always.
        """
        raise FileNotFoundError(
            f"Put loop_bound.json (scripts/loop_bound/make_loop_bound.py) in {self.raw_dir}"
        )

    def process(self):
        """Convert every record and save the collated dataset."""
        records = json.load(open(osp.join(self.raw_dir, "loop_bound.json")))
        data_list = [record_to_data(r) for r in records]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
