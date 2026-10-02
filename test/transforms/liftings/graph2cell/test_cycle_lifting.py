"""Test the cell cycles lifting."""

import networkx as nx
import torch

from topobench.transforms.liftings.graph2cell import CellCycleLifting


class TestCellCycleLifting:
    """Test the CellCycleLifting class."""

    def setup_method(self):
        """Initialise the CellCycleLifting class."""
        self.lifting = CellCycleLifting()

    def test_lift_topology(self, simple_graph_1):
        """Test the lift_topology method.

        Parameters
        ----------
        simple_graph_1 : Data
            A simple graph used for testing.
        """
        data = simple_graph_1
        lifted_data = self.lifting.forward(data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with incidence_1."

        incidence = lifted_data.incidence_2.to_dense()
        # A cycle basis is not unique. NetworkX versions can choose different
        # valid bases, so test the cycle-space contract, not one traversal order.
        assert incidence.shape == (13, 6)  # m - n + 1 independent cycles
        assert torch.all((incidence == 0) | (incidence == 1))
        degrees = expected_incidence_1 @ incidence
        assert torch.all((degrees == 0) | (degrees == 2))
        edges = [
            tuple(torch.where(column != 0)[0].tolist())
            for column in expected_incidence_1.T
        ]
        for cycle in incidence.T:
            graph = nx.Graph(
                [
                    edge
                    for edge, selected in zip(edges, cycle, strict=True)
                    if selected
                ]
            )
            assert len(graph) >= 3
            assert nx.is_connected(graph)

        # Verify independence over GF(2), the field of the undirected cycle space.
        reduced = incidence.bool().clone()
        rank = 0
        for column in range(reduced.shape[1]):
            candidates = torch.where(reduced[rank:, column])[0]
            assert len(candidates), "Cycle columns are linearly dependent"
            pivot = rank + int(candidates[0])
            reduced[[rank, pivot]] = reduced[[pivot, rank]]
            for row in range(rank + 1, reduced.shape[0]):
                if reduced[row, column]:
                    reduced[row] ^= reduced[rank]
            rank += 1
        assert rank == 6
