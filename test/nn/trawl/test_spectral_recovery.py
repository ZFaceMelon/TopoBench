"""Recover low-frequency features when the ordinary ARPACK iteration stalls."""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from topobench.data.utils.trawl.legacy_features import hasse_spectral_pse


def test_sparse_spectral_retry_preserves_eigenproblem(monkeypatch):
    original = spla.eigsh
    calls = []

    def fail_first(matrix, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise spla.ArpackNoConvergence(
                "forced nonconvergence",
                np.empty(0),
                np.empty((matrix.shape[0], 0)),
            )
        values, vectors = original(matrix, **kwargs)
        np.testing.assert_allclose(
            matrix @ vectors, vectors * values, atol=1e-8
        )
        assert np.max(values) < 0.5
        return values, vectors

    monkeypatch.setattr(spla, "eigsh", fail_first)
    size = 64
    neighbors = [[(i - 1) % size, (i + 1) % size] for i in range(size)]
    probabilities = [np.array([0.5, 0.5]) for _ in neighbors]
    with pytest.warns(RuntimeWarning, match="shift-invert"):
        features = hasse_spectral_pse(
            neighbors,
            probabilities,
            heat_times=[0.1],
            electrostatic_betas=[0.5],
            laplacian_dim=4,
        )
    assert features.shape == (size, 6)
    assert np.isfinite(features).all()
    assert np.linalg.norm(features[:, 2:]) > 0
    assert calls[0]["which"] == "SM"
    assert calls[1]["sigma"] < 0
