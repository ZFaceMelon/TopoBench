"""Frozen ARPACK stationary-distribution convention for historical profiles."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs


def arpack_start(size: int) -> np.ndarray:
    """Fixed ARPACK starting vector.

    Without ``v0`` ARPACK draws its start from internal state that advances
    with every call in the process, so identical graphs could receive
    different eigenvectors (sign and degenerate-subspace choices). A fixed
    start makes each solve a pure function of its matrix.

    Parameters
    ----------
    size : int
        Length of the starting vector.

    Returns
    -------
    np.ndarray
        Standard-normal vector drawn from a generator seeded with 0.
    """
    return np.random.default_rng(0).normal(size=size)


def _dominant_left_vector(P: sp.csr_matrix) -> np.ndarray:
    """Frozen ARPACK convention for one weakly connected component.

    Parameters
    ----------
    P : sp.csr_matrix
        Row-stochastic transition matrix of the component.

    Returns
    -------
    np.ndarray
        Nonnegative stationary vector with unit L1 norm (when nonzero).
    """
    n = P.shape[0]
    if n <= 2:
        values, vectors = np.linalg.eig(P.toarray().T)
        pi = np.abs(np.real(vectors[:, np.argmin(np.abs(values - 1))]))
        return pi / pi.sum() if pi.sum() > 0 else np.full(n, 1.0 / n)
    eigenvalues, eigenvectors = eigs(
        P.T,
        k=1,
        which="LM",  # largest magnitude — equals 1 for a valid stochastic matrix
        maxiter=100_000,
        v0=arpack_start(n),
    )
    if abs(eigenvalues[0] - 1) > 1e-6:
        # Bipartite components also have |lambda| = 1 at lambda = -1; that
        # eigenvector is not stationary. Select the eigenvalue-one mode.
        eigenvalues, eigenvectors = eigs(
            P.T, k=1, which="LR", maxiter=100_000, v0=arpack_start(n)
        )

    # eigs returns complex arrays; imaginary part should be negligible
    pi = np.real(eigenvectors[:, 0])
    pi = np.abs(pi)  # guard against sign / phase ambiguity

    l1_norm = pi.sum()
    if l1_norm > 0:
        pi = pi / l1_norm

    return pi


def compute_stationary_distribution(P: sp.csr_matrix) -> np.ndarray:
    """Find the dominant left eigenvector π such that  π^T P = π^T.

    Equivalently, π is the right eigenvector of P^T with eigenvalue 1.
    The result is scaled so that ||π||_1 = 1.

    A disconnected chain has one stationary vector per component, and a
    single ARPACK vector can vanish on whole components (making the Chung
    Laplacian the identity there). Components are therefore solved
    separately and weighted by their share of states.

    Parameters
    ----------
    P : sp.csr_matrix
        Row-stochastic transition matrix.

    Returns
    -------
    np.ndarray
        Stationary distribution π of shape ``(n,)``.
    """
    P = sp.csr_matrix(P)
    n = P.shape[0]
    count, labels = connected_components(P, directed=True, connection="weak")
    if count <= 1:
        return _dominant_left_vector(P)
    pi = np.zeros(n, dtype=np.float64)
    for component in range(count):
        index = np.flatnonzero(labels == component)
        local = (
            np.ones(1)
            if len(index) == 1
            else _dominant_left_vector(P[index][:, index])
        )
        pi[index] = local * (len(index) / n)
    return pi


def compute_chung_laplacian(P: sp.csr_matrix, pi: np.ndarray) -> sp.csr_matrix:
    """Chung normalized Laplacian (symmetric form).

    Computes

      L = I - (Φ^{1/2} P Φ^{-1/2} + Φ^{-1/2} P^T Φ^{1/2}) / 2

    where Φ = diag(π).

    Parameters
    ----------
    P : sp.csr_matrix
        Row-stochastic transition matrix.
    pi : np.ndarray
        Stationary distribution of ``P``; zero entries get a zero inverse.

    Returns
    -------
    sp.csr_matrix
        Symmetric Chung Laplacian of shape ``(n, n)``.
    """
    n = P.shape[0]
    sqrt_pi = np.sqrt(pi)
    inv_sqrt_pi = np.where(sqrt_pi > 0, 1.0 / sqrt_pi, 0.0)

    Phi_sqrt = sp.diags(sqrt_pi, format="csr")
    Phi_inv_sqrt = sp.diags(inv_sqrt_pi, format="csr")

    # Symmetrized, π-normalized transition operator
    S = (Phi_sqrt @ P @ Phi_inv_sqrt + Phi_inv_sqrt @ P.T @ Phi_sqrt) * 0.5

    identity = sp.identity(n, format="csr", dtype=np.float64)
    L = identity - S
    return L.tocsr()
