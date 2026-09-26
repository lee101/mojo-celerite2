"""Dense reference inference built on the Mojo kernels.

`celerite2` solves the GP system in O(N) with its semiseparable structure.  That
structure lives in the compiled `celerite2.driver` extension, which this port
leaves alone.  What the port *can* add is the exact dense form: assemble the
covariance with the ported `to_dense` kernel, factor it with the ported
Cholesky, and solve.  It is O(N^3) and therefore not for large N, but it is an
independent implementation of the same posterior, which is exactly what makes
it a good parity check: `log_likelihood` here and
`celerite2.numpy.GaussianProcess.log_likelihood` there must agree.
"""

from __future__ import annotations

import numpy as np

from . import _lib

__all__ = ["covariance", "cholesky", "solve", "log_likelihood", "predict_mean"]


def covariance(term, x, diag=None, workers=None) -> np.ndarray:
    """Dense N x N covariance matrix of `term` at coordinates `x`."""
    x = np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64)
    if diag is None:
        diag = np.zeros_like(x)
    else:
        diag = np.ascontiguousarray(np.atleast_1d(diag), dtype=np.float64)
    if diag.shape != x.shape:
        raise ValueError("dimension mismatch")
    return term.to_dense(x, diag, workers=workers)


def cholesky(k: np.ndarray):
    """Cholesky factor and log|det| of a symmetric positive definite matrix."""
    return _lib.cholesky(k)


def solve(l: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``L L^T x = y`` given the lower Cholesky factor ``L``."""
    return _lib.cho_solve(l, y)


def log_likelihood(term, t, y, yerr=None, diag=None) -> float:
    """Exact dense log marginal likelihood of a zero-mean GP."""
    t = np.ascontiguousarray(np.atleast_1d(t), dtype=np.float64)
    y = np.ascontiguousarray(np.atleast_1d(y), dtype=np.float64)
    if t.shape != y.shape:
        raise ValueError("dimension mismatch")
    if yerr is not None and diag is not None:
        raise ValueError("only one of 'diag' and 'yerr' can be provided")
    if diag is None:
        diag = np.zeros_like(t)
        if yerr is not None:
            diag = diag + np.ascontiguousarray(
                np.atleast_1d(yerr), dtype=np.float64
            ) ** 2
    else:
        diag = np.ascontiguousarray(np.atleast_1d(diag), dtype=np.float64)
    k = covariance(term, t, diag)
    _, logdet = cholesky(k)
    alpha = solve(k, y)
    # celerite2 folds the -0.5 * N * log(2 pi) constant into its `norm`.
    return float(
        -0.5 * (y @ alpha) - 0.5 * logdet - 0.5 * t.size * np.log(2.0 * np.pi)
    )


def predict_mean(term, t, y, t_star, yerr=None, diag=None) -> np.ndarray:
    """Posterior mean of a zero-mean GP at `t_star` given data at `t`."""
    t = np.ascontiguousarray(np.atleast_1d(t), dtype=np.float64)
    y = np.ascontiguousarray(np.atleast_1d(y), dtype=np.float64)
    t_star = np.ascontiguousarray(np.atleast_1d(t_star), dtype=np.float64)
    d = np.zeros_like(t) if diag is None else np.ascontiguousarray(
        np.atleast_1d(diag), dtype=np.float64
    )
    if yerr is not None:
        d = d + np.ascontiguousarray(
            np.atleast_1d(yerr), dtype=np.float64
        ) ** 2
    k = covariance(term, t, d)
    _, _ = cholesky(k)
    alpha = solve(k, y)
    kstar = np.ascontiguousarray(term.get_value(t_star[:, None] - t[None, :]))
    return kstar @ alpha
