"""Parity tests for the dense path: `to_dense`, Cholesky, solve, likelihood.

The strongest test in the file is `test_log_likelihood_matches_celerite_solver`:
the port builds the covariance with its own O(N^2) Mojo loop and factors it
with its own Cholesky, while `celerite2` builds the identical matrix with the
O(N) semiseparable solver in its compiled driver.  Two unrelated algorithms
landing on the same log marginal likelihood is hard to fake.
"""

import numpy as np
import pytest

import celerite2.terms as ct
from celerite2.numpy import GaussianProcess

import mojo_celerite2 as m

RTOL = 1e-10


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20260926)
    t = np.sort(rng.uniform(0.0, 10.0, 48))
    y = rng.standard_normal(48)
    yerr = 0.05 + 0.1 * rng.random(48)
    return t, y, yerr


def _terms(name):
    p = ct.SHOTerm.get_test_parameters()
    return m.SHOTerm(**p), ct.SHOTerm(**p)


def test_to_dense_matches_upstream(data):
    t, _, _ = data
    mine, theirs = _terms("SHOTerm")
    zero = np.zeros_like(t)
    got = mine.to_dense(t, zero)
    want = theirs.to_dense(t, zero)
    assert got.shape == (t.size, t.size)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)


def test_to_dense_is_symmetric_exactly(data):
    """K[i, j] and K[j, i] are the same |x[i] - x[j]| through the same code, so
    the bit patterns must agree.  A transposed index would not."""
    t, _, _ = data
    mine, _ = _terms("SHOTerm")
    k = mine.to_dense(t, np.zeros_like(t))
    np.testing.assert_array_equal(k, k.T)


def test_to_dense_diagonal_adds_diag(data):
    t, _, _ = data
    mine, _ = _terms("SHOTerm")
    d = np.linspace(0.01, 0.5, t.size)
    k0 = mine.to_dense(t, np.zeros_like(t))
    k1 = mine.to_dense(t, d)
    np.testing.assert_allclose(
        np.diag(k1), np.diag(k0) + d, rtol=RTOL, atol=0.0
    )


def test_to_dense_threaded_matches_serial():
    mine, _ = _terms("SHOTerm")
    rng = np.random.default_rng(5)
    n = 220
    t = np.sort(rng.uniform(0.0, 10.0, n))
    d = rng.random(n)
    serial = mine.to_dense(t, d, workers=1)
    for workers in (2, 3, 8):
        np.testing.assert_array_equal(
            mine.to_dense(t, d, workers=workers), serial
        )


def test_cholesky_matches_numpy(data):
    t, _, yerr = data
    mine, _ = _terms("SHOTerm")
    k = mine.to_dense(t, yerr**2)
    l, logdet = m.cholesky(k.copy())
    ln = np.linalg.cholesky(k)
    np.testing.assert_allclose(np.tril(l), ln, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(
        logdet, np.linalg.slogdet(k)[1], rtol=1e-11, atol=1e-10
    )


def test_cholesky_reconstructs_the_matrix(data):
    t, _, yerr = data
    mine, _ = _terms("SHOTerm")
    k = mine.to_dense(t, yerr**2)
    l, _ = m.cholesky(k.copy())
    np.testing.assert_allclose(l @ l.T, k, rtol=1e-10, atol=1e-12)


def test_cholesky_rejects_indefinite():
    a = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        m.cholesky(a)


def test_cho_solve_matches_numpy(data):
    t, _, yerr = data
    mine, _ = _terms("SHOTerm")
    rng = np.random.default_rng(11)
    b = rng.standard_normal(t.size)
    k = mine.to_dense(t, yerr**2)
    l, _ = m.cholesky(k.copy())
    np.testing.assert_allclose(
        m.cho_solve(l, b), np.linalg.solve(k, b), rtol=1e-9, atol=1e-11
    )


def test_log_likelihood_matches_celerite_solver(data):
    t, y, yerr = data
    mine, theirs = _terms("SHOTerm")
    gp = GaussianProcess(theirs)
    gp.compute(t, yerr=yerr)
    want = float(gp.log_likelihood(y))
    got = m.dense.log_likelihood(mine, t, y, yerr=yerr)
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_predict_mean_matches_celerite_solver(data):
    t, y, yerr = data
    mine, theirs = _terms("SHOTerm")
    gp = GaussianProcess(theirs)
    gp.compute(t, yerr=yerr)
    want = gp.predict(y)
    got = m.dense.predict_mean(mine, t, y, t, yerr=yerr)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


def test_log_likelihood_with_explicit_diag(data):
    t, y, _ = data
    mine, theirs = _terms("SHOTerm")
    d = np.full_like(t, 0.04)
    gp = GaussianProcess(theirs)
    gp.compute(t, diag=d)
    np.testing.assert_allclose(
        m.dense.log_likelihood(mine, t, y, diag=d),
        float(gp.log_likelihood(y)),
        rtol=1e-9,
    )


def test_log_likelihood_rejects_both_diag_and_yerr(data):
    t, y, _ = data
    mine, _ = _terms("SHOTerm")
    with pytest.raises(ValueError):
        m.dense.log_likelihood(
            mine, t, y, yerr=np.ones_like(t), diag=np.ones_like(t)
        )
