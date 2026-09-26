"""Correctness-gated benchmark for mojo-celerite2.

Every case checks numerical agreement with the pure-NumPy formulation before
timing, so a regression in the Mojo kernels shows up as a correctness failure
rather than a suspiciously good number.


Baselines are the fastest reasonable NumPy formulation, not a Python loop:
NumPy broadcasts the lags and evaluates the same exp/cos/sin reduction with
`np.exp`, `np.cos` and `np.sin` over whole arrays, so it is already vectorised
and is the fair comparison.

The Mojo column is reported twice, serial and 8-threaded.  The serial number is
the one that says whether the kernel itself is good; the threaded number is the
one that says what a caller actually gets, because the shim fans out over a
thread pool once the array is large enough.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_celerite2 as m  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _numpy_value(coeffs, tau):
    """The array-at-a-time reduction, exactly as `celerite2.terms` does it."""
    ar, cr, ac, bc, cc, dc = coeffs
    t = np.abs(tau)[..., None]
    k = np.zeros(t.shape[:-1])
    if len(ar):
        k = k + np.sum(ar * np.exp(-cr * t), axis=-1)
    if len(ac):
        k = k + np.sum(
            np.exp(-cc * t) * (ac * np.cos(dc * t) + bc * np.sin(dc * t)),
            axis=-1,
        )
    return k


def _numpy_psd(coeffs, omega):
    ar, cr, ac, bc, cc, dc = coeffs
    w2 = omega[..., None] ** 2
    psd = np.zeros(w2.shape[:-1])
    if len(ar):
        psd = psd + np.sum(ar * cr / (cr**2 + w2), axis=-1)
    if len(ac):
        w02 = cc**2 + dc**2
        num = (ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2
        den = w2**2 + 2.0 * (cc * cc - dc * dc) * w2 + w02 * w02
        psd = psd + np.sum(num / den, axis=-1)
    return np.sqrt(2.0 / np.pi) * psd


def bench_get_value(n: int = 1 << 22, workers: int = 8):
    term = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
    coeffs = term.get_coefficients()
    tau = np.abs(np.random.default_rng(0).standard_normal(n)) * 4.0

    def numpy_value():
        return _numpy_value(coeffs, tau)

    want = numpy_value()
    assert np.allclose(
        term.get_value(tau, workers=1), want, rtol=1e-10
    ), "get_value mismatch"
    np.testing.assert_array_equal(
        term.get_value(tau, workers=workers), term.get_value(tau, workers=1)
    )

    return (
        f"get_value n={n}",
        _time(numpy_value),
        _time(lambda: term.get_value(tau, workers=1)),
        _time(lambda: term.get_value(tau, workers=workers)),
    )


def bench_get_psd(n: int = 1 << 22, workers: int = 8):
    term = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
    coeffs = term.get_coefficients()
    w = np.abs(np.random.default_rng(1).standard_normal(n)) * 4.0

    def numpy_psd():
        return _numpy_psd(coeffs, w)

    want = numpy_psd()
    assert np.allclose(
        term.get_psd(w, workers=1), want, rtol=1e-10
    ), "get_psd mismatch"
    assert np.allclose(term.get_psd(w, workers=workers), want, rtol=1e-10)

    return (
        f"get_psd n={n}",
        _time(numpy_psd),
        _time(lambda: term.get_psd(w, workers=1)),
        _time(lambda: term.get_psd(w, workers=workers)),
    )


def bench_to_dense(n: int = 1024, workers: int = 8):
    term = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
    coeffs = term.get_coefficients()
    x = np.sort(np.random.default_rng(2).uniform(0.0, 10.0, n))
    diag = np.full(n, 0.01)

    def numpy_dense():
        k = _numpy_value(coeffs, x[:, None] - x[None, :])
        k[np.diag_indices_from(k)] += diag
        return k

    want = numpy_dense()
    assert np.allclose(
        term.to_dense(x, diag, workers=1), want, rtol=1e-10
    ), "to_dense mismatch"
    assert np.allclose(term.to_dense(x, diag, workers=workers), want, rtol=1e-10)

    return (
        f"to_dense n={n}",
        _time(numpy_dense),
        _time(lambda: term.to_dense(x, diag, workers=1)),
        _time(lambda: term.to_dense(x, diag, workers=workers)),
    )


def bench_cholesky(n: int = 1500):
    """Dense Cholesky against LAPACK through numpy.  LAPACK is blocked and
    tuned; the unblocked Mojo kernel is expected to lose here, and that is the
    result.  There is no thread fan-out: the factorisation is sequential in its
    inner loop, so the serial column is the only honest one."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((n, n))
    k = (a @ a.T + n * np.eye(n)) / n
    l, logdet = m.cholesky(k.copy())
    ref = np.linalg.cholesky(k)
    np.testing.assert_allclose(np.tril(l), ref, rtol=1e-11, atol=1e-13)
    # Same definition on both sides: 2 * sum(log L_ii).  Comparing against
    # numpy's LU-based slogdet would be comparing two different estimators.
    np.testing.assert_allclose(
        logdet, 2.0 * np.sum(np.log(np.diag(ref))), rtol=1e-8, atol=1e-9
    )
    return (
        f"cholesky n={n}",
        _time(lambda: np.linalg.cholesky(k)),
        _time(lambda: m.cholesky(k.copy())),
        float("nan"),
    )


def main():
    print(
        f"{'case':<20}{'numpy':>12}{'mojo serial':>14}{'ratio':>8}"
        f"{'mojo x8':>12}{'ratio':>8}"
    )
    print("-" * 74)
    for fn in (
        bench_get_value,
        bench_get_psd,
        bench_to_dense,
        bench_cholesky,
    ):
        label, ref, serial, par = fn()
        r1 = ref / serial if serial else float("nan")
        r8 = ref / par if par == par and par else float("nan")
        print(
            f"{label:<20}{ref * 1e3:>9.1f}ms{serial * 1e3:>11.1f}ms{r1:>7.2f}x"
            f"{par * 1e3:>9.1f}ms{r8:>7.2f}x"
        )


if __name__ == "__main__":
    main()
