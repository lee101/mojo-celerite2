"""Parity tests: the Mojo lag loops against `celerite2.terms`.

Every test here compares against the real `celerite2` package, which is
installed in the test venv.  Mojo emits FMA, so the floating point comparisons
carry a tolerance; the *exact* assertions are only on index-like structure
(shapes, symmetry, chunk boundaries) where the bit pattern must match.
"""

import numpy as np
import pytest

import celerite2.terms as ct

import mojo_celerite2 as m

# The kernels accumulate a handful of transcendentals; FMA and libm differences
# put the Mojo and NumPy results a few ULP apart.  1e-11 is ~5 orders tighter
# than the discrepancy a real coefficient/index bug would produce.
RTOL = 1e-11

PAIRS = [
    ("RealTerm", m.RealTerm, ct.RealTerm),
    ("ComplexTerm", m.ComplexTerm, ct.ComplexTerm),
    ("SHOTerm", m.SHOTerm, ct.SHOTerm),
    ("Matern32Term", m.Matern32Term, ct.Matern32Term),
    ("RotationTerm", m.RotationTerm, ct.RotationTerm),
]


def _pair(which):
    for name, mine_cls, theirs_cls in PAIRS:
        if name == which:
            params = theirs_cls.get_test_parameters()
            return mine_cls(**params), theirs_cls(**params)
    raise KeyError(which)


@pytest.fixture(scope="module")
def lags():
    rng = np.random.default_rng(1234)
    return np.sort(np.abs(rng.standard_normal(257)) * 4.0)


@pytest.mark.parametrize("name,_,__", PAIRS, ids=[p[0] for p in PAIRS])
def test_get_value_matches_upstream(name, _, __, lags):
    mine, theirs = _pair(name)
    got = mine.get_value(lags)
    want = theirs.get_value(lags)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("name,_,__", PAIRS, ids=[p[0] for p in PAIRS])
def test_get_value_is_even(name, _, __, lags):
    """A dropped |tau| in the kernel shows up here as an asymmetry."""
    mine, _ = _pair(name)
    pos = mine.get_value(lags)
    neg = mine.get_value(-lags)
    # Both sides feed the identical |tau| into the same loop, so this is exact.
    np.testing.assert_array_equal(neg, pos)


@pytest.mark.parametrize("name,_,__", PAIRS, ids=[p[0] for p in PAIRS])
def test_get_psd_matches_upstream(name, _, __, lags):
    mine, theirs = _pair(name)
    np.testing.assert_allclose(
        mine.get_psd(lags), theirs.get_psd(lags), rtol=RTOL, atol=0.0
    )


@pytest.mark.parametrize("name,_,__", PAIRS, ids=[p[0] for p in PAIRS])
def test_get_psd_ignores_frequency_sign(name, _, __, lags):
    mine, theirs = _pair(name)
    np.testing.assert_allclose(
        mine.get_psd(-lags), theirs.get_psd(lags), rtol=RTOL, atol=0.0
    )


@pytest.mark.parametrize("name,_,__", PAIRS, ids=[p[0] for p in PAIRS])
def test_coefficients_match_upstream(name, _, __):
    """The coefficient algebra is NumPy on both sides; it must agree exactly
    in structure (lengths, real vs complex split) and to rounding in value."""
    mine, theirs = _pair(name)
    for got, want in zip(mine.get_coefficients(), theirs.get_coefficients()):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0)


def test_value_at_zero_matches_closed_form():
    """k(0) for each term, checked against the coefficient expression at tau=0.

    At tau = 0 the kernel is sum(ar) + sum(ac), which is a different reduction
    from the general lag loop, so this catches a dropped oscillatory branch.
    """
    for name, _, _ in PAIRS:
        mine, _ = _pair(name)
        ar, cr, ac, bc, cc, dc = mine.get_coefficients()
        closed = float(ar.sum() + ac.sum())
        assert mine.get_value(np.zeros(1))[0] == pytest.approx(
            closed, rel=1e-14
        )


def test_term_sum_and_product_match_upstream(lags):
    a_mine, a_theirs = _pair("SHOTerm")
    b_mine, b_theirs = _pair("Matern32Term")
    # Combining two kernels adds a second round of coefficient algebra, and
    # the sum crosses zero where a pure relative comparison stops meaning
    # anything, so the floor is set by the kernel's own scale.
    want = (a_theirs + b_theirs).get_value(lags)
    np.testing.assert_allclose(
        (a_mine + b_mine).get_value(lags),
        want,
        rtol=1e-9,
        atol=1e-12 * np.abs(want).max(),
    )
    # The product of an underdamped SHO with a Matern-3/2 compounds two rounds
    # of coefficient algebra, so the relative agreement is looser than a
    # single kernel evaluation.
    np.testing.assert_allclose(
        (a_mine * b_mine).get_value(lags),
        (a_theirs * b_theirs).get_value(lags),
        rtol=1e-9,
        atol=1e-12,
    )
    # The product of a SHO (underdamped, complex) with a Matern-3/2 (complex)
    # exercises the complex-by-complex branch, including the sign of bc.
    for got, want in zip(
        (a_mine * b_mine).get_coefficients(),
        (a_theirs * b_theirs).get_coefficients(),
    ):
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)


def test_term_diff_matches_upstream(lags):
    """TermDiff flips the sign of the real part and rotates the complex one."""
    a_mine, a_theirs = _pair("SHOTerm")
    np.testing.assert_allclose(
        m.TermDiff(a_mine).get_value(lags),
        ct.TermDiff(a_theirs).get_value(lags),
        rtol=RTOL,
        atol=0.0,
    )


def test_term_diff_real_branch_matches_upstream(lags):
    """A RealTerm gives a non-empty real branch, which the SHO/Matern cases
    never touch: TermDiff maps (a, c) to (-a c^2, c)."""
    r_mine, r_theirs = _pair("RealTerm")
    np.testing.assert_allclose(
        m.TermDiff(r_mine).get_value(lags),
        ct.TermDiff(r_theirs).get_value(lags),
        rtol=RTOL,
        atol=0.0,
    )
    for got, want in zip(
        m.TermDiff(r_mine).get_coefficients(),
        ct.TermDiff(r_theirs).get_coefficients(),
    ):
        np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0)


def test_shape_is_preserved(lags):
    mine, _ = _pair("SHOTerm")
    tau = lags[:60].reshape(6, 10)
    assert mine.get_value(tau).shape == (6, 10)
    assert mine.get_psd(tau).shape == (6, 10)
    np.testing.assert_allclose(
        mine.get_value(tau), mine.get_value(tau).reshape(-1).reshape(6, 10)
    )


def test_threaded_matches_serial():
    """Chunked fan-out must produce bit-identical output to a serial call.

    Each element is computed by the same scalar code regardless of which
    chunk owns it, so an off-by-one in the chunk boundaries shows up here.
    """
    mine, _ = _pair("SHOTerm")
    n = 1 << 17
    tau = np.sort(np.abs(np.random.default_rng(7).standard_normal(n)) * 4.0)
    serial = mine.get_value(tau, workers=1)
    for workers in (2, 4, 8, 16):
        np.testing.assert_array_equal(
            mine.get_value(tau, workers=workers), serial
        )


def test_two_dimensional_input_is_flattened_consistently():
    mine, theirs = _pair("SHOTerm")
    tau = np.linspace(0.0, 3.0, 120).reshape(12, 10)
    np.testing.assert_allclose(
        mine.get_value(tau), theirs.get_value(tau), rtol=RTOL, atol=0.0
    )
