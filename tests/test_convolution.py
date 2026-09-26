"""Parity tests for the boxcar-convolution kernel (`TermConvolution`).

The convolved kernel has two closed forms that meet at tau == delta.  A wrong
sign in either branch, or a `>` where upstream has `>=`, is invisible on one
side of delta and shows up either in the parity comparison or in the
continuity check below.
"""

import numpy as np
import pytest

import celerite2.terms as ct

import mojo_celerite2 as m

RTOL = 1e-10

BASE = [
    ("SHOTerm", m.SHOTerm, ct.SHOTerm),
    ("Matern32Term", m.Matern32Term, ct.Matern32Term),
    ("RealTerm", m.RealTerm, ct.RealTerm),
    ("ComplexTerm", m.ComplexTerm, ct.ComplexTerm),
]


def _conv_pair(name, delta):
    for label, mine_cls, theirs_cls in BASE:
        if label == name:
            p = theirs_cls.get_test_parameters()
            return (
                m.TermConvolution(mine_cls(**p), delta),
                ct.TermConvolution(theirs_cls(**p), delta),
            )
    raise KeyError(name)


@pytest.mark.parametrize("delta", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("name,_,__", BASE, ids=[b[0] for b in BASE])
def test_conv_value_matches_upstream(name, _, __, delta):
    mine, theirs = _conv_pair(name, delta)
    tau = np.linspace(0.0, 3.0 * delta, 151)
    got = mine.get_value(tau)
    want = theirs.get_value(tau)
    # The convolved kernel crosses zero, so a pure relative test is meaningless
    # near the crossing; the absolute floor is a small fraction of the kernel's
    # own scale rather than of each sample.
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * np.abs(want).max()
    )


@pytest.mark.parametrize("delta", [0.25, 1.0])
@pytest.mark.parametrize("name,_,__", BASE, ids=[b[0] for b in BASE])
def test_conv_value_is_continuous_at_delta(name, _, __, delta):
    """The two branches of the closed form must join at tau == delta.

    A sign error in the `tau < delta` correction term shifts one side only, so
    the jump does not shrink with the probe distance; a correct implementation
    shrinks it linearly.  Both halves are asserted.
    """
    mine, _ = _conv_pair(name, delta)
    at = mine.get_value(np.array([delta]))[0]
    scale = max(abs(at), 1e-12)

    def gap(eps):
        below = mine.get_value(np.array([delta - eps]))[0]
        above = mine.get_value(np.array([delta + eps]))[0]
        return abs(below - above)

    wide, narrow = 1e-7 * delta, 1e-9 * delta
    # A genuine jump would be O(scale); continuity leaves only the O(eps) kink.
    assert gap(narrow) < 1e-5 * scale
    assert 20.0 < gap(wide) / gap(narrow) < 200.0


@pytest.mark.parametrize("delta", [0.25, 1.0])
@pytest.mark.parametrize("name,_,__", BASE, ids=[b[0] for b in BASE])
def test_conv_psd_matches_upstream(name, _, __, delta):
    mine, theirs = _conv_pair(name, delta)
    omega = np.linspace(0.0, 6.0, 121)
    np.testing.assert_allclose(
        mine.get_psd(omega), theirs.get_psd(omega), rtol=RTOL, atol=0.0
    )


def test_conv_psd_at_zero_is_finite():
    """sinc(0) is the limit 1, not a 0/0 division."""
    mine, _ = _conv_pair("SHOTerm", 0.5)
    assert np.isfinite(mine.get_psd(np.array([0.0]))[0])


def test_conv_coefficients_match_upstream():
    """`get_coefficients` on the convolution is the semiseparable form of the
    same kernel; upstream's two evaluations differ slightly, so the port is
    held to upstream rather than to its own coefficients."""
    p = ct.SHOTerm.get_test_parameters()
    mine = m.TermConvolution(m.SHOTerm(**p), 0.75)
    theirs = ct.TermConvolution(ct.SHOTerm(**p), 0.75)
    for got, want in zip(mine.get_coefficients(), theirs.get_coefficients()):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)


def test_convolution_must_be_outermost():
    inner = m.SHOTerm(**ct.SHOTerm.get_test_parameters())
    with pytest.raises(TypeError):
        m.TermConvolution(inner, 1.0) + m.RealTerm(a=1.0, c=1.0)
