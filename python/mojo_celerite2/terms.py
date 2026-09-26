"""Celerite2 terms with the lag/dense evaluation loops in Mojo.

The coefficient algebra (the `(ar, cr, ac, bc, cc, dc)` 6-tuple and the sum,
product, derivative and convolution rules) is scalar bookkeeping and stays
here in NumPy.  Every loop over lags or over the dense N x N grid goes through
`mojo_celerite2._lib`, i.e. into `src/kernels.mojo`.

The API mirrors `celerite2.terms` so a test can hand the same parameters to
both packages and compare.
"""

from __future__ import annotations

from itertools import chain, product

import numpy as np

from . import _lib

__all__ = [
    "Term",
    "TermSum",
    "TermProduct",
    "TermDiff",
    "TermConvolution",
    "RealTerm",
    "ComplexTerm",
    "SHOTerm",
    "Matern32Term",
    "RotationTerm",
]


class Term:
    """Abstract base term.  Subclasses implement :meth:`get_coefficients`."""

    def __add__(self, b):
        return TermSum(self, b)

    def __mul__(self, b):
        return TermProduct(self, b)

    def get_coefficients(self):
        raise NotImplementedError("subclasses must implement get_coefficients")

    @property
    def terms(self):
        return [self]

    def get_value(self, tau, workers=None):
        """Kernel value at the lags `tau` (the sign is discarded, as upstream)."""
        return _lib.get_value(self.get_coefficients(), tau, workers=workers)

    def get_psd(self, omega, workers=None):
        """Power spectral density at the angular frequencies `omega`."""
        return _lib.get_psd(self.get_coefficients(), omega, workers=workers)

    def to_dense(self, x, diag, workers=None):
        """Dense N x N covariance matrix for this term at coordinates `x`."""
        return _lib.to_dense(self.get_coefficients(), x, diag, workers=workers)


class TermSum(Term):
    def __init__(self, *terms):
        for term in terms:
            if isinstance(term, TermConvolution):
                raise TypeError(
                    "a TermConvolution must be the outer term in the kernel"
                )
        self._terms = terms

    @property
    def terms(self):
        return self._terms

    def get_coefficients(self):
        coeffs = (t.get_coefficients() for t in self.terms)
        return tuple(np.concatenate(c) for c in zip(*coeffs))


class TermProduct(Term):
    def __init__(self, term1, term2):
        if isinstance(term1, TermConvolution) or isinstance(
            term2, TermConvolution
        ):
            raise TypeError(
                "a TermConvolution must be the outer term in the kernel"
            )
        self.term1 = term1
        self.term2 = term2

    @property
    def terms(self):
        return [self.term1, self.term2]

    def get_coefficients(self):
        c1 = self.term1.get_coefficients()
        c2 = self.term2.get_coefficients()

        ar, cr = [], []
        for (aj, cj), (ak, ck) in product(zip(c1[0], c1[1]), zip(c2[0], c2[1])):
            ar.append(aj * ak)
            cr.append(cj + ck)

        ac, bc, cc, dc = [], [], [], []
        gen = product(zip(c1[0], c1[1]), zip(*c2[2:]))
        gen = chain(gen, product(zip(c2[0], c2[1]), zip(*c1[2:])))
        for (aj, cj), (ak, bk, ck, dk) in gen:
            ac.append(aj * ak)
            bc.append(aj * bk)
            cc.append(cj + ck)
            dc.append(dk)

        for (aj, bj, cj, dj), (ak, bk, ck, dk) in product(
            zip(*c1[2:]), zip(*c2[2:])
        ):
            ac.append(0.5 * (aj * ak + bj * bk))
            bc.append(0.5 * (bj * ak - aj * bk))
            cc.append(cj + ck)
            dc.append(dj - dk)

            ac.append(0.5 * (aj * ak - bj * bk))
            bc.append(0.5 * (bj * ak + aj * bk))
            cc.append(cj + ck)
            dc.append(dj + dk)

        return list(map(np.array, (ar, cr, ac, bc, cc, dc)))


class TermDiff(Term):
    """First derivative of a term with respect to the lag."""

    def __init__(self, term):
        if isinstance(term, TermConvolution):
            raise TypeError(
                "a TermConvolution must be the outer term in the kernel"
            )
        self.term = term

    @property
    def terms(self):
        return [self.term]

    def get_coefficients(self):
        coeffs = self.term.get_coefficients()
        a, b, c, d = coeffs[2:]
        return [
            -coeffs[0] * coeffs[1] ** 2,
            coeffs[1],
            a * (d**2 - c**2) + 2 * b * c * d,
            b * (d**2 - c**2) - 2 * a * c * d,
            c,
            d,
        ]


class TermConvolution(Term):
    """Integral of another term over a boxcar of width ``delta``."""

    def __init__(self, term, delta):
        if not isinstance(term, Term):
            raise TypeError("the inner term must be a Term")
        self.term = term
        self.delta = float(delta)

    @property
    def terms(self):
        return [self.term]

    def get_coefficients(self):
        ar, cr, a, b, c, d = self.term.get_coefficients()
        crd = cr * self.delta
        coeffs = [2 * ar * (np.cosh(crd) - 1) / crd**2, cr]

        cd = c * self.delta
        dd = d * self.delta
        c2 = c**2
        d2 = d**2
        factor = 2.0 / (self.delta * (c2 + d2)) ** 2
        cos_term = np.cosh(cd) * np.cos(dd) - 1
        sin_term = np.sinh(cd) * np.sin(dd)

        c1 = a * (c2 - d2) + 2 * b * c * d
        c2c = b * (c2 - d2) - 2 * a * c * d

        coeffs += [
            factor * (c1 * cos_term - c2c * sin_term),
            factor * (c2c * cos_term + c1 * sin_term),
            c,
            d,
        ]
        return coeffs

    def get_value(self, tau0, workers=None):
        return _lib.conv_value(
            self.term.get_coefficients(), tau0, self.delta, workers=workers
        )

    def get_psd(self, omega, workers=None):
        omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
        psd0 = self.term.get_psd(omega, workers=workers)
        arg = 0.5 * self.delta * omega
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc = np.where(arg != 0.0, np.sin(arg) / np.where(arg != 0.0, arg, 1.0), 1.0)
        return psd0 * sinc**2


class RealTerm(Term):
    r"""k(tau) = a exp(-c tau)."""

    @staticmethod
    def get_test_parameters():
        return dict(a=1.5, c=0.7)

    def __init__(self, *, a, c):
        self.a = float(a)
        self.c = float(c)

    def get_coefficients(self):
        e = np.empty(0)
        return np.array([self.a]), np.array([self.c]), e, e, e, e


class ComplexTerm(Term):
    r"""k(tau) = exp(-c tau) (a cos(d tau) + b sin(d tau))."""

    @staticmethod
    def get_test_parameters():
        return dict(a=1.5, b=0.7, c=0.7, d=0.5)

    def __init__(self, *, a, b, c, d):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def get_coefficients(self):
        e = np.empty(0)
        return (
            e,
            e,
            np.array([self.a]),
            np.array([self.b]),
            np.array([self.c]),
            np.array([self.d]),
        )


class SHOTerm(Term):
    r"""Stochastically driven damped harmonic oscillator.

    Accepts either the physical parameters ``sigma, tau, rho`` or the
    equivalent ``S0, w0, Q``, exactly one of each set, as upstream does.
    """

    def __init__(self, *, eps=1e-5, **kwargs):
        def pick(name, alt, compute):
            names = (name, alt)
            if sum(k in kwargs for k in names) != 1:
                raise ValueError(f"exactly one of {set(names)} must be defined")
            if name in kwargs:
                return float(kwargs.pop(name))
            return compute(float(kwargs.pop(alt)))

        # Order matters: w0 must exist before Q, and both before S0.
        self.w0 = pick("w0", "rho", lambda rho: 2 * np.pi / rho)
        self.Q = pick("Q", "tau", lambda tau: 0.5 * self.w0 * tau)
        self.S0 = pick(
            "S0", "sigma", lambda sigma: sigma**2 / (self.w0 * self.Q)
        )
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        self.eps = float(eps)

    @staticmethod
    def get_test_parameters():
        return dict(sigma=1.5, tau=2.345, rho=3.4)

    def overdamped(self):
        Q = self.Q
        f = np.sqrt(max(1.0 - 4.0 * Q**2, self.eps))
        e = np.empty(0)
        return (
            0.5 * self.S0 * self.w0 * Q * np.array([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * np.array([1.0 - f, 1.0 + f]),
            e,
            e,
            e,
            e,
        )

    def underdamped(self):
        Q = self.Q
        f = np.sqrt(max(4.0 * Q**2 - 1.0, self.eps))
        a = self.S0 * self.w0 * Q
        c = 0.5 * self.w0 / Q
        e = np.empty(0)
        return (
            e,
            e,
            np.array([a]),
            np.array([a / f]),
            np.array([c]),
            np.array([c * f]),
        )

    def get_coefficients(self):
        return self.overdamped() if self.Q < 0.5 else self.underdamped()


class Matern32Term(Term):
    r"""Celerite approximation to a Matern-3/2 kernel."""

    def __init__(self, *, sigma, rho, eps=0.01):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.eps = float(eps)

    @staticmethod
    def get_test_parameters():
        return dict(sigma=1.5, rho=2.345)

    def get_coefficients(self):
        w0 = np.sqrt(3) / self.rho
        S0 = self.sigma**2 / w0
        e = np.empty(0)
        return (
            e,
            e,
            np.array([w0 * S0]),
            np.array([w0**2 * S0 / self.eps]),
            np.array([w0]),
            np.array([self.eps]),
        )


class RotationTerm(TermSum):
    r"""Two SHOTerm modes, one at ``period`` and one at ``0.5 * period``."""

    def __init__(self, *, sigma, period, Q0, dQ, f):
        self.sigma = float(sigma)
        self.period = float(period)
        self.Q0 = float(Q0)
        self.dQ = float(dQ)
        self.f = float(f)

        self.amp = self.sigma**2 / (1 + self.f)

        Q1 = 0.5 + self.Q0 + self.dQ
        w1 = 4 * np.pi * Q1 / (self.period * np.sqrt(4 * Q1**2 - 1))
        S1 = self.amp / (w1 * Q1)

        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * np.sqrt(4 * Q2**2 - 1))
        S2 = self.f * self.amp / (w2 * Q2)

        super().__init__(
            SHOTerm(S0=S1, w0=w1, Q=Q1), SHOTerm(S0=S2, w0=w2, Q=Q2)
        )

    @staticmethod
    def get_test_parameters():
        return dict(sigma=1.5, period=3.45, Q0=1.3, dQ=1.05, f=0.5)
