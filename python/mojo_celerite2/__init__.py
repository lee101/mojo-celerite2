"""Compute-oriented subset of `celerite2` with the lag and dense loops in Mojo.

The package installs alongside the real `celerite2`; it never imports it.
"""

from __future__ import annotations

from . import dense
from ._lib import (
    cholesky,
    cho_solve,
    conv_value,
    get_psd,
    get_value,
    to_dense,
)
from .terms import (
    ComplexTerm,
    Matern32Term,
    RealTerm,
    RotationTerm,
    SHOTerm,
    Term,
    TermConvolution,
    TermDiff,
    TermProduct,
    TermSum,
)

__version__ = "0.1.0"

__all__ = [
    "ComplexTerm",
    "Matern32Term",
    "RealTerm",
    "RotationTerm",
    "SHOTerm",
    "Term",
    "TermConvolution",
    "TermDiff",
    "TermProduct",
    "TermSum",
    "cholesky",
    "cho_solve",
    "conv_value",
    "dense",
    "get_psd",
    "get_value",
    "to_dense",
]
