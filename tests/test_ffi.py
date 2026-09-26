"""FFI contract tests.

Buffers cross the C ABI as 64-bit addresses.  If a kernel's `argtypes` list is
one entry short, ctypes converts the surplus argument with its default `c_int`
rule and silently truncates the address to 32 bits.  Small NumPy allocations
usually live below 4 GiB and the truncation goes unnoticed, so these tests pin
the declared arity and then exercise the kernels on buffers mapped *above* 4 GiB.
"""

import ctypes

import numpy as np
import pytest

import mojo_celerite2 as m
from mojo_celerite2 import _lib

_HIGH = 0x2_0000_0000  # 8 GiB, comfortably outside the 32-bit range
_SIZE = 1 << 20

_PROT_READ, _PROT_WRITE = 1, 2
_MAP_PRIVATE, _MAP_ANONYMOUS = 2, 0x20


def _high_buffer(n: int) -> np.ndarray:
    """A contiguous float64 array backed by an anonymous mapping at 8 GiB."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    addr = libc.mmap(
        ctypes.c_void_p(_HIGH),
        ctypes.c_size_t(n * 8),
        _PROT_READ | _PROT_WRITE,
        _MAP_PRIVATE | _MAP_ANONYMOUS,
        -1,
        0,
    )
    if addr is None or addr == ctypes.c_void_p(-1).value:
        pytest.skip("could not map a high address on this system")
    assert addr >= 2**32, "mmap did not honour the address hint"
    return np.ctypeslib.as_array(
        ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(ctypes.c_double)),
        shape=(n,),
    )


@pytest.fixture
def high():
    return _high_buffer


def test_every_kernel_declares_its_full_arity():
    for name, arity in _lib._ARITY.items():
        fn = getattr(_lib.lib, name)
        assert len(fn.argtypes) == arity, name


def test_addresses_are_declared_64_bit():
    for name in _lib._ARITY:
        for t in getattr(_lib.lib, name).argtypes:
            assert t in (ctypes.c_int64, ctypes.c_double), name


def test_value_works_above_four_gib(high):
    term = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
    n = 1 << 16
    tau = np.abs(np.random.default_rng(3).standard_normal(n)) * 4.0

    out = high(n)
    coeffs = term.get_coefficients()
    ar, cr, ac, bc, cc, dc = (np.ascontiguousarray(c) for c in coeffs)
    flat = np.ascontiguousarray(tau.reshape(-1))
    _lib.lib.ct2_value(
        0, n, flat.ctypes.data, ar.ctypes.data, ar.size, cr.ctypes.data,
        ac.ctypes.data, ac.size, bc.ctypes.data, cc.ctypes.data,
        dc.ctypes.data, out.ctypes.data,
    )
    np.testing.assert_allclose(
        out, term.get_value(tau), rtol=0.0, atol=0.0
    )


def test_psd_works_above_four_gib(high):
    term = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
    n = 1 << 16
    omega = np.abs(np.random.default_rng(4).standard_normal(n)) * 4.0

    out = high(n)
    ar, cr, ac, bc, cc, dc = (
        np.ascontiguousarray(c) for c in term.get_coefficients()
    )
    flat = np.ascontiguousarray(omega.reshape(-1))
    _lib.lib.ct2_psd(
        0, n, flat.ctypes.data, ar.ctypes.data, ar.size, cr.ctypes.data,
        ac.ctypes.data, ac.size, bc.ctypes.data, cc.ctypes.data,
        dc.ctypes.data, out.ctypes.data,
    )
    np.testing.assert_allclose(out, term.get_psd(omega), rtol=0.0, atol=0.0)


def test_convolution_works_above_four_gib(high):
    base = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
    term = m.TermConvolution(base, 0.5)
    n = 1 << 15
    tau = np.linspace(0.0, 1.5, n)

    out = high(n)
    ar, cr, ac, bc, cc, dc = (
        np.ascontiguousarray(c) for c in base.get_coefficients()
    )
    flat = np.ascontiguousarray(tau)
    _lib.lib.ct2_conv_value(
        0, n, flat.ctypes.data, ctypes.c_double(0.5),
        ar.ctypes.data, ar.size, cr.ctypes.data, ac.ctypes.data, ac.size,
        bc.ctypes.data, cc.ctypes.data, dc.ctypes.data, out.ctypes.data,
    )
    np.testing.assert_allclose(out, term.get_value(tau), rtol=0.0, atol=0.0)


def test_large_buffers_round_trip():
    """The same thing the high-address test does, reached by ordinary allocation:
    an 8M-element array is served by mmap and lands above 4 GiB on this box."""
    n = 1 << 23
    term = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
    tau = np.abs(np.random.default_rng(5).standard_normal(n)) * 4.0
    out = np.empty(n)
    if out.ctypes.data < 2**32 and tau.ctypes.data < 2**32:
        pytest.skip("no array landed above 4 GiB")
    np.testing.assert_allclose(
        term.get_value(tau, workers=1),
        term.get_value(tau, workers=8),
        rtol=0.0,
        atol=0.0,
    )
