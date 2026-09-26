"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory.  Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-celerite2.so"

_I64 = ctypes.c_int64
_F64 = ctypes.c_double

# Below this many elements the thread fan-out costs more than it saves.
_PARALLEL_MIN = 1 << 15


# Parameter count of every exported kernel, checked against the declared
# argtypes at load time.  A short argtypes list is silent: ctypes converts the
# surplus argument with its default `c_int` rule, which truncates a 64-bit
# address to 32 bits and only crashes once a buffer lands above 4 GiB.
_ARITY = {
    "ct2_value": 12,
    "ct2_psd": 12,
    "ct2_to_dense": 14,
    "ct2_conv_value": 13,
    "ct2_cholesky": 3,
    "ct2_cho_solve": 4,
}

def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    i64, f64, ci = _I64, _F64, ctypes.c_int

    def sig(name, restype, argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        if len(fn.argtypes) != _ARITY[name]:
            raise RuntimeError(
                f"{name}: declared {len(fn.argtypes)} argtypes, kernel takes "
                f"{_ARITY[name]}"
            )

    sig("ct2_value", None, [i64] * 12)
    sig("ct2_psd", None, [i64] * 12)
    sig("ct2_to_dense", None, [i64] * 14)
    sig("ct2_conv_value", None, [i64] * 3 + [f64] + [i64] * 9)
    sig("ct2_cholesky", ci, [i64] * 3)
    sig("ct2_cho_solve", None, [i64] * 4)
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _coeff_arrays(coeffs):
    """Normalise a celerite2 6-tuple of coefficient arrays to contiguous float64."""
    return tuple(
        np.ascontiguousarray(c, dtype=np.float64) for c in coeffs
    )


def _chunks(n, workers):
    """Split [0, n) into contiguous parts, or None when running serially is better."""
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if workers <= 1 or n < _PARALLEL_MIN:
        return None
    step = -(-n // workers)
    return [(lo, min(lo + step, n)) for lo in range(0, n, step)]


def _run(call, n, workers=None):
    parts = _chunks(n, workers)
    if parts is None:
        call(0, n)
        return
    with ThreadPoolExecutor(max_workers=len(parts)) as ex:
        list(ex.map(lambda p: call(*p), parts))


def get_value(coeffs, tau, workers=None) -> np.ndarray:
    """Semiseparable kernel value at |tau|; mirrors `Term.get_value`."""
    ar, cr, ac, bc, cc, dc = _coeff_arrays(coeffs)
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    shape = tau.shape
    flat = np.ascontiguousarray(tau.reshape(-1))
    out = np.empty(flat.size, dtype=np.float64)
    call = lambda lo, hi: lib.ct2_value(  # noqa: E731
        lo, hi, _addr(flat), _addr(ar), ar.size, _addr(cr),
        _addr(ac), ac.size, _addr(bc), _addr(cc), _addr(dc), _addr(out),
    )
    _run(call, flat.size, workers)
    return out.reshape(shape)


def get_psd(coeffs, omega, workers=None) -> np.ndarray:
    """Power spectral density; mirrors `Term.get_psd`."""
    ar, cr, ac, bc, cc, dc = _coeff_arrays(coeffs)
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    shape = omega.shape
    flat = np.ascontiguousarray(omega.reshape(-1))
    out = np.empty(flat.size, dtype=np.float64)
    call = lambda lo, hi: lib.ct2_psd(  # noqa: E731
        lo, hi, _addr(flat), _addr(ar), ar.size, _addr(cr),
        _addr(ac), ac.size, _addr(bc), _addr(cc), _addr(dc), _addr(out),
    )
    _run(call, flat.size, workers)
    return out.reshape(shape)


def to_dense(coeffs, x, diag, workers=None) -> np.ndarray:
    ar, cr, ac, bc, cc, dc = _coeff_arrays(coeffs)
    x = np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64)
    diag = np.ascontiguousarray(np.atleast_1d(diag), dtype=np.float64)
    n = x.size
    k = np.empty(n * n, dtype=np.float64)
    call = lambda r0, r1: lib.ct2_to_dense(  # noqa: E731
        r0, r1, n, _addr(x), _addr(ar), ar.size, _addr(cr),
        _addr(ac), ac.size, _addr(bc), _addr(cc), _addr(dc),
        _addr(diag), _addr(k),
    )
    parts = _chunks(n, workers) if n * n >= _PARALLEL_MIN else None
    if parts is None:
        call(0, n)
    else:
        with ThreadPoolExecutor(max_workers=len(parts)) as ex:
            list(ex.map(lambda p: call(*p), parts))
    return k.reshape(n, n)


def conv_value(coeffs, tau0, delta, workers=None) -> np.ndarray:
    """Boxcar-convolved kernel; mirrors `TermConvolution.get_value`."""
    ar, cr, ac, bc, cc, dc = _coeff_arrays(coeffs)
    tau0 = np.atleast_1d(np.asarray(tau0, dtype=np.float64))
    shape = tau0.shape
    flat = np.ascontiguousarray(tau0.reshape(-1))
    out = np.empty(flat.size, dtype=np.float64)
    call = lambda lo, hi: lib.ct2_conv_value(  # noqa: E731
        lo, hi, _addr(flat), ctypes.c_double(float(delta)),
        _addr(ar), ar.size, _addr(cr), _addr(ac), ac.size,
        _addr(bc), _addr(cc), _addr(dc), _addr(out),
    )
    _run(call, flat.size, workers)
    return out.reshape(shape)


def cholesky(k: np.ndarray):
    """In-place Cholesky, returning the lower factor and log|det|."""
    k = np.ascontiguousarray(k, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("'k' must be square")
    n = k.shape[0]
    logdet = np.zeros(1, dtype=np.float64)
    if n == 0:
        return k, 0.0
    if lib.ct2_cholesky(n, _addr(k), _addr(logdet)) != 0:
        raise np.linalg.LinAlgError("matrix is not positive definite")
    return k, float(logdet[0])


def cho_solve(l: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve L L^T x = y given the lower Cholesky factor."""
    l = np.ascontiguousarray(l, dtype=np.float64)
    y = np.ascontiguousarray(np.atleast_1d(y), dtype=np.float64)
    if y.size != l.shape[0]:
        raise ValueError("dimension mismatch")
    out = np.empty_like(y)
    lib.ct2_cho_solve(l.shape[0], _addr(l), _addr(y), _addr(out))
    return out
