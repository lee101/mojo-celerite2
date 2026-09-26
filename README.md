# mojo-celerite2

`mojo-celerite2` is the compute-oriented subset of
[celerite2](https://celerite2.readthedocs.io/) — Foreman-Mackey's 1-D Gaussian
process solver for radial-velocity time series — with the kernel evaluation and
dense linear algebra loops implemented in Mojo and callable from Python.

The Python package is named `mojo_celerite2`, so it installs alongside the real
`celerite2` and the tests compare the two directly.

```python
import numpy as np
import mojo_celerite2 as m

term = m.SHOTerm(sigma=1.5, tau=2.345, rho=3.4)
tau = np.linspace(0.0, 10.0, 1000)

term.get_value(tau)          # kernel value at each lag
term.get_psd(tau)            # power spectral density at each frequency
term.to_dense(t, np.zeros_like(tau))   # N x N covariance matrix
m.dense.log_likelihood(term, t, y, yerr=e)   # exact GP log marginal likelihood
```

## What is actually in celerite2

Worth being explicit, because the name suggests otherwise: `celerite2` is **not**
a geospatial or spatial-indexing package. It has nothing to do with R-trees,
spatial hashing or haversine distances. It is a semiseparable 1-D Gaussian
process library. The numeric core is

```
k(tau) = sum_j  a_j exp(-c_j tau)                                     (real terms)
       + sum_j  exp(-c_j tau) (a_j cos(d_j tau) + b_j sin(d_j tau))     (damped oscillators)
```

plus its Fourier transform (the PSD) and the linear algebra needed to condition
a GP on data. That is what this port covers.

## Covered subset

| area | implemented API | Mojo kernel |
| --- | --- | --- |
| Kernel evaluation over lags | `Term.get_value(tau)`, `get_value` | `ct2_value` |
| Power spectral density | `Term.get_psd(omega)`, `get_psd` | `ct2_psd` |
| Dense covariance assembly | `Term.to_dense(x, diag)`, `to_dense` | `ct2_to_dense` |
| Boxcar-convolved kernel | `TermConvolution.get_value` / `.get_psd`, `conv_value` | `ct2_conv_value` |
| Term algebra | `TermSum`, `TermProduct`, `TermDiff`, `TermConvolution` coefficient rules | — (scalar NumPy) |
| Term classes | `RealTerm`, `ComplexTerm`, `SHOTerm`, `Matern32Term`, `RotationTerm` | — (scalar NumPy) |
| Dense Cholesky + log determinant | `cholesky`, `dense.solve`, `cho_solve` | `ct2_cholesky`, `ct2_cho_solve` |
| Dense exact GP posterior | `dense.log_likelihood`, `dense.predict_mean`, `dense.covariance` | built on the above |

**Not implemented, and why.** The O(N) semiseparable machinery — `get_celerite_matrices`,
`Term.dot`, and the `compute` / `apply_inverse` / `log_likelihood` / `predict` /
`sample` methods of `GaussianProcess` — lives in `celerite2.driver`, a compiled
C extension. It is not Python and it is not a loop this port can usefully
reimplement, so it stays with the real package. `pymc`, `pymc3`, `jax` and
`theano` terms are thin wrappers over it. `OriginalCeleriteTerm` adapts
`celerite.terms` objects and is left upstream. There is no mean-function or
`jitter` support: `dense.log_likelihood` is explicitly the zero-mean model.

`dense` is the one API here that upstream does not have. It exists because it
is the honest way to *check* the kernels above: the port assembles the
covariance with its own O(N^2) loop and factors it with its own Cholesky, and
those must agree with celerite2's O(N) solver. It is O(N^3) and is not meant
for large N; celerite2 itself is the right tool there.

The numerical contract is C-contiguous `float64`. Inputs are widened or
narrowed to `float64` by `np.asarray(..., dtype=np.float64)`; complex dtypes
and `float32` inputs are converted, not rejected, and the output is always
`float64`.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-celerite2.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task. The tests additionally need the
real `celerite2` installed, since they compare against it.

## Performance

Best-of-five wall clock, same process, against vectorised NumPy. Every case
verifies numerical agreement before timing. Mojo is reported twice because the
shim fans out over a thread pool for large inputs: the serial column says
whether the kernel is good, the 8-thread column says what a caller gets.

| case | numpy | mojo serial | ratio | mojo x8 | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `get_value` n=4194304 | 625 ms | 433 ms | 1.44x | 172 ms | 3.64x |
| `get_psd` n=4194304 | 185 ms | 49 ms | 3.78x | 49 ms | 3.80x |
| `to_dense` n=1024 | 159 ms | 182 ms | 0.87x | 87 ms | 1.83x |
| `cholesky` n=1500 | 1207 ms | 2292 ms | 0.53x | — | — |

Reproduce with `pixi run bench`.

Honest reading of these numbers:

- `get_value` is transcendental-bound, so it is compute bound rather than
  bandwidth bound and the thread fan-out pays: about 3.6x over NumPy end to
  end, 1.4x from the kernel alone.
- `get_psd` is a rational function of `omega^2` with no transcendentals, and
  the serial kernel is already ~3.8x NumPy. Threading adds nothing here.
- `to_dense` is the clearest case: the N x N lag grid is bandwidth bound, the
  serial Mojo loop is at or below NumPy's broadcast, and the win comes from
  splitting the rows across threads.
- `cholesky` **loses**, by about 2x. The kernel is unblocked, right-looking
  and scalar, while `numpy.linalg.cholesky` is LAPACK's blocked, tuned
  implementation. This is a real loss and it is reported as one. The Cholesky
  exists for parity checking, not for speed; for large N use scipy's
  `cho_factor`, which is what celerite2's users would reach for anyway.

These timings were taken on a shared 36-core box with roughly thirty other
builds running. Run-to-run variance on the *reference* column reached 2x, so
treat single digits as indicative, not precise.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-celerite2.so`.

The `python/mojo_celerite2` layer owns every array. It normalises inputs to
contiguous `float64`, then makes one call into the kernel per loop. Buffers
cross the C ABI as 64-bit addresses and are reconstructed in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric. Each loop kernel takes a `[lo, hi)` range so the shim can split
the work over a `ThreadPoolExecutor`; ctypes releases the GIL for the foreign
call, so that fan-out is real parallelism.

### The FFI trap this port hit

The first version of `_lib.py` declared eleven `argtypes` for a kernel with
twelve parameters. ctypes converts a surplus argument with its default `c_int`
rule, so `out_addr` was truncated to 32 bits. Every test still passed, because
small NumPy allocations sit below 4 GiB on this box. It only surfaced when a
4M-element array — served by `mmap`, and therefore above 4 GiB — segfaulted.

Two things guard against it now: `_load` refuses to bind a symbol whose
declared arity differs from the kernel signature, and `tests/test_ffi.py`
maps real buffers at a fixed 8 GiB address and runs the kernels over them.

## Tests

```
$ PYTHONPATH=python pytest tests -q
81 passed
```

`tests/test_terms.py` compares `get_value`, `get_psd` and the coefficient
algebra against `celerite2.terms` for all five term classes.
`tests/test_convolution.py` does the same for the boxcar convolution, and
additionally checks that the two closed-form branches join continuously at
`tau == delta` — a sign error in the `tau < delta` correction moves one side
only, so the jump does not shrink with the probe distance.
`tests/test_dense.py` checks `to_dense` against upstream, the Cholesky against
`numpy.linalg.cholesky`, and the dense log marginal likelihood and posterior
mean against `celerite2.numpy.GaussianProcess` — the port's O(N^2) + O(N^3)
route and celerite2's O(N) route agreeing on the same answer.

Mojo emits FMA, so results match NumPy and celerite2 only to a tolerance, never
bit for bit. `assert_allclose` with a justified `rtol` is used throughout.
Exact equality (`assert_array_equal`) is used only where the operation is
genuinely index-like: chunked and serial runs of the same kernel, the symmetry
of `to_dense`, and the two sides of a `|tau|` fold.

## License

MIT
