"""Semiseparable kernel evaluation for the celerite2 subset.

A celerite kernel is a sum of damped exponentials

    k(tau) = sum_j ar[j] * exp(-cr[j] * tau)
           + sum_j exp(-cc[j] * tau) * (ac[j]*cos(dc[j]*tau) + bc[j]*sin(dc[j]*tau))

so every term in `celerite2/terms.py` reduces to that shape once its
coefficient algebra is evaluated.  The coefficient algebra is scalar bookkeeping
and stays in Python; the loops over lags and over the dense N x N grid are what
this file implements.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.  Each loop kernel
takes a `[lo, hi)` range so the Python shim can fan chunks out over a thread
pool; these kernels are transcendental-heavy, so they are compute bound and
chunking pays.
"""

from std.math import abs, cos, cosh, exp, log, sin, sinh, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime PI: Float64 = 3.14159265358979323846
comptime SQRT_2_OVER_PI: Float64 = 0.7978845608028654


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@export("ct2_value")
def ct2_value(
    lo: Int,
    hi: Int,
    tau_addr: Int,
    ar_addr: Int,
    jr: Int,
    cr_addr: Int,
    ac_addr: Int,
    jc: Int,
    bc_addr: Int,
    cc_addr: Int,
    dc_addr: Int,
    out_addr: Int,
) abi("C"):
    """Evaluate the semiseparable kernel at |tau[i]| for i in [lo, hi)."""
    var tau = fp(tau_addr)
    var ar = fp(ar_addr)
    var cr = fp(cr_addr)
    var ac = fp(ac_addr)
    var bc = fp(bc_addr)
    var cc = fp(cc_addr)
    var dc = fp(dc_addr)
    var out = fp(out_addr)
    for i in range(lo, hi):
        var t = abs(tau[unsafe_offset=i])
        var k = Float64(0.0)
        for j in range(jr):
            k += ar[unsafe_offset=j] * exp(-cr[unsafe_offset=j] * t)
        for j in range(jc):
            var arg = dc[unsafe_offset=j] * t
            var osc = ac[unsafe_offset=j] * cos(arg) + bc[unsafe_offset=j] * sin(
                arg
            )
            k += exp(-cc[unsafe_offset=j] * t) * osc
        out[unsafe_offset=i] = k


@export("ct2_psd")
def ct2_psd(
    lo: Int,
    hi: Int,
    omega_addr: Int,
    ar_addr: Int,
    jr: Int,
    cr_addr: Int,
    ac_addr: Int,
    jc: Int,
    bc_addr: Int,
    cc_addr: Int,
    dc_addr: Int,
    out_addr: Int,
) abi("C"):
    """Power spectral density of the semiseparable kernel at omega[i]."""
    var w = fp(omega_addr)
    var ar = fp(ar_addr)
    var cr = fp(cr_addr)
    var ac = fp(ac_addr)
    var bc = fp(bc_addr)
    var cc = fp(cc_addr)
    var dc = fp(dc_addr)
    var out = fp(out_addr)
    for i in range(lo, hi):
        var w2 = w[unsafe_offset=i] * w[unsafe_offset=i]
        var acc = Float64(0.0)
        for j in range(jr):
            var cj = cr[unsafe_offset=j]
            acc += ar[unsafe_offset=j] * cj / (cj * cj + w2)
        for j in range(jc):
            var cj = cc[unsafe_offset=j]
            var dj = dc[unsafe_offset=j]
            var c2 = cj * cj
            var d2 = dj * dj
            var w02 = c2 + d2
            var num = (ac[unsafe_offset=j] * cj + bc[unsafe_offset=j] * dj) * w02 + (
                ac[unsafe_offset=j] * cj - bc[unsafe_offset=j] * dj
            ) * w2
            var den = w2 * w2 + 2.0 * (c2 - d2) * w2 + w02 * w02
            acc += num / den
        out[unsafe_offset=i] = SQRT_2_OVER_PI * acc


@export("ct2_to_dense")
def ct2_to_dense(
    row0: Int,
    row1: Int,
    n: Int,
    x_addr: Int,
    ar_addr: Int,
    jr: Int,
    cr_addr: Int,
    ac_addr: Int,
    jc: Int,
    bc_addr: Int,
    cc_addr: Int,
    dc_addr: Int,
    diag_addr: Int,
    k_addr: Int,
) abi("C"):
    """K[i, j] = k(|x[i] - x[j]|) for rows [row0, row1), plus diag on the diagonal.

    `k_addr` is a row-major n x n buffer.
    """
    var x = fp(x_addr)
    var ar = fp(ar_addr)
    var cr = fp(cr_addr)
    var ac = fp(ac_addr)
    var bc = fp(bc_addr)
    var cc = fp(cc_addr)
    var dc = fp(dc_addr)
    var diag = fp(diag_addr)
    var k = fp(k_addr)
    for i in range(row0, row1):
        var xi = x[unsafe_offset=i]
        var base = i * n
        for j in range(n):
            var t = abs(xi - x[unsafe_offset=j])
            var acc = Float64(0.0)
            for r in range(jr):
                acc += ar[unsafe_offset=r] * exp(-cr[unsafe_offset=r] * t)
            for r in range(jc):
                var arg = dc[unsafe_offset=r] * t
                var osc = ac[unsafe_offset=r] * cos(arg) + bc[unsafe_offset=r] * sin(
                    arg
                )
                acc += exp(-cc[unsafe_offset=r] * t) * osc
            k[unsafe_offset=base + j] = acc
        k[unsafe_offset=base + i] += diag[unsafe_offset=i]


@export("ct2_conv_value")
def ct2_conv_value(
    lo: Int,
    hi: Int,
    tau0_addr: Int,
    delta: Float64,
    ar_addr: Int,
    jr: Int,
    cr_addr: Int,
    ac_addr: Int,
    jc: Int,
    bc_addr: Int,
    cc_addr: Int,
    dc_addr: Int,
    out_addr: Int,
) abi("C"):
    """Boxcar-convolved kernel value at |tau0[i]| for i in [lo, hi).

    Mirrors `TermConvolution.get_value`: the closed form differs on either side
    of tau == delta, so both branches are evaluated and the lag picks one.
    """
    var tau0 = fp(tau0_addr)
    var ar = fp(ar_addr)
    var cr = fp(cr_addr)
    var ac = fp(ac_addr)
    var bc = fp(bc_addr)
    var cc = fp(cc_addr)
    var dc = fp(dc_addr)
    var out = fp(out_addr)
    for i in range(lo, hi):
        var t = abs(tau0[unsafe_offset=i])
        var dpt = delta + t
        var dmt = delta - t

        # Real exponentials: the tau > delta branch, then the tau < delta fixup.
        var k_large_real = Float64(0.0)
        var k_small = Float64(0.0)
        for j in range(jr):
            var cj = cr[unsafe_offset=j]
            var crd = cj * delta
            var norm = 2.0 * ar[unsafe_offset=j] / (crd * crd)
            k_large_real += norm * (cosh(crd) - 1.0) * exp(-cj * t)
        k_small = k_large_real
        for j in range(jr):
            var cj = cr[unsafe_offset=j]
            var crd = cj * delta
            var norm = 2.0 * ar[unsafe_offset=j] / (crd * crd)
            var crdmt = cj * dmt
            k_small += norm * (crdmt - sinh(crdmt))
        var k_large = k_large_real

        for j in range(jc):
            var cj = cc[unsafe_offset=j]
            var dj = dc[unsafe_offset=j]
            var aj = ac[unsafe_offset=j]
            var bj = bc[unsafe_offset=j]
            var c2 = cj * cj
            var d2 = dj * dj
            var c2pd2 = c2 + d2
            var cd = cj * delta
            var dd = dj * delta
            var c1 = aj * (c2 - d2) + 2.0 * bj * cj * dj
            var c2c = bj * (c2 - d2) - 2.0 * aj * cj * dj
            var norm = 1.0 / ((delta * c2pd2) * (delta * c2pd2))
            var k0 = exp(-cj * t)
            var cdt = cos(dj * t)
            var sdt = sin(dj * t)

            var cos_term = 2.0 * (cosh(cd) * cos(dd) - 1.0)
            var sin_term = 2.0 * (sinh(cd) * sin(dd))
            var factor = k0 * norm
            k_large += (c1 * cos_term - c2c * sin_term) * factor * cdt
            k_large += (c2c * cos_term + c1 * sin_term) * factor * sdt

            var edmt = exp(-cj * dmt)
            var edpt = exp(-cj * dpt)
            var cos_term2 = (
                edmt * cos(dj * dmt) + edpt * cos(dj * dpt) - 2.0 * k0 * cdt
            )
            var sin_term2 = (
                edmt * sin(dj * dmt) + edpt * sin(dj * dpt) - 2.0 * k0 * sdt
            )
            k_small += 2.0 * (aj * cj + bj * dj) * c2pd2 * dmt * norm
            k_small += (c1 * cos_term2 + c2c * sin_term2) * norm

        if t >= delta:
            out[unsafe_offset=i] = k_large
        else:
            out[unsafe_offset=i] = k_small


@export("ct2_cholesky")
def ct2_cholesky(n: Int, k_addr: Int, logdet_addr: Int) abi("C") -> Int:
    """In-place Cholesky of a row-major n x n matrix; the strict upper triangle
    of `k_addr` is zeroed, so the buffer holds L with K = L L^T.

    Returns 0 on success and -1 if the matrix is not numerically positive
    definite.  `logdet_addr` receives log|det K| = 2 * sum(log L_ii).
    """
    var k = fp(k_addr)
    var out = fp(logdet_addr)
    var logdet = Float64(0.0)
    for i in range(n):
        var d = k[unsafe_offset=i * n + i]
        for j in range(i):
            var v = k[unsafe_offset=i * n + j]
            d -= v * v
        if not (d > 0.0):
            return -1
        var lii = sqrt(d)
        k[unsafe_offset=i * n + i] = lii
        logdet += 2.0 * log(lii)
        for r in range(i + 1, n):
            var s = k[unsafe_offset=r * n + i]
            for j in range(i):
                s -= k[unsafe_offset=r * n + j] * k[unsafe_offset=i * n + j]
            k[unsafe_offset=r * n + i] = s / lii
            k[unsafe_offset=i * n + r] = 0.0
    out[unsafe_offset=0] = logdet
    return 0


@export("ct2_cho_solve")
def ct2_cho_solve(n: Int, l_addr: Int, y_addr: Int, out_addr: Int) abi("C"):
    """Solve L L^T x = y in place, given the lower Cholesky factor in `l_addr`."""
    var l = fp(l_addr)
    var y = fp(y_addr)
    var out = fp(out_addr)
    for i in range(n):
        var s = y[unsafe_offset=i]
        for j in range(i):
            s -= l[unsafe_offset=i * n + j] * out[unsafe_offset=j]
        out[unsafe_offset=i] = s / l[unsafe_offset=i * n + i]
    for i in range(n - 1, -1, -1):
        var s = out[unsafe_offset=i]
        for j in range(i + 1, n):
            s -= l[unsafe_offset=j * n + i] * out[unsafe_offset=j]
        out[unsafe_offset=i] = s / l[unsafe_offset=i * n + i]
