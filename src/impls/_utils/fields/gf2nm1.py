import numpy as np
import numba as nb

from ..intmath import ispowprime_1_15

# ==========================================================================
# Define Field
# ==========================================================================


def make_feld(N: int):
    if not 1 <= N <= 15:
        raise ValueError(f"Spupported N in [1, 15], got={N}")  # due to uint16 usage
    if not ispowprime_1_15(N):
        raise ValueError("Spupported 2^N - 1 appears to be not a prime")

    n = np.uint16(N)
    mask = np.uint16((1 << N) - 1)  # a prime

    return n, mask


@nb.njit(inline="always", fastmath=True)
def add(a: np.uint16, b: np.uint16, n: np.uint16, mask: np.uint16) -> np.uint16:
    s = a + b
    r = (s & mask) + (s >> n)
    t = r - mask
    return t + (np.uint16(r < mask) * mask)


@nb.njit(inline="always", fastmath=True)
def neg(a: np.uint16, mask: np.uint16) -> np.uint16:
    return mask - a


@nb.njit(inline="always", fastmath=True)
def mul(a: np.uint16, b: np.uint16, n: np.uint16, mask: np.uint16) -> np.uint16:
    # Widen to avoid overflow: for n<=15, (2^n-2)^2 < 2^30 fits in uint32
    x = np.uint32(a) * np.uint32(b)

    m = np.uint32(mask)

    # Mersenne reduction: x mod (2^n - 1) via end-around-carry folds
    # For x < 2^(2n) and n<=15, two folds are enough.
    x = (x & m) + (x >> n)
    x = (x & m) + (x >> n)

    # Branchless normalization into [0, mask-1] with mask representing 0
    # (i.e., if x == mask => 0; if x < mask => x; if x > mask => x-mask)
    t = np.uint32(x - m)
    x = t + (np.uint32(x < m) * m)

    return np.uint16(x)


@nb.njit(inline="always", fastmath=True)
def _pow(base: np.uint16, e: int, n: np.uint16, mask: np.uint16) -> np.uint16:
    if e < 0:
        raise ValueError("negative exponent not supported for mod (2^n-1)")
    if e == 0:
        return np.uint16(1)

    res = np.uint16(1)

    ee = e
    while ee > 0:
        if ee & 1:
            res = mul(res, base, n, mask)
        ee >>= 1
        if ee:
            base = mul(base, base, n, mask)
    return res


@nb.njit(inline="always", fastmath=True)
def inv(a: np.uint16, n: np.uint16, mask: np.uint16) -> np.uint16:
    # In a field mod prime p, 0 has no inverse
    if a == 0:
        return np.uint16(0)

    # Fermat's little theorem: a^(p-2) mod p
    # exponent fits in int for n<=15 (p<=32767)
    return _pow(a, int(mask) - 2, n, mask)


# ==========================================================================
# Polynomial Utils on this field
# ==========================================================================


@nb.njit(inline="always")
def interpolate_newton_poly(
    y: np.ndarray,  # (m,) uint16, values f(x[i])
    x: np.ndarray,  # (m,) uint16, distinct x[i]
    n: np.uint16,
    mask: np.uint16,
) -> np.ndarray:
    """
    Interpolate the unique polynomial f of degree <= m-1 such that f(x[i]) = y[i] using Newton divided differences.

    Returns:
    coeffs: (m,) uint16 newtonian coefficients a0..a_{m-1}
            so that f(t) = a0 + a1*(t-x0) + a2*(t-x0)(t-x1) + ... + a_{m-1}*Π_{j=0}^{m-2}(t-xj)

    Complexity:
    Time:  O(m^2) field ops
    Space: O(m)
    """
    m = x.shape[0]
    if y.shape[0] != m:
        raise ValueError("x and y must have same length")

    # ---------- Newton divided differences (in-place) ----------
    # dd[i] will end up holding a_i = f[x0..xi] (Newton coefficients)
    dd = y.copy()

    # Compute divided differences:
    # for k=1..m-1:
    #   for i=m-1..k:
    #       dd[i] = (dd[i] - dd[i-1]) / (x[i] - x[i-k])
    #
    # We update from the end to avoid overwriting needed dd[i-1].
    for k in range(1, m):
        for i in range(m - 1, k - 1, -1):
            num = dd[i]
            den = dd[i - 1]
            # num - den
            num = add(num, neg(den, mask), n, mask)

            # x[i] - x[i-k]
            xi = x[i]
            xj = x[i - k]
            dx = add(xi, neg(xj, mask), n, mask)
            if dx == 0:
                raise ValueError(
                    "duplicate x nodes (division by zero in interpolation)"
                )

            inv_dx = inv(dx, n, mask)
            dd[i] = mul(num, inv_dx, n, mask)

    # Now dd[i] == a_i, and Newton form is:
    # f(t) = a0 + a1*(t-x0) + a2*(t-x0)(t-x1) + ... + a_{m-1}*Π_{j=0}^{m-2}(t-xj)

    return dd


@nb.njit(cache=True)
def newton_to_poly(
    coeffs: np.ndarray,  # (m,) uint16: Newton coeffs on `basis`
    basis: np.ndarray,  # (m,) uint16: nodes x0..x_{m-1} (distinct)
    n: np.uint16,
    mask: np.uint16,
) -> np.ndarray:
    """
    Convert Newton form on `basis`:
        f(t) = a0 + a1*(t-x0) + a2*(t-x0)(t-x1) + ... + a_{m-1}*Π_{j=0}^{m-2}(t-xj)
    into monomial coefficients c[0..m-1]:
        f(t) = c0 + c1*t + c2*t^2 + ... + c_{m-1}*t^{m-1}

    Uses Horner in Newton form, maintaining monomial coefficients.
    Time O(m^2). Space O(m).
    """
    m = coeffs.shape[0]
    if basis.shape[0] != m:
        raise ValueError("coeffs and basis must have same length")

    out = np.zeros(m, dtype=np.uint16)
    tmp = np.zeros(m, dtype=np.uint16)

    if m == 0:
        return out

    # Start with P(t) = coeffs[m-1]
    deg = 0
    out[0] = np.uint16(coeffs[m - 1])

    # Horner: P <- coeffs[k] + (t - basis[k]) * P
    for k in range(m - 2, -1, -1):
        alpha = basis[k]
        minus_alpha = neg(alpha, mask)

        # Clear tmp[0..deg+1]
        for i in range(deg + 2):
            tmp[i] = np.uint16(0)

        # Multiply by (t - alpha):
        # new[0]     = (-alpha) * p0
        # new[j]     = p_{j-1} + (-alpha) * p_j   for 1<=j<=deg
        # new[deg+1] = p_deg
        tmp[0] = mul(out[0], minus_alpha, n, mask)
        for j in range(1, deg + 1):
            term = mul(out[j], minus_alpha, n, mask)
            tmp[j] = add(out[j - 1], term, n, mask)
        tmp[deg + 1] = out[deg]

        # Add constant coeffs[k]
        tmp[0] = add(tmp[0], coeffs[k], n, mask)

        # Swap tmp -> out (only needed prefix)
        for i in range(deg + 2):
            out[i] = tmp[i]
        deg += 1

    return out


@nb.njit(cache=True)
def interpolate_poly(
    y: np.ndarray,  # (m,) uint16, values f(x[i])
    x: np.ndarray,  # (m,) uint16, distinct x[i]
    n: np.uint16,
    mask: np.uint16,
) -> np.ndarray:
    dd = interpolate_newton_poly(y, x, n, mask)
    return newton_to_poly(dd, x, n, mask)


@nb.njit(cache=True)
def evaluate_poly(
    t: np.uint16,
    coeffs: np.ndarray,  # shape (m,), dtype uint16  (c0..c_{m-1})
    n: np.uint16,
    mask: np.uint16,
) -> np.uint16:
    """
    Evaluate monomial-basis polynomial:
        f(t) = c0 + c1*t + c2*t^2 + ... + c_{m-1}*t^{m-1}
    via Horner:
        acc = c_{m-1}
        for k = m-2 .. 0:
            acc = acc*t + c_k
    """
    m = coeffs.shape[0]
    if m == 0:
        return np.uint16(0)

    acc = np.uint16(coeffs[m - 1])
    for k in range(m - 2, -1, -1):
        acc = add(mul(acc, t, n, mask), coeffs[k], n, mask)

    return acc
