import numpy as np
import numba as nb

# ==========================================================================
# Primality
# ==========================================================================

_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _miller_rabin_witness(n: int, a: int) -> bool:
    """Returns True if n passes the Miller-Rabin test for witness a."""
    if n < 2:
        return False
    if n == a:
        return True
    if n % a == 0:
        return False

    # Write n-1 = 2^r * d with d odd
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True

    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True

    return False


def _is_prime(n: int) -> bool:
    """Deterministic primality test for n < 3.3e24."""
    if n < 2:
        return False
    for w in _WITNESSES:
        if n == w:
            return True
        if n % w == 0:
            return False
    return all(_miller_rabin_witness(n, w) for w in _WITNESSES)


def largest_prime_below_2n(n: int) -> int:
    """Find largest prime p < 2^n. Requires n >= 2."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    candidate = (1 << n) - 1
    while candidate >= 2:
        if _is_prime(candidate):
            return candidate
        candidate -= 2
    raise ValueError(f"No prime found below 2^{n}")


# ==========================================================================
# Define Field
# ==========================================================================


def make_field(N: int):
    if not 2 <= N <= 16:
        raise ValueError(f"Supported N in [2, 16], got={N}")
    q = largest_prime_below_2n(N)
    return q, N


# ==========================================================================
# Arithmetic
# ==========================================================================


@nb.njit(inline="always", fastmath=True)
def add(a: np.uint16, b: np.uint16, q: int) -> np.uint16:
    return np.uint16((np.uint32(a) + np.uint32(b)) % np.uint32(q))


@nb.njit(inline="always", fastmath=True)
def neg(a: np.uint16, q: int) -> np.uint16:
    return np.uint16((np.uint32(q) - np.uint32(a)) % np.uint32(q))


@nb.njit(inline="always", fastmath=True)
def mul(a: np.uint16, b: np.uint16, q: int) -> np.uint16:
    return np.uint16((np.uint32(a) * np.uint32(b)) % np.uint32(q))


@nb.njit(inline="always", fastmath=True)
def _pow(base: np.uint16, e: int, q: int) -> np.uint16:
    if e < 0:
        raise ValueError("negative exponent")
    if e == 0:
        return np.uint16(1)
    res = np.uint16(1)
    b = base
    ee = e
    while ee > 0:
        if ee & 1:
            res = mul(res, b, q)
        ee >>= 1
        if ee:
            b = mul(b, b, q)
    return res


@nb.njit(inline="always", fastmath=True)
def inv(a: np.uint16, q: int) -> np.uint16:
    if a == 0:
        return np.uint16(0)
    return _pow(a, q - 2, q)


# ==========================================================================
# Polynomial Utils
# ==========================================================================


@nb.njit(inline="always")
def interpolate_newton_poly(
    y: np.ndarray,  # (m,) uint16, values f(x[i])
    x: np.ndarray,  # (m,) uint16, distinct x[i]
    q: int,
) -> np.ndarray:
    """
    Newton divided differences over GF(q).
    Returns Newton coefficients a0..a_{m-1}.
    """
    m = x.shape[0]
    if y.shape[0] != m:
        raise ValueError("x and y must have same length")

    dd = y.copy()

    for k in range(1, m):
        for i in range(m - 1, k - 1, -1):
            num = add(dd[i], neg(dd[i - 1], q), q)
            dx = add(x[i], neg(x[i - k], q), q)
            if dx == 0:
                raise ValueError(
                    "duplicate x nodes (division by zero in interpolation)"
                )
            dd[i] = mul(num, inv(dx, q), q)

    return dd


@nb.njit(cache=True)
def newton_to_poly(
    coeffs: np.ndarray,  # (m,) uint16: Newton coeffs
    basis: np.ndarray,  # (m,) uint16: nodes x0..x_{m-1}
    q: int,
) -> np.ndarray:
    """
    Convert Newton form to monomial coefficients c[0..m-1]:
        f(t) = c0 + c1*t + c2*t^2 + ... + c_{m-1}*t^{m-1}
    """
    m = coeffs.shape[0]
    if basis.shape[0] != m:
        raise ValueError("coeffs and basis must have same length")

    out = np.zeros(m, dtype=np.uint16)
    tmp = np.zeros(m, dtype=np.uint16)

    if m == 0:
        return out

    deg = 0
    out[0] = np.uint16(coeffs[m - 1])

    for k in range(m - 2, -1, -1):
        alpha = basis[k]
        minus_alpha = neg(alpha, q)

        for i in range(deg + 2):
            tmp[i] = np.uint16(0)

        tmp[0] = mul(out[0], minus_alpha, q)
        for j in range(1, deg + 1):
            term = mul(out[j], minus_alpha, q)
            tmp[j] = add(out[j - 1], term, q)
        tmp[deg + 1] = out[deg]

        tmp[0] = add(tmp[0], coeffs[k], q)

        for i in range(deg + 2):
            out[i] = tmp[i]
        deg += 1

    return out


@nb.njit(cache=True)
def interpolate_poly(
    y: np.ndarray,  # (m,) uint16, values f(x[i])
    x: np.ndarray,  # (m,) uint16, distinct x[i]
    q: int,
) -> np.ndarray:
    dd = interpolate_newton_poly(y, x, q)
    return newton_to_poly(dd, x, q)


@nb.njit(cache=True)
def evaluate_poly(
    t: np.uint16,
    coeffs: np.ndarray,  # shape (m,), dtype uint16  (c0..c_{m-1})
    q: int,
) -> np.uint16:
    """Evaluate monomial-basis polynomial via Horner."""
    m = coeffs.shape[0]
    if m == 0:
        return np.uint16(0)

    acc = np.uint16(coeffs[m - 1])
    for k in range(m - 2, -1, -1):
        acc = add(mul(acc, t, q), coeffs[k], q)

    return acc
