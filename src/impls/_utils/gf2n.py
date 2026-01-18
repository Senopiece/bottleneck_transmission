import numpy as np
import numba as nb

# ---------------------------
# Canonical polynomial book: primitive trinomials for GF(2^n), n in [1, 16]
#    p(x) = x^n + x^a + 1
# ---------------------------

PRIMITIVE_TRINOMIAL_A = {
    1: 0,  # x + 1  (special case: x^1 + 1)
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 1,
    7: 1,
    8: 1,
    9: 4,
    10: 3,
    11: 2,
    12: 1,
    13: 3,
    14: 1,
    15: 1,
    16: 10,
}


def primitive_poly_int(n: int) -> int:
    """
    Returns p as an integer bitmask, where bit i is coeff of x^i.
    Monic term x^n is always set.
    """
    if not (1 <= n <= 16):
        raise ValueError(
            "This canonical polynomial book supports packet_bitsize in [1,16]."
        )
    a = PRIMITIVE_TRINOMIAL_A[n]
    if n == 1:
        return (1 << 1) | 1  # x + 1
    return (1 << n) | (1 << a) | 1


# ---------------------------
# GF(2^n) multiply (Numba-friendly), n <= 16 assumed here
#    Uses stepwise reduction (like AES), requiring:
#      poly: full polynomial with x^n term set
#      red : poly without x^n term (low n bits)
# ---------------------------


@nb.njit(inline="always")
def gf2n_mul(
    a: np.uint16,
    b: np.uint16,
    n: int,
    mask: np.uint16,
    red: np.uint16,
):
    res = np.uint16(0)
    aa = a & mask
    bb = b & mask

    for _ in range(n):
        if bb & 1:
            res ^= aa
        bb >>= 1

        carry = (aa >> (n - 1)) & 1
        aa = (aa << 1) & mask
        if carry:
            aa ^= red

    return res & mask


@nb.njit(cache=True)
def monomials_gf2n(
    x: np.uint16,
    m: int,  # number of monomials
    n: int,
    mask: np.uint16,
    red: np.uint16,
):
    """
    m = number of monomials to compute.
    out[i] = x^i in GF(2^n), polynomial basis mod the chosen primitive poly.
    """
    acc = np.uint16(1)
    xx = np.uint16(x) & mask

    out = np.empty(m, dtype=np.uint16)

    for i in range(m):
        out[i] = acc
        acc = gf2n_mul(acc, xx, n, mask, red)  # type: ignore

    return out


@nb.njit(inline="always")
def gf2n_pow(
    a: np.uint16, e: int, n: int, mask: np.uint16, red: np.uint16
) -> np.uint16:
    res = np.uint16(1)
    base = a & mask
    ee = e
    while ee > 0:
        if ee & 1:
            res = gf2n_mul(res, base, n, mask, red)
        ee >>= 1
        if ee:
            base = gf2n_mul(base, base, n, mask, red)
    return res


@nb.njit(inline="always")
def gf2n_inv(a: np.uint16, n: int, mask: np.uint16, red: np.uint16) -> np.uint16:
    # In GF(2^n), a^{-1} = a^(2^n - 2) for a != 0
    if a == 0:
        # Numba can't raise Python exceptions cleanly in all contexts; return 0 and let caller detect.
        return np.uint16(0)
    return gf2n_pow(a, (1 << n) - 2, n, mask, red)


@nb.njit(cache=True)
def gf2n_solve_and_invert(
    X: np.ndarray,  # (m, m) uint16
    b: np.ndarray,  # (m,)   uint16, row-vector in aX=b
    n: int,
    mask: np.uint16,
    red: np.uint16,
):
    """
    Solve aX=b and compute X^{-1} over GF(2^n), with 'a' obtained during elimination.

    Method:
      Solve X^T u = b^T via Gauss-Jordan on [X^T | I | b^T].
      Then u = a^T, and (X^T)^{-1} = (X^{-1})^T.

    Returns:
      a        : (m,)   uint16
      Xinv     : (m,m)  uint16
      singular : bool
    """
    m = X.shape[0]

    # Work with transpose so the RHS updates yield a^T directly.
    A = X.T.copy()  # (m,m)

    # This will become (X^T)^{-1}
    InvT = np.zeros((m, m), dtype=np.uint16)
    for i in range(m):
        InvT[i, i] = np.uint16(1)

    # RHS column = b^T (store as length-m vector)
    rhs = b.copy()

    singular = False

    for col in range(m):
        # Find pivot
        pivot = -1
        for r in range(col, m):
            if A[r, col] != 0:
                pivot = r
                break
        if pivot == -1:
            singular = True
            break

        # Swap rows in all blocks
        if pivot != col:
            A[col, :], A[pivot, :] = A[pivot, :].copy(), A[col, :].copy()
            InvT[col, :], InvT[pivot, :] = InvT[pivot, :].copy(), InvT[col, :].copy()
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]

        # Normalize pivot row
        inv_p = gf2n_inv(A[col, col], n, mask, red)
        if inv_p == 0:
            singular = True
            break

        for j in range(m):
            A[col, j] = gf2n_mul(A[col, j], inv_p, n, mask, red)
            InvT[col, j] = gf2n_mul(InvT[col, j], inv_p, n, mask, red)
        rhs[col] = gf2n_mul(rhs[col], inv_p, n, mask, red)

        # Eliminate other rows
        for r in range(m):
            if r == col:
                continue
            factor = A[r, col]
            if factor == 0:
                continue

            for j in range(m):
                A[r, j] ^= gf2n_mul(factor, A[col, j], n, mask, red)
                InvT[r, j] ^= gf2n_mul(factor, InvT[col, j], n, mask, red)
            rhs[r] ^= gf2n_mul(factor, rhs[col], n, mask, red)

    if singular:
        # Return placeholders; caller should check singular
        return np.zeros(m, dtype=np.uint16), np.zeros((m, m), dtype=np.uint16), True

    # rhs now equals a^T, so a is rhs (same entries, just interpreted as row)
    a = rhs

    # Xinv = (InvT)^T
    Xinv = InvT.T.copy()

    return a, Xinv, False


@nb.njit(cache=True)
def gf2n_matvec_right(
    v: np.ndarray,  # shape (m,),   dtype uint16
    X: np.ndarray,  # shape (m, m), dtype uint16
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> np.ndarray:
    """
    Compute y = v @ X over GF(2^n), where v is a row-vector stored as shape (m,).

    Returns:
      y: shape (m,), dtype uint16
    """
    m = X.shape[0]
    y = np.empty(m, dtype=np.uint16)

    for j in range(m):
        acc = np.uint16(0)
        for i in range(m):
            acc ^= gf2n_mul(v[i], X[i, j], n, mask, red)
        y[j] = acc

    return y


@nb.njit(inline="always")
def gf2n_dot(
    x: np.ndarray,  # shape (m,), dtype uint16
    y: np.ndarray,  # shape (m,), dtype uint16
    n: int,
    mask: np.uint16,
    red: np.uint16,
):
    """
    Compute ⟨x, y⟩ = sum_i x[i] * y[i] over GF(2^n).

    Returns:
      scalar in GF(2^n) as uint16
    """
    m = x.shape[0]
    acc = np.uint16(0)
    for i in range(m):
        acc ^= gf2n_mul(x[i], y[i], n, mask, red)
    return acc


@nb.njit(cache=True)
def step(
    x: np.uint16, coeffs: np.ndarray, m: int, n: int, mask: np.uint16, red: np.uint16
) -> np.uint16:
    return gf2n_dot(coeffs, monomials_gf2n(x, m, n, mask, red), n, mask, red)
