from typing import List, Tuple
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


import numba as nb
import numpy as np


@nb.njit(cache=True)
def solve_linear_system_gf2n(
    A: np.ndarray,  # (rows, cols) uint16
    b: np.ndarray,  # (rows,) uint16
    n: int,
    mask: np.uint16,
    red: np.uint16,
):
    """
    Solve A x = b over GF(2^n) using Gauss-Jordan elimination.

    Returns:
      x  : (cols,) uint16 (contents undefined if ok==False)
      ok : bool    (False if singular/inconsistent)
    """
    Ac = A.copy()
    bc = b.copy()

    rows, cols = Ac.shape
    x = np.empty(cols, dtype=np.uint16)

    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find pivot
        pivot = -1
        for r in range(pivot_row, rows):
            if Ac[r, col] != np.uint16(0):
                pivot = r
                break
        if pivot == -1:
            # no pivot in this column => singular for square/overdetermined solve
            return x, False

        # Swap rows pivot <-> pivot_row
        if pivot != pivot_row:
            for j in range(cols):
                tmp = Ac[pivot_row, j]
                Ac[pivot_row, j] = Ac[pivot, j]
                Ac[pivot, j] = tmp
            tmpb = bc[pivot_row]
            bc[pivot_row] = bc[pivot]
            bc[pivot] = tmpb

        # Normalize pivot row
        piv = Ac[pivot_row, col]
        inv_p = gf2n_inv(piv, n, mask, red)
        if inv_p == np.uint16(0):
            return x, False

        for j in range(col, cols):
            Ac[pivot_row, j] = gf2n_mul(Ac[pivot_row, j], inv_p, n, mask, red)
        bc[pivot_row] = gf2n_mul(bc[pivot_row], inv_p, n, mask, red)

        # Eliminate column from all other rows
        for r in range(rows):
            if r == pivot_row:
                continue
            factor = Ac[r, col]
            if factor == np.uint16(0):
                continue
            for j in range(col, cols):
                Ac[r, j] = Ac[r, j] ^ gf2n_mul(factor, Ac[pivot_row, j], n, mask, red)
            bc[r] = bc[r] ^ gf2n_mul(factor, bc[pivot_row], n, mask, red)

        pivot_row += 1

    # Check consistency for extra rows (if rows > cols):
    # If a row is all zeros in A but b != 0 => inconsistent
    for r in range(cols, rows):
        all_zero = True
        for j in range(cols):
            if Ac[r, j] != np.uint16(0):
                all_zero = False
                break
        if all_zero and bc[r] != np.uint16(0):
            return x, False

    # If we got here, the system is solved (for square or full-rank tall systems)
    # Solution sits in first 'cols' entries of bc after Gauss-Jordan for a full pivot sequence.
    for j in range(cols):
        x[j] = bc[j]
    return x, True


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


@nb.njit(inline="always")
def poly_div_gf2n(
    numerator: np.ndarray,
    denominator: np.ndarray,
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polynomial long division over GF(2^n), coeffs in ascending order.
    Returns (quotient, remainder).
    """
    num = numerator.copy()
    den = denominator
    deg_num = num.shape[0] - 1
    deg_den = den.shape[0] - 1

    if deg_num < deg_den:
        return (
            np.zeros(1, dtype=np.uint16),
            num,
        )

    quotient = np.zeros(deg_num - deg_den + 1, dtype=np.uint16)
    lead = den[deg_den]
    inv_lead = gf2n_inv(lead, n, mask, red) if lead != 1 else np.uint16(1)

    for i in range(deg_num, deg_den - 1, -1):
        coef = num[i]
        if coef == 0:
            continue
        if lead != 1:
            coef = gf2n_mul(coef, inv_lead, n, mask, red)
        q_index = i - deg_den
        quotient[q_index] = coef
        for j in range(deg_den + 1):
            num[q_index + j] ^= gf2n_mul(coef, den[j], n, mask, red)

    remainder = num[:deg_den]
    return quotient, remainder


@nb.njit(cache=True)
def berlekamp_welch(
    xs: np.ndarray,  # (k,) uint16
    ys: np.ndarray,  # (k,) uint16
    m: int,
    n: int,
    mask: np.uint16,
    red: np.uint16,
):
    """
    Numba-friendly Berlekamp-Welch.

    Returns:
      P  : (m,) uint16 (candidate poly coeffs)
      ok : bool
    """
    k = xs.shape[0]
    P_out = np.zeros(m, dtype=np.uint16)

    if k < m:
        return P_out, False

    max_t = (k - m) // 2
    if max_t < 0:
        max_t = 0

    for t in range(max_t, -1, -1):
        unknowns = m + 2 * t
        if k < unknowns:
            continue

        A = np.zeros((k, unknowns), dtype=np.uint16)
        bb = np.zeros(k, dtype=np.uint16)

        for i in range(k):
            x = xs[i]
            y = ys[i]
            x_pows = monomials_gf2n(x, m + t, n, mask, red)  # length m+t

            # N part
            for j in range(m + t):
                A[i, j] = x_pows[j]

            # E part (excluding leading 1)
            if t != 0:
                for j in range(t):
                    A[i, m + t + j] = gf2n_mul(y, x_pows[j], n, mask, red)

            # RHS
            if t != 0:
                bb[i] = gf2n_mul(y, x_pows[t], n, mask, red)
            else:
                bb[i] = y

        sol, ok = solve_linear_system_gf2n(A, bb, n, mask, red)
        if not ok:
            continue

        # N = sol[:m+t]
        # E = [sol[m+t : m+2t], 1]
        if t != 0:
            E = np.empty(t + 1, dtype=np.uint16)
            for j in range(t):
                E[j] = sol[m + t + j]
            E[t] = np.uint16(1)

            Npoly = np.empty(m + t, dtype=np.uint16)
            for j in range(m + t):
                Npoly[j] = sol[j]

            P, rem = poly_div_gf2n(Npoly, E, n, mask, red)

            # require exact division
            for j in range(rem.shape[0]):
                if rem[j] != np.uint16(0):
                    ok = False
                    break
            if not ok:
                continue
        else:
            P = np.empty(m, dtype=np.uint16)
            # sol has length m in this case
            for j in range(m):
                P[j] = sol[j]

        # Enforce length exactly m (manual pad/truncate)
        Pm = np.zeros(m, dtype=np.uint16)
        L = P.shape[0]
        if L >= m:
            for j in range(m):
                Pm[j] = P[j]
        else:
            for j in range(L):
                Pm[j] = P[j]

        mismatches = 0
        for i in range(k):
            if step(xs[i], Pm, m, n, mask, red) != ys[i]:
                mismatches += 1
                if mismatches > t:
                    break

        if mismatches <= t:
            return Pm, True

    # Fallback: interpolate from first m samples
    X = np.empty((m, m), dtype=np.uint16)
    Y = np.empty(m, dtype=np.uint16)
    for i in range(m):
        X[i, :] = monomials_gf2n(xs[i], m, n, mask, red)
        Y[i] = ys[i]

    coeffs, _, singular = gf2n_solve_and_invert(X, Y, n, mask, red)
    if singular:
        return P_out, False
    return coeffs, True
