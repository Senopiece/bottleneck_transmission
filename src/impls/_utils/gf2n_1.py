import numpy as np
import numba as nb


@nb.njit(inline="always")
def gf2n_1_reduce(
    a: np.uint16,
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
) -> np.uint16:
    # exact reduction for uint16 domain, n<=15
    s = a
    r = np.uint16((s & mask) + (s >> n))
    # one more fold is harmless and still cheap
    r = np.uint16((r & mask) + (r >> n))
    if r >= mask:
        r -= mask
    return r


@nb.njit(inline="always", fastmath=True)
def gf2n_1_add(
    a: np.uint16,  # < mask
    b: np.uint16,  # < mask
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
) -> np.uint16:
    s = np.uint16(a + b)
    r = np.uint16((s & mask) + (s >> n))
    t = np.uint16(r - mask)
    return np.uint16(t + (np.uint16(r < mask) * mask))


@nb.njit(inline="always", fastmath=True)
def gf2n_1_mul(
    a: np.uint16,
    b: np.uint16,
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
) -> np.uint16:
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


@nb.njit(cache=True)
def monomials_gf2n_1(
    x: np.uint16,  # < mask
    m: int,  # number of monomials
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
):
    """
    m = number of monomials to compute.
    out[i] = x^i in Z/(2^n - 1)Z
    """
    acc = np.uint16(1)

    out = np.empty(m, dtype=np.uint16)

    for i in range(m):
        out[i] = acc
        acc = gf2n_1_mul(acc, x, n, mask)

    return out


@nb.njit(inline="always")
def gf2n_1_pow(
    base: np.uint16,  # < mask
    e: int,
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
) -> np.uint16:
    if e < 0:
        raise ValueError("negative exponent not supported for mod (2^n-1)")
    if e == 0:
        return np.uint16(1)

    res = np.uint16(1)

    ee = e
    while ee > 0:
        if ee & 1:
            res = gf2n_1_mul(res, base, n, mask)
        ee >>= 1
        if ee:
            base = gf2n_1_mul(base, base, n, mask)
    return res


@nb.njit(inline="always")
def gf2n_1_inv(
    a: np.uint16,  # < mask
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
) -> np.uint16:
    # In a field mod prime p, 0 has no inverse
    if a == 0:
        return np.uint16(0)

    # Fermat's little theorem: a^(p-2) mod p
    # exponent fits in int for n<=15 (p<=32767)
    return gf2n_1_pow(a, int(mask) - 2, n, mask)


@nb.njit(cache=True)
def gf2n_1_solve_and_invert(
    X: np.ndarray,  # (m, m) uint16, each entry < mask
    b: np.ndarray,  # (m,)   uint16, row-vector in aX=b, each entry < mask
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
):
    """
    Solve aX=b and compute X^{-1} over GF(2^n - 1), with 'a' obtained during elimination.

    Returns:
      a        : (m,)   uint16, each entry < mask
      Xinv     : (m,m)  uint16, each entry < mask
      singular : bool
    """
    m = X.shape[0]

    # Work on transpose so we can do standard "A x = rhs" elimination
    A = X.T.copy()

    # Augment with identity to get inverse of A; later Xinv = (A^{-1})^T
    Aug = np.empty((m, 2 * m), dtype=np.uint16)
    for i in range(m):
        for j in range(m):
            Aug[i, j] = A[i, j]
        for j in range(m):
            Aug[i, m + j] = np.uint16(1) if i == j else np.uint16(0)

    rhs = b.copy()

    # Gauss-Jordan elimination on [Aug | rhs]
    for col in range(m):
        # Find pivot row >= col with nonzero Aug[row, col]
        piv = -1
        for r in range(col, m):
            if Aug[r, col] != 0:
                piv = r
                break
        if piv == -1:
            # Singular
            return np.empty(m, dtype=np.uint16), np.empty((m, m), dtype=np.uint16), True

        # Swap pivot row into position
        if piv != col:
            for j in range(2 * m):
                tmp = Aug[col, j]
                Aug[col, j] = Aug[piv, j]
                Aug[piv, j] = tmp
            tmp = rhs[col]
            rhs[col] = rhs[piv]
            rhs[piv] = tmp

        # Scale pivot row to make pivot = 1
        pivot_val = Aug[col, col]  # nonzero
        inv_piv = gf2n_1_inv(pivot_val, n, mask)  # valid since field and pivot_val!=0

        for j in range(2 * m):
            Aug[col, j] = gf2n_1_mul(Aug[col, j], inv_piv, n, mask)
        rhs[col] = gf2n_1_mul(rhs[col], inv_piv, n, mask)

        # Eliminate all other rows in this column
        for r in range(m):
            if r == col:
                continue
            factor = Aug[r, col]
            if factor == 0:
                continue
            # row_r = row_r - factor * row_col, but in mod p subtraction is + (p - ...)
            # implement as row_r + (p-factor)*row_col
            neg_factor = np.uint16(0) if factor == 0 else np.uint16(mask - factor)
            for j in range(2 * m):
                term = gf2n_1_mul(neg_factor, Aug[col, j], n, mask)
                Aug[r, j] = gf2n_1_add(Aug[r, j], term, n, mask)
            term = gf2n_1_mul(neg_factor, rhs[col], n, mask)
            rhs[r] = gf2n_1_add(rhs[r], term, n, mask)

    # Now Aug[:, :m] should be I, Aug[:, m:] is A^{-1}
    Ainv = np.empty((m, m), dtype=np.uint16)
    for i in range(m):
        for j in range(m):
            Ainv[i, j] = Aug[i, m + j]

    # Xinv = (Ainv)^T because A = X^T
    Xinv = np.empty((m, m), dtype=np.uint16)
    for i in range(m):
        for j in range(m):
            Xinv[i, j] = Ainv[j, i]

    # Solution: a^T = A^{-1} b^T  => a = b @ X^{-1}
    # We already computed rhs = a^T
    a = rhs  # (m,) row-vector entries

    return a, Xinv, False


@nb.njit(inline="always")
def gf2n_1_dot(
    x: np.ndarray,  # shape (m,), dtype uint16, each entry < mask
    y: np.ndarray,  # shape (m,), dtype uint16, each entry < mask
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
):
    """
    Compute xTy = sum_i x[i] * y[i] over GF(2^n - 1).

    Returns:
      scalar in GF(2^n - 1) as uint16, < mask
    """
    m = x.shape[0]
    acc = np.uint16(0)
    for i in range(m):
        term = gf2n_1_mul(x[i], y[i], n, mask)
        acc = gf2n_1_add(acc, term, n, mask)
    return acc


@nb.njit(cache=True)
def step(
    x: np.uint16,  # < mask
    coeffs: np.ndarray,  # shape (m,), dtype uint16, each entry < mask
    m: int,  # number of monomials
    n: int,  # 1 <= n <= 15
    mask: np.uint16,  # must be 2^n - 1 (extracted as arg for speed) - must be a prime
) -> np.uint16:
    return gf2n_1_dot(coeffs, monomials_gf2n_1(x, m, n, mask), n, mask)
