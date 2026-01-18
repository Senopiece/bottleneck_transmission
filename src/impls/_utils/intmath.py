import math


def min_m_such_that_2n_minus_1_pow_k_ge_2p(
    p: int,
    n: int,
    hi: int,
) -> int:
    """
    Finds the smallest m such that (2**n - 1)**m >= 2**p,
    with an explicit upper bound hi.

    Raises ValueError if no such m <= hi exists.

    Exact integer arithmetic; no floating-point logs.
    Time: O(log hi) big-int exponentiations (binary search).
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    if n <= 0:
        raise ValueError("n must be >= 1")
    if hi < 0:
        raise ValueError("hi must be >= 0")

    if p == 0:
        return 0

    B = (1 << n) - 1  # 2**n - 1
    if B <= 1:
        raise ValueError("n=1 makes log2(2**n - 1)=0; undefined for p>0")

    target = 1 << p  # 2**p

    def ok(m: int) -> bool:
        return pow(B, m) >= target

    # If even hi is insufficient, fail early
    if not ok(hi):
        raise ValueError(f"no solution: (2**{n} - 1)**m < 2**{p} for all m <= {hi}")

    # Binary search on [0, hi]
    lo, hi0 = 0, hi
    while lo < hi0:
        mid = (lo + hi0) // 2
        if ok(mid):
            hi0 = mid
        else:
            lo = mid + 1

    return lo


def floor_2n_log2_2n_minus_1(n: int) -> int:
    """
    Compute floor( 2^n * log2(2^n - 1) ) exactly, using integer-only fixed-point log2.

    Complexity: O(n) big-int multiplications on ~O(n)-bit integers.

    Valid for all integers n >= 1.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 1:
        raise ValueError("n must be >= 1")

    B = (1 << n) - 1  # 2^n - 1
    p = n  # number of fractional bits we need (Qp)

    # Integer part k = floor(log2(B))
    k = B.bit_length() - 1  # works for all B>=1

    # Represent y = B / 2^k in Qp fixed-point: y_fp = floor(y * 2^p) exactly here via shifting.
    # Since y = B * 2^{-k}, y*2^p = B * 2^{p-k}.
    y_fp = B << (p - k)  # exact because p-k >= 0 for this B and p=n

    two_fp = 2 << p  # represents 2.0 in Qp

    frac = 0
    # Extract p fractional bits of log2(y)
    for i in range(p):
        # Square in Qp: (y_fp/2^p)^2 * 2^p = y_fp^2 / 2^p
        y_fp = (y_fp * y_fp) >> p

        if y_fp >= two_fp:
            y_fp >>= 1
            frac |= 1 << (p - 1 - i)

    # Qp representation of log2(B) is (k << p) + frac, which equals floor(2^p * log2(B)).
    return (k << p) + frac


MERSENNE_PRIME_EXPS = {2, 3, 5, 7, 13}


def ispowprime_1_15(n: int) -> bool:
    return n in MERSENNE_PRIME_EXPS
