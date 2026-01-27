from typing import Tuple
import numpy as np
import numba as nb

# ==========================================================================
# Define Field
# ==========================================================================

# ---------------------------
# Canonical polynomial book: primitive trinomials for GF(2^n), n in [1, 15]
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
}


def _primitive_poly_int(n: int) -> int:
    """
    Returns p as an integer bitmask, where bit i is coeff of x^i.
    Monic term x^n is always set.
    """
    if not (1 <= n <= 15):
        raise ValueError("This canonical polynomial book supports n in [1,15].")
    a = PRIMITIVE_TRINOMIAL_A[n]
    if n == 1:
        return (1 << 1) | 1  # x + 1
    return (1 << n) | (1 << a) | 1


def make_field(N: int):
    n = N
    poly = np.uint16(_primitive_poly_int(N))  # includes x^n term
    mask = np.uint16((1 << N) - 1)  # keep n bits
    red = np.uint16(poly & mask)  # poly without x^n term (low n bits)

    return n, mask, red


@nb.njit(inline="always")
def add(
    a: np.uint16,
    b: np.uint16,
    mask: np.uint16,
):
    return (a ^ b) & mask


@nb.njit(inline="always")
def neg(a: np.uint16):
    return a


@nb.njit(inline="always")
def sub(
    a: np.uint16,
    b: np.uint16,
    mask: np.uint16,
):
    return (a ^ b) & mask


@nb.njit(inline="always")
def mul(
    a: np.uint16,
    b: np.uint16,
    n: int,
    mask: np.uint16,
    red: np.uint16,
):
    res = np.uint16(0)

    # a = a & mask
    # b = b & mask

    for _ in range(n):
        if b & 1:
            res ^= a
        b >>= 1  # type: ignore

        carry = (a >> (n - 1)) & 1
        a = (a << 1) & mask  # type: ignore
        if carry:
            a ^= red

    return res & mask


@nb.njit(inline="always")
def pow(base: np.uint16, e: int, n: int, mask: np.uint16, red: np.uint16) -> np.uint16:
    res = np.uint16(1)
    while e > 0:
        if e & 1:
            res = mul(res, base, n, mask, red)
        e >>= 1
        if e:
            base = mul(base, base, n, mask, red)
    return res


@nb.njit(inline="always")
def inv(a: np.uint16, n: int, mask: np.uint16, red: np.uint16) -> np.uint16:
    # In GF(2^n), a^{-1} = a^(2^n - 2) for a != 0
    if a == 0:
        # Numba can't raise Python exceptions cleanly in all contexts; return 0 and let caller detect.
        return np.uint16(0)
    return pow(a, int(mask) - 1, n, mask, red)


# ==========================================================================
# Polynomial Utils on this field
# ==========================================================================


@nb.njit(inline="always")
def newton_to_falling_factorial(
    coeffs: np.ndarray,  # (m,) uint16: Newton coeffs on `basis`
    basis: np.ndarray,  # (m,) uint16: nodes x0..x_{m-1} (distinct)
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> np.ndarray:
    """
    Convert Newton form on `basis`:
        f(t) = c0 + c1*(t-x0) + c2*(t-x0)(t-x1) + ... + c_{m-1}*Π_{j=0}^{m-2}(t-xj)
    into falling-factorial form anchored at 0:
        f(t) = a0 + a1*t + a2*t(t-1) + a3*t(t-1)(t-2) + ...
             + a_{m-1}*Π_{r=0}^{m-2}(t-r)

    I.e. basis polynomials:
        q0(t)=1
        qk(t)=∏_{r=0}^{k-1}(t-r),  k>=1

    Uses Horner in Newton form, maintaining coefficients in q_k basis.
    No inversions. Time O(m^2). Space O(m).
    """
    m = coeffs.shape[0]
    if basis.shape[0] != m:
        raise ValueError("coeffs and basis must have same length")

    a = np.zeros(m, dtype=np.uint16)  # current polynomial in q_k basis
    nxt = np.zeros(m, dtype=np.uint16)

    if m == 0:
        return a

    # Start with P(t) = c_{m-1}
    deg = 0
    a[0] = np.uint16(coeffs[m - 1])

    # Horner: P <- c_k + (t - x_k) * P
    #
    # Need a "multiply-by-(t-alpha)" operator in q-basis:
    #   q_{i+1}(t) = q_i(t) * (t - i)
    # Identity:
    #   (t - alpha) q_i(t) = q_{i+1}(t) + (i - alpha) q_i(t)
    #
    for k in range(m - 2, -1, -1):
        alpha = basis[k]

        # Clear only the needed prefix of nxt (0..deg+1)
        for i in range(deg + 2):
            nxt[i] = np.uint16(0)

        # Coefficient of q_{deg+1} is a[deg]
        nxt[deg + 1] = a[deg]

        # nxt[0] = a[0] * (0 - alpha) = a[0] * (-alpha)
        minus_alpha = neg(alpha)
        nxt[0] = mul(a[0], minus_alpha, n, mask, red)

        # for i = 1..deg:
        # nxt[i] = a[i-1] + a[i] * (i - alpha)
        for i in range(1, deg + 1):
            i_minus_alpha = sub(np.uint16(i), alpha, mask)
            term = mul(a[i], i_minus_alpha, n, mask, red)
            nxt[i] = add(a[i - 1], term, mask)

        # swap a <- nxt, degree increases by 1
        for i in range(deg + 2):
            a[i] = nxt[i]
        deg += 1

        # add constant term c_k
        a[0] = add(a[0], coeffs[k], mask)

    return a


@nb.njit(inline="always")
def interpolate_newton_poly(
    y: np.ndarray,  # (m,) uint16, values f(x[i])
    x: np.ndarray,  # (m,) uint16, distinct x[i]
    n: int,
    mask: np.uint16,
    red: np.uint16,
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
            num = sub(num, den, mask)

            # x[i] - x[i-k]
            xi = x[i]
            xj = x[i - k]
            dx = sub(xi, xj, mask)
            if dx == 0:
                raise ValueError(
                    "duplicate x nodes (division by zero in interpolation)"
                )

            inv_dx = inv(dx, n, mask, red)
            dd[i] = mul(num, inv_dx, n, mask, red)

    # Now dd[i] == a_i, and Newton form is:
    # f(t) = a0 + a1*(t-x0) + a2*(t-x0)(t-x1) + ... + a_{m-1}*Π_{j=0}^{m-2}(t-xj)

    return dd


@nb.njit(cache=True)
def interpolate_poly_falling_factorial(
    y: np.ndarray,  # (m,) uint16, values f(x[i])
    x: np.ndarray,  # (m,) uint16, distinct x[i]
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> np.ndarray:
    dd = interpolate_newton_poly(y, x, n, mask, red)
    return newton_to_falling_factorial(dd, x, n, mask, red)


@nb.njit(cache=True)
def first_points_to_falling_factorial_coeffs(
    y: np.ndarray,  # (m,) uint16, y[i] = f(i) for i=0..m-1
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> np.ndarray:
    """
    Recover falling-factorial coefficients a[0..m-1] in GF(2^n) for nodes t=i (uint16(i)):

        f(t) = sum_{k=0}^{m-1} a[k] * q_k(t),
        q_0(t)=1, q_k(t)=prod_{r=0}^{k-1} (t - r).

    Uses triangular structure q_k(i)=0 for k>i.
    Cost: ~O(m^2/2) mul/add, and ONLY 1 inversion total.
    """
    m = y.shape[0]
    a = np.empty(m, dtype=np.uint16)
    if m == 0:
        return a

    # --------------------------
    # Pass 1: compute diagonals D[i] = q_i(i) = ∏_{r=0}^{i-1} (i - r)
    # --------------------------
    D = np.empty(m, dtype=np.uint16)
    D[0] = np.uint16(1)

    for i in range(1, m):
        t = np.uint16(i)
        prod = np.uint16(1)  # q_0(i)
        # build up to q_i(i)
        for k in range(1, i + 1):
            # factor = (i - (k-1)) in the field; in char 2, '-'=='+' so it's XOR
            factor = sub(t, np.uint16(k - 1), mask)  # (i XOR (k-1)) & mask
            prod = mul(prod, factor, n, mask, red)
        D[i] = prod

    # --------------------------
    # Invert all diagonals with ONE inversion:
    # prefix products P[i] = D[0]*...*D[i]
    # then invD[i] recovered by backward sweep.
    # --------------------------
    P = np.empty(m, dtype=np.uint16)
    P[0] = D[0]
    for i in range(1, m):
        P[i] = mul(P[i - 1], D[i], n, mask, red)

    invP = inv(P[m - 1], n, mask, red)  # the only inversion
    invD = np.empty(m, dtype=np.uint16)

    for i in range(m - 1, -1, -1):
        prevP = np.uint16(1) if i == 0 else P[i - 1]
        invD[i] = mul(invP, prevP, n, mask, red)  # inv(D[i]) = inv(P[i]) * P[i-1]
        invP = mul(invP, D[i], n, mask, red)  # now invP == inv(P[i-1])

    # --------------------------
    # Pass 2: forward substitution to recover a
    # y[i] = Σ_{k=0}^{i} a[k] q_k(i), with q_k(i) built incrementally.
    # --------------------------
    a[0] = np.uint16(y[0])

    for i in range(1, m):
        t = np.uint16(i)

        # rhs = y[i] - Σ_{k=0}^{i-1} a[k] q_k(i)
        # but '-' == '+' so rhs = y[i] + Σ ...
        rhs = np.uint16(y[i])

        # q_0(i) = 1 term
        rhs = sub(rhs, a[0], mask)  # rhs ^= a0*1

        prod = np.uint16(1)  # q_0(i)
        # accumulate k=1..i-1 using q_k(i)
        for k in range(1, i):
            factor = sub(t, np.uint16(k - 1), mask)  # (i - (k-1))
            prod = mul(prod, factor, n, mask, red)  # prod = q_k(i)
            rhs = sub(rhs, mul(a[k], prod, n, mask, red), mask)

        # now prod == q_{i-1}(i); extend once to get q_i(i)=D[i]
        factor = sub(t, np.uint16(i - 1), mask)
        prod = mul(prod, factor, n, mask, red)  # prod == D[i]

        # a[i] = rhs / D[i]
        a[i] = mul(rhs, invD[i], n, mask, red)

    return a


@nb.njit(cache=True)
def evaluate_poly_falling_factorial(
    t: np.uint16,
    coeffs: np.ndarray,  # shape (m,), dtype uint16  (a0..a_{m-1})
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> np.uint16:
    """
    Evaluate polynomial in falling-factorial basis anchored at 0:

        f(t) = a0
             + a1*t
             + a2*t(t-1)
             + ...
             + a_{m-1} * Π_{j=0}^{m-2} (t-j)

    Horner-like scheme:
        acc = a_{m-1}
        for k = m-2 .. 0:
            acc = a_k + (t - k) * acc

    Complexity: O(m) field mul/add, no inversions.
    """
    m = coeffs.shape[0]
    if m == 0:
        return np.uint16(0)

    acc = np.uint16(coeffs[m - 1])
    for k in range(m - 2, -1, -1):
        tk = sub(t, np.uint16(k), mask)  # (t - k)
        acc = add(mul(acc, tk, n, mask, red), coeffs[k], mask)

    return acc


@nb.njit(cache=True)
def evaluate_poly_falling_factorial_first_points(
    coeffs: np.ndarray,  # (m,) uint16 : a0..a_{m-1}
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> np.ndarray:
    """
    Compute y[t] = f(t) for t=0..m-1 where f(t)=sum_{k=0}^{m-1} a[k]*(t)_k
    and (t)_k = t(t-1)...(t-k+1) in the field.

    essentially equivalent to calling
    for t in range(m):
        evaluate_poly_falling_factorial(t, coeffs)

    but faster coz of utilizing the structure

    Exploits (t)_k == 0 for k>t (because one factor becomes (t - t)=0).
    Total work ~ sum_{t=0}^{m-1} O(t) = O(m^2/2).
    """
    m = coeffs.shape[0]
    y = np.empty(m, dtype=np.uint16)
    if m == 0:
        return y

    y[0] = np.uint16(coeffs[0])

    for ti in range(1, m):
        t = np.uint16(ti)

        acc = np.uint16(coeffs[0])

        # prod = (t)_k as we increase k
        prod = np.uint16(1)  # (t)_0
        # For k=1..t: prod *= (t-(k-1)); acc += a[k]*prod
        for k in range(1, ti + 1):
            factor = sub(t, np.uint16(k - 1), mask)  # (t-(k-1))
            prod = mul(prod, factor, n, mask, red)
            acc = add(acc, mul(coeffs[k], prod, n, mask, red), mask)

        y[ti] = acc

    return y


@nb.njit(cache=True)
def newton_to_poly(
    coeffs: np.ndarray,  # (m,) uint16: Newton coeffs on `basis`
    basis: np.ndarray,  # (m,) uint16: nodes x0..x_{m-1} (distinct)
    n: int,
    mask: np.uint16,
    red: np.uint16,
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
        minus_alpha = neg(alpha)

        # Clear tmp[0..deg+1]
        for i in range(deg + 2):
            tmp[i] = np.uint16(0)

        # Multiply by (t - alpha):
        # new[0]     = (-alpha) * p0
        # new[j]     = p_{j-1} + (-alpha) * p_j   for 1<=j<=deg
        # new[deg+1] = p_deg
        tmp[0] = mul(out[0], minus_alpha, n, mask, red)
        for j in range(1, deg + 1):
            term = mul(out[j], minus_alpha, n, mask, red)
            tmp[j] = add(out[j - 1], term, mask)
        tmp[deg + 1] = out[deg]

        # Add constant coeffs[k]
        tmp[0] = add(tmp[0], coeffs[k], mask)

        # Swap tmp -> out (only needed prefix)
        for i in range(deg + 2):
            out[i] = tmp[i]
        deg += 1

    return out


@nb.njit(cache=True)
def interpolate_poly(
    y: np.ndarray,  # (m,) uint16, values f(x[i])
    x: np.ndarray,  # (m,) uint16, distinct x[i]
    n: int,
    mask: np.uint16,
    red: np.uint16,
) -> np.ndarray:
    dd = interpolate_newton_poly(y, x, n, mask, red)
    return newton_to_poly(dd, x, n, mask, red)


@nb.njit(cache=True)
def evaluate_poly(
    t: np.uint16,
    coeffs: np.ndarray,  # shape (m,), dtype uint16  (c0..c_{m-1})
    n: int,
    mask: np.uint16,
    red: np.uint16,
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
        acc = add(mul(acc, t, n, mask, red), coeffs[k], mask)

    return acc
