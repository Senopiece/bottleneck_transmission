import math
import random
from typing import List, Set, Tuple

# Keep state-space bounded; sparse-chain variants are tailored for small packets.
MAX_PREFIX_INPUT_BITS = 16

__all__ = [
    "MAX_PREFIX_INPUT_BITS",
    "robust_soliton_cdf",
    "subset_from_x",
    "pack_prefix_symbols",
    "peel_add_equation",
    "peel_propagate",
]


def robust_soliton_cdf(k: int, c: float = 0.1, delta: float = 0.05) -> List[float]:
    if k <= 0:
        return [1.0]

    ideal = [0.0 for _ in range(k)]
    ideal[0] = 1.0 / k
    for d in range(2, k + 1):
        ideal[d - 1] = 1.0 / (d * (d - 1))

    R = c * math.log(k / delta) * math.sqrt(k)
    if R < 1.0:
        R = 1.0
    k_over_R = max(1, int(math.floor(k / R)))

    tau = [0.0 for _ in range(k)]
    for d in range(1, k + 1):
        if d < k_over_R:
            tau[d - 1] = R / (d * k)
        elif d == k_over_R:
            tau[d - 1] = R * math.log(R / delta) / k

    normalizer = sum(ideal[i] + tau[i] for i in range(k))
    probs = [(ideal[i] + tau[i]) / normalizer for i in range(k)]

    # Small-k tuning: force a stronger singleton ripple for tiny packets.
    if k <= 16:
        target_p1 = 0.35 if k <= 8 else 0.25
        p1 = probs[0]
        if p1 < target_p1:
            rest_sum = max(1e-12, 1.0 - p1)
            scale = (1.0 - target_p1) / rest_sum
            probs[0] = target_p1
            for i in range(1, k):
                probs[i] *= scale

    cdf: List[float] = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf


def subset_from_x(
    x: int,
    k: int,
    cdf: List[float],
    salt: int,
    singleton_limit: int,
) -> List[int]:
    if k <= 0:
        return []

    xi = int(x)
    limit = max(0, min(singleton_limit, k))
    if limit > 0 and xi < limit:
        return [xi]

    rng = random.Random((xi ^ salt) & 0xFFFFFFFFFFFFFFFF)
    draw = rng.random()

    degree = 1
    for i, threshold in enumerate(cdf):
        if draw <= threshold:
            degree = i + 1
            break

    degree = max(1, min(degree, k))
    if degree == k:
        return list(range(k))

    subset = rng.sample(range(k), degree)
    subset.sort()
    return subset


def pack_prefix_symbols(symbols: List[int], z: int) -> int:
    x = 0
    for symbol in symbols:
        x = (x << z) | int(symbol)
    return x


def peel_add_equation(
    x: int,
    y: int,
    k: int,
    cdf: List[float],
    salt: int,
    symbols: List[int | None],
    pending: List[Tuple[Set[int], int]],
) -> bool:
    if k <= 0:
        return False

    subset = subset_from_x(x, k, cdf, salt, singleton_limit=k)
    rhs = int(y)

    unknown: Set[int] = set()
    for idx in subset:
        known = symbols[idx]
        if known is None:
            unknown.add(idx)
        else:
            rhs ^= known

    if not unknown:
        return False

    changed = False
    if len(unknown) == 1:
        idx = next(iter(unknown))
        if symbols[idx] is None:
            symbols[idx] = rhs
            changed = True
        return changed

    pending.append((unknown, rhs))
    return changed


def peel_propagate(
    symbols: List[int | None], pending: List[Tuple[Set[int], int]]
) -> bool:
    changed_any = False

    progressed = True
    while progressed:
        progressed = False
        next_pending: List[Tuple[Set[int], int]] = []

        for unknown, rhs in pending:
            curr_unknown = set(unknown)
            curr_rhs = int(rhs)

            for idx in list(curr_unknown):
                known = symbols[idx]
                if known is not None:
                    curr_rhs ^= known
                    curr_unknown.remove(idx)

            if not curr_unknown:
                continue

            if len(curr_unknown) == 1:
                idx = next(iter(curr_unknown))
                if symbols[idx] is None:
                    symbols[idx] = curr_rhs
                    progressed = True
                    changed_any = True
                continue

            next_pending.append((curr_unknown, curr_rhs))

        pending[:] = next_pending

    return changed_any
