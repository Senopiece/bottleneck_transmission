from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from ._interface import Producer, Recoverer


def _payload_bits_for_d(d: int) -> int:
    if d <= 1:
        return 0
    return (d - 1).bit_length()


def _int_to_bits(value: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        shift = length - 1 - i
        bits[i] = (value >> shift) & 1
    return bits


def _bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def _vector_to_int(bits: np.ndarray) -> int:
    value = 0
    for i, bit in enumerate(bits):
        value |= (int(bit) & 1) << i
    return value


def _add_to_basis_rref_vector(v_bits: np.ndarray, basis: list[int]):
    v = _vector_to_int(v_bits)
    for r in basis:
        pivot = 1 << (r.bit_length() - 1)
        if v & pivot:
            v ^= r

    if v == 0:
        return False, basis

    pivot = 1 << (v.bit_length() - 1)
    new_basis = []
    for r in basis:
        if r & pivot:
            r ^= v
        new_basis.append(r)

    new_basis.append(v)
    new_basis.sort(reverse=True)
    return True, new_basis


def _invert_gf2(matrix: np.ndarray) -> np.ndarray | None:
    rows, cols = matrix.shape
    if rows != cols:
        return None
    if rows == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    aug = np.concatenate([matrix.copy(), np.eye(rows, dtype=np.uint8)], axis=1)
    for i in range(rows):
        pivot = None
        for r in range(i, rows):
            if aug[r, i]:
                pivot = r
                break
        if pivot is None:
            return None
        if pivot != i:
            aug[[i, pivot]] = aug[[pivot, i]]
        for r in range(rows):
            if r != i and aug[r, i]:
                aug[r] ^= aug[i]
    return aug[:, rows:]


def _max_payload_bits_for_n(packet_bits: int) -> int:
    if packet_bits <= 0:
        return 0
    best = 0
    for input_bits in range(0, packet_bits):
        output_bits = packet_bits - input_bits
        if output_bits <= 0:
            continue
        max_rank = 1 << input_bits
        capacity = max_rank * output_bits
        if capacity > best:
            best = capacity
    return best


@lru_cache(maxsize=None)
def _monomial_masks(input_bits: int, rank: int) -> tuple[int, ...]:
    if rank <= 1:
        return ()
    target = rank - 1
    masks: list[int] = []
    for degree in range(1, input_bits + 1):
        for combo in combinations(range(input_bits), degree):
            mask = 0
            for idx in combo:
                mask |= 1 << idx
            masks.append(mask)
            if len(masks) >= target:
                return tuple(masks)
    return tuple(masks)


def expand(x: np.ndarray, input_bits: int, rank: int) -> np.ndarray:
    x = np.array(x, dtype=np.uint8) & 1
    if x.size != input_bits:
        raise ValueError(f"Expected {input_bits} input bits, got {x.size}")
    if rank <= 0:
        return np.zeros(0, dtype=np.uint8)

    xmask = 0
    for i, bit in enumerate(x):
        xmask |= (int(bit) & 1) << i

    masks = _monomial_masks(input_bits, rank)
    out = np.zeros(rank, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        out[idx] = 1 if (xmask & mask) == mask else 0
    out[-1] = 1
    return out


def _payload_to_matrix(
    payload_bits: np.ndarray, output_bits: int, rank: int, zero_pad: int
) -> np.ndarray:
    capacity = output_bits * rank
    bits = np.array(payload_bits, dtype=np.uint8) & 1
    if bits.size > capacity:
        raise ValueError("payload exceeds available capacity")

    pad_len = capacity - bits.size
    if zero_pad != pad_len:
        raise ValueError("zero_pad mismatch")
    if pad_len:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])

    order = "F" if rank > output_bits else "C"
    return bits.reshape((output_bits, rank), order=order)


def _matrix_to_payload(
    matrix: np.ndarray, payload_bits: int, rank: int, output_bits: int
) -> np.ndarray:
    if payload_bits <= 0:
        return np.zeros(0, dtype=np.uint8)
    order = "F" if rank > output_bits else "C"
    flat = matrix.reshape(-1, order=order)
    return flat[:payload_bits].copy()


def _ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return x.bit_length() - (1 if (x & (x - 1)) == 0 else 0)


def _select_layout(payload_bits: int, packet_bits: int):
    """
    Selects a optimal layout given payload_bits and packet_bits.

    Choose (rank, input_bits, output_bits, zero_pad) with:
      - output_bits = packet_bits - input_bits > 0
      - rank * output_bits >= payload_bits
    Optimize lexicographically:
      1) rank (min)
      2) zero_pad (min)
      3) input_bits (min)

    Note: we would need to collect rank linearly inpependent observations to recover the original data. So we need it to be as small as possible.
    Also lowering the rank increases output_bits that increases partitioning so the reconstruction becomes more parellelizable.
    TODO: maybe sometimes it can be beneficial of having not the minimum possible rank, but achieving better results somehow (usually when rank includes constant term of monomials it starts performing poorly) (e.g. for 10,5 to use (5, 3, 2, 0) instead of (4, 2, 3, 0) - so theoretically its having lower rank, but the second option has better resilience from errors).
    """

    p = payload_bits
    n = packet_bits

    best = None  # (rank, zero_pad, input_bits, output_bits)

    # A safe upper bound: if input_bits=0 allowed, rank=p is enough (when n>0).
    # We also cap rank by 2^n *something; but p is typically the meaningful bound.
    for rank in range(1, max(2, p + 1)):
        min_in = _ceil_log2(rank)

        # input_bits can range up to n-1 (must leave at least 1 output bit)
        for input_bits in range(min_in, n):
            output_bits = n - input_bits
            capacity = rank * output_bits
            if capacity < p:
                continue

            zero_pad = capacity - p
            cand = (rank, zero_pad, input_bits, output_bits)

            if best is None or cand[:3] < best[:3]:
                best = cand

        # rank is primary objective; once we've found any solution at this rank,
        # we can stop (because larger ranks are worse).
        if best is not None and best[0] == rank:
            break

    if best is None:
        return None

    rank, zero_pad, input_bits, output_bits = best
    return rank, input_bits, output_bits, zero_pad


class SystematicProducer(Producer):
    def __init__(self, n: int, index: int, d: int):
        if index < 0 or index >= d:
            raise ValueError(f"index must be in [0, {d})")

        self.n = n
        self.d = d
        self.payload_bits = _payload_bits_for_d(d)
        max_bits = _max_payload_bits_for_n(n)
        if self.payload_bits > max_bits:
            raise ValueError(
                f"Cannot encode payload with {self.payload_bits} bits, max supported for n={n} is {max_bits}"
            )

        layout = _select_layout(self.payload_bits, n)
        if layout is None:
            raise ValueError(
                f"Cannot encode payload with {self.payload_bits} bits and packet size {n}"
            )
        self.rank, self.input_bits, self.output_bits, self.zero_pad = layout

        payload = _int_to_bits(index, self.payload_bits)
        self.A = _payload_to_matrix(
            payload, self.output_bits, self.rank, self.zero_pad
        ).astype(np.uint8)

        seed_material = (
            (index << 24)
            ^ (n << 16)
            ^ (self.rank << 8)
            ^ (self.input_bits << 4)
            ^ 0x51D5A7
        )
        self.rng = random.Random(seed_material)

    def generate(self) -> np.ndarray:
        if self.input_bits == 0:
            x = np.zeros(0, dtype=np.uint8)
        else:
            x = np.array(
                [self.rng.randrange(2) for _ in range(self.input_bits)],
                dtype=np.uint8,
            )

        fx = expand(x, self.input_bits, self.rank)
        y = (self.A @ fx) % 2
        if self.input_bits == 0:
            return y.astype(np.uint8)
        return np.concatenate([x, y]).astype(np.uint8)


class SystematicRecoverer(Recoverer):
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self.payload_bits = _payload_bits_for_d(d)
        max_bits = _max_payload_bits_for_n(n)
        if self.payload_bits > max_bits:
            raise ValueError(
                f"Cannot recover payload with {self.payload_bits} bits, max supported for n={n} is {max_bits}"
            )

        layout = _select_layout(self.payload_bits, n)
        if layout is None:
            raise ValueError(
                f"Cannot recover payload with {self.payload_bits} bits and packet size {n}"
            )
        self.rank, self.input_bits, self.output_bits, self.zero_pad = layout

        self.basis: list[int] = []
        self.x_cols: list[np.ndarray] = []
        self.y_cols: list[np.ndarray] = []
        self.recovered_index: int | None = 0 if self.payload_bits == 0 else None

    def feed(self, data: np.ndarray | None) -> int | None:
        if self.recovered_index is not None:
            return self.recovered_index
        if self.payload_bits == 0:
            self.recovered_index = 0
            return self.recovered_index
        if data is None:
            return None

        packet = np.array(data, dtype=np.uint8) & 1
        assert packet.shape == (self.n,), "packet shape mismatch"

        x = packet[: self.input_bits]
        y = packet[self.input_bits :]
        fx = expand(x, self.input_bits, self.rank)

        indep, new_basis = _add_to_basis_rref_vector(fx, self.basis)
        if not indep:
            return None
        self.basis = new_basis
        self.x_cols.append(fx)
        self.y_cols.append(y)

        if len(self.x_cols) < self.rank:
            return None

        X = np.column_stack(self.x_cols)
        Y = np.column_stack(self.y_cols)
        Xinv = _invert_gf2(X)
        if Xinv is None:
            return None

        A = (Y @ Xinv) % 2
        payload = _matrix_to_payload(A, self.payload_bits, self.rank, self.output_bits)
        self.recovered_index = _bits_to_int(payload)
        return self.recovered_index


def override_D(n: int) -> int:
    max_bits = _max_payload_bits_for_n(n)
    return 1 << max_bits if max_bits > 0 else 1


def producer_constructor(index: int, n: int, d: int) -> Producer:
    return SystematicProducer(n, index, d)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    return SystematicRecoverer(n, d)
