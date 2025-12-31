from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from ._interface import Estimator, Packet, Payload, Protocol, Config, Sampler


# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1

# Monomials + shuffle featurizer


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


def _int_to_bits_le(value: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)
    nbytes = (length + 7) // 8
    raw = value.to_bytes(nbytes, "little", signed=False)
    packed = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")
    return bits[:length]


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


def _add_to_basis_rref_int(v: int, basis: list[int]):
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


def _normalize_payload(payload: Payload, payload_bits: int) -> np.ndarray:
    bits = np.array(payload, dtype=np.uint8).reshape(-1) & 1
    if bits.size != payload_bits:
        raise ValueError(
            f"Expected payload of {payload_bits} bits, got {bits.size} bits"
        )
    return bits


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


@lru_cache(maxsize=None)
def _full_monomial_masks(input_bits: int) -> np.ndarray:
    if input_bits < 0:
        raise ValueError("input_bits must be >= 0")
    return np.arange(1 << input_bits, dtype=np.uint64)


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


def expand_full(x: np.ndarray, input_bits: int) -> np.ndarray:
    x = np.array(x, dtype=np.uint8) & 1
    if x.size != input_bits:
        raise ValueError(f"Expected {input_bits} input bits, got {x.size}")

    xmask = 0
    for i, bit in enumerate(x):
        xmask |= (int(bit) & 1) << i

    masks = _full_monomial_masks(input_bits)
    return ((masks & xmask) == masks).astype(np.uint8)


def _g_seed(rank: int, input_bits: int) -> int:
    return (rank << 32) ^ (input_bits << 16) ^ 0x9E3779B97F4A7C15


@lru_cache(maxsize=None)
def _deterministic_G(rank: int, input_bits: int) -> np.ndarray:
    if rank < 0:
        raise ValueError("rank must be >= 0")
    m = 1 << input_bits
    if rank == 0:
        return np.zeros((0, m), dtype=np.uint8)
    if m < rank:
        raise ValueError("monomial count must be >= rank")

    rng = random.Random(_g_seed(rank, input_bits))
    rows: list[np.ndarray] = []
    basis: list[int] = []
    relax_after = max(64, rank * 64)
    tries = 0
    min_weight = m // 4 if m >= 16 else 1
    max_weight = m - min_weight if m >= 16 else m

    while len(rows) < rank:
        row_int = rng.getrandbits(m)
        if row_int == 0:
            tries += 1
            continue

        if m >= 16 and tries < relax_after:
            weight = row_int.bit_count()
            if weight < min_weight or weight > max_weight:
                tries += 1
                continue

        indep, basis = _add_to_basis_rref_int(row_int, basis)
        if not indep:
            tries += 1
            continue

        rows.append(_int_to_bits_le(row_int, m))
        tries += 1

    return np.stack(rows, axis=0).astype(np.uint8, copy=False)


def _project_monomials(x: np.ndarray, input_bits: int, G: np.ndarray) -> np.ndarray:
    monomials = expand_full(x, input_bits)
    return (G @ monomials) % 2


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
    TODO: maybe apply ml in search of optimal layouting
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


class _SystematicSamplerState:
    def __init__(
        self,
        packet_bits: int,
        payload_bits: int,
        payload: Payload,
        layout: tuple[int, int, int, int],
    ):
        self.n = packet_bits
        self.payload_bits = payload_bits
        self.rank, self.input_bits, self.output_bits, self.zero_pad = layout
        self.G = _deterministic_G(self.rank, self.input_bits)

        payload_bits_arr = _normalize_payload(payload, payload_bits)
        self.A = _payload_to_matrix(
            payload_bits_arr, self.output_bits, self.rank, self.zero_pad
        ).astype(np.uint8)

        payload_seed = _bits_to_int(payload_bits_arr)
        seed_material = (
            (payload_seed << 24)
            ^ (packet_bits << 16)
            ^ (self.rank << 8)
            ^ (self.input_bits << 4)
            ^ 0x51D5A7
        )
        self.rng = random.Random(seed_material)

    def generate(self) -> Packet:
        if self.input_bits == 0:
            x = np.zeros(0, dtype=np.uint8)
        else:
            x = np.array(
                [self.rng.randrange(2) for _ in range(self.input_bits)],
                dtype=np.uint8,
            )

        fx = _project_monomials(x, self.input_bits, self.G)
        y = (self.A @ fx) % 2
        if self.input_bits == 0:
            packet = y
        else:
            packet = np.concatenate([x, y])
        return packet.astype(np.bool_, copy=False)


class _SystematicEstimatorState:
    def __init__(
        self,
        packet_bits: int,
        payload_bits: int,
        layout: tuple[int, int, int, int],
    ):
        self.n = packet_bits
        self.payload_bits = payload_bits
        self.rank, self.input_bits, self.output_bits, self.zero_pad = layout
        self.G = _deterministic_G(self.rank, self.input_bits)

        self.basis: list[int] = []
        self.x_cols: list[np.ndarray] = []
        self.y_cols: list[np.ndarray] = []
        self.recovered_payload: np.ndarray | None = (
            np.zeros(0, dtype=np.bool_) if self.payload_bits == 0 else None
        )

    def progress(self) -> float:
        if self.recovered_payload is not None:
            return 1.0
        if self.rank <= 0:
            return 0.0
        return min(1.0, len(self.x_cols) / self.rank)

    def feed(self, data: Packet | None) -> Payload | None:
        if self.recovered_payload is not None:
            return self.recovered_payload
        if self.payload_bits == 0:
            self.recovered_payload = np.zeros(0, dtype=np.bool_)
            return self.recovered_payload
        if data is None:
            return None

        packet = np.array(data, dtype=np.uint8).reshape(-1) & 1
        if packet.shape != (self.n,):
            raise ValueError("packet shape mismatch")

        x = packet[: self.input_bits]
        y = packet[self.input_bits :]
        fx = _project_monomials(x, self.input_bits, self.G)

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
        self.recovered_payload = payload.astype(np.bool_, copy=False)
        return self.recovered_payload


def create_protocol(config: Config) -> Protocol:
    packet_bits = int(config.packet_bitsize)
    payload_bits = int(config.payload_bitsize)
    if payload_bits < 0:
        raise ValueError("payload_bitsize must be >= 0")

    max_bits = max_payload_bitsize(packet_bits)
    if payload_bits > max_bits:
        raise ValueError(
            f"Cannot encode payload with {payload_bits} bits, max supported for "
            f"packet size {packet_bits} is {max_bits}"
        )

    layout = _select_layout(payload_bits, packet_bits)
    if layout is None:
        raise ValueError(
            f"Cannot encode payload with {payload_bits} bits and packet size {packet_bits}"
        )

    def make_sampler(payload: Payload) -> Sampler:
        sampler_state = _SystematicSamplerState(
            packet_bits, payload_bits, payload, layout
        )
        while True:
            yield sampler_state.generate()

    def make_estimator() -> Estimator:
        estimator_state = _SystematicEstimatorState(packet_bits, payload_bits, layout)
        packet = yield estimator_state.progress()
        while True:
            recovered = estimator_state.feed(packet)
            if recovered is not None:
                return recovered
            packet = yield estimator_state.progress()

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_payload_bitsize(packet_bitsize: int) -> int:
    """Return the maximum payload size (in bits) for packets of the given size."""
    return 2 ** (packet_bitsize - 1)
