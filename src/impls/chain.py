import random

import numpy as np

from ._utils.intmath import floor_2n_log2_2n_minus_1
from ._utils.conversions import bits_to_int, int_to_bits
from ._interface import Estimator, Packet, Message, Protocol, Config, Sampler


# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1

# NOTE: Unlike traditional fountain codes this protocol requires deletion observation
#        so its rateless but not fountain

# TODO: remove redundancies and optimize (numba)
# TODO: extract reusable funcs
# TODO: refactor to use pure generators


def _normalize_payload(payload: Message, message_bitsize: int) -> np.ndarray:
    bits = np.array(payload, dtype=np.uint8).reshape(-1) & 1
    if bits.size != message_bitsize:
        raise ValueError(
            f"Expected payload of {message_bitsize} bits, got {bits.size} bits"
        )
    return bits


def _add_to_basis_rref_vector(v_bits: np.ndarray, basis: list[int]):
    v = bits_to_int(v_bits)
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


def _invert_mod2(matrix: np.ndarray) -> np.ndarray | None:
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


_MASK64 = (1 << 64) - 1
_MIX64_CONST = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    v = (value + _MIX64_CONST) & _MASK64
    z = v
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    return z ^ (z >> 31)


def _hash_feature_vector(x_int: int, seed: int, m: int) -> np.ndarray:
    if m <= 0:
        return np.zeros(0, dtype=np.uint8)
    out = np.zeros(m, dtype=np.uint8)
    state = (seed ^ (x_int * _MIX64_CONST)) & _MASK64
    idx = 0
    while idx < m:
        state = _splitmix64(state)
        chunk = state
        take = min(64, m - idx)
        for bit in range(take):
            out[idx + bit] = (chunk >> bit) & 1
        idx += take
    return out


def _seed_stream(n: int, m: int):
    state = (n << 32) ^ m ^ _MIX64_CONST
    while True:
        state = _splitmix64(state)
        yield state


def _build_feature_basis(n: int, m: int, seed: int):
    basis: list[int] = []
    x_basis: list[int] = []
    cols: list[np.ndarray] = []

    for x in range(1, 1 << n):
        Fx = _hash_feature_vector(x, seed, m)
        indep, basis = _add_to_basis_rref_vector(Fx, basis)
        if indep:
            x_basis.append(x)
            cols.append(Fx)
            if len(basis) == m:
                break

    if len(x_basis) != m:
        raise ValueError("Could not build feature basis for mapping")

    X = np.column_stack(cols)
    Xinv = _invert_mod2(X)
    if Xinv is None:
        raise ValueError("Feature basis matrix is singular")
    return x_basis, X, Xinv


def _find_hash_seed_and_basis(n: int, m: int, max_tries: int = 4096):
    stream = _seed_stream(n, m)
    for _ in range(max_tries):
        seed = next(stream)
        try:
            x_basis, X, Xinv = _build_feature_basis(n, m, seed)
        except ValueError:
            continue
        return seed, x_basis, X, Xinv
    raise ValueError("Could not find hash seed with full-rank feature matrix")


def _encode_payload_index_to_Y(index: int, n: int, m: int, x_basis: list[int]):
    q = (1 << n) - 1
    cols = []

    for i in range(m):
        d = (index % q) + 1
        index //= q
        col = np.array([(d >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.uint8)
        cols.append(col)

    return np.column_stack(cols)


def _decode_Y_to_payload_index(
    Y: np.ndarray, n: int, m: int, x_basis: list[int]
) -> int:
    q = (1 << n) - 1
    index = 0
    base = 1

    for i in range(m):
        col = Y[:, i]
        d = 0
        for b in col:
            d = (d << 1) | int(b)

        index += d * base
        base *= q

    return index


def _infer_m_for_message_bitsize(
    packet_bitsize: int, message_bitsize: int
) -> tuple[int, int]:
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    q = (1 << packet_bitsize) - 1
    if q <= 0:
        raise ValueError("packet_bitsize must be >= 2 for chain")

    target = 1 << message_bitsize
    capacity = 1
    m = 0
    max_m = 1 << packet_bitsize

    while capacity < target:
        m += 1
        if m > max_m:
            raise ValueError(
                "message_bitsize exceeds chain capacity for the given packet_bitsize"
            )
        capacity *= q

    return m, capacity


def _nonlinear_generator(n: int, m: int, seed: int, A: np.ndarray):
    def int_to_vec(x: int) -> np.ndarray:
        return np.array([(x >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.uint8)

    def vec_to_int(v: np.ndarray) -> int:
        out = 0
        for b in v:
            out = (out << 1) | int(b)
        return out

    def nonlinear_step(x: int) -> int:
        F = _hash_feature_vector(x, seed, m)
        yb = (A @ F) % 2
        return vec_to_int(yb)

    Nstates = 1 << n
    tails = set(i for i in range(Nstates))
    for x in range(Nstates):
        y = nonlinear_step(x)
        tails.discard(y)

    visited = {0}
    all_states = set(range(Nstates))
    prev = 0
    curr = 1

    def reset():
        nonlocal prev, curr
        prev = 0
        tails_unvisited = set(tails) - visited
        remaining = all_states - visited
        curr = (
            random.choice(list(tails_unvisited))
            if tails_unvisited
            else random.choice(list(remaining))
        )

    reset()

    while True:
        if curr in visited:
            if curr != 0 and prev != 0:
                yield int_to_vec(curr)

            yield np.zeros(n, dtype=np.bool_)

            if len(visited) == Nstates:
                visited = {0}

            reset()

        yield int_to_vec(curr)
        visited.add(curr)
        prev = curr
        curr = nonlinear_step(curr)


class _ChainSamplerState:
    def __init__(
        self,
        packet_bitsize: int,
        message_bitsize: int,
        payload: Message,
        m: int,
        capacity: int,
        seed: int,
        x_basis: list[int],
        Xinv: np.ndarray,
    ):
        self.n = packet_bitsize
        message_bits_arr = _normalize_payload(payload, message_bitsize)
        payload_index = bits_to_int(message_bits_arr)
        if payload_index >= capacity:
            raise ValueError("message_bitsize exceeds chain capacity")

        Y = _encode_payload_index_to_Y(payload_index, self.n, m, x_basis)
        A = (Y @ Xinv) % 2
        self._gen = _nonlinear_generator(self.n, m, seed, A)

    def generate(self) -> Packet:
        return next(self._gen)


class _ChainEstimatorState:
    def __init__(
        self,
        packet_bitsize: int,
        message_bitsize: int,
        m: int,
        seed: int,
        x_basis: list[int],
        X: np.ndarray,
    ):
        self.n = packet_bitsize
        self.message_bitsize = message_bitsize
        self.m = m
        self.seed = seed
        self.x_basis = x_basis
        self.X = X

        self.prev: np.ndarray | None = None
        self.transitions: set[tuple[bytes, bytes]] = set()
        self.basis: list[int] = []
        self.x_cols: list[np.ndarray] = []
        self.y_cols: list[np.ndarray] = []
        self.recovered_payload: np.ndarray | None = None

    def progress(self) -> float:
        if self.recovered_payload is not None:
            return 1.0
        if self.m <= 0:
            return 0.0
        return min(1.0, len(self.x_cols) / self.m)

    def _try_recover(self) -> np.ndarray | None:
        if self.recovered_payload is not None:
            return self.recovered_payload
        if len(self.x_cols) < self.m:
            return None

        Xobs = np.column_stack(self.x_cols)
        Yobs = np.column_stack(self.y_cols)
        Xinv = _invert_mod2(Xobs)
        if Xinv is None:
            return None

        A = (Yobs @ Xinv) % 2
        Yrec = (A @ self.X) % 2
        payload_index = _decode_Y_to_payload_index(Yrec, self.n, self.m, self.x_basis)
        message_bitsize = int_to_bits(payload_index, self.message_bitsize)
        self.recovered_payload = message_bitsize.astype(np.bool_, copy=False)
        return self.recovered_payload

    def feed(self, data: Packet | None) -> Message | None:
        if self.recovered_payload is not None:
            return self.recovered_payload
        if self.message_bitsize == 0:
            self.recovered_payload = np.zeros(0, dtype=np.bool_)
            return self.recovered_payload
        if data is None:
            self.prev = None
            return None

        packet = np.array(data, dtype=np.uint8).reshape(-1) & 1
        if packet.shape != (self.n,):
            raise ValueError("packet shape mismatch")

        if not packet.any():
            self.prev = None
            return self.recovered_payload

        if self.prev is not None and self.prev.any():
            key = (self.prev.tobytes(), packet.tobytes())
            if key not in self.transitions:
                self.transitions.add(key)
                x_int = bits_to_int(self.prev)
                Fx = _hash_feature_vector(x_int, self.seed, self.m)
                indep, new_basis = _add_to_basis_rref_vector(Fx, self.basis)
                if indep:
                    self.basis = new_basis
                    self.x_cols.append(Fx)
                    self.y_cols.append(packet.copy())
                    self._try_recover()

        self.prev = packet.copy()
        return self.recovered_payload


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = config.packet_bitsize
    message_bitsize = config.message_bitsize
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")
    if packet_bitsize <= 0:
        raise ValueError("packet_bitsize must be > 0")
    if message_bitsize == 0:
        raise ValueError("message_bitsize must be > 0")

    m, capacity = _infer_m_for_message_bitsize(packet_bitsize, message_bitsize)
    seed, x_basis, X, Xinv = _find_hash_seed_and_basis(packet_bitsize, m)

    def make_sampler(payload: Message) -> Sampler:
        sampler_state = _ChainSamplerState(
            packet_bitsize,
            message_bitsize,
            payload,
            m,
            capacity,
            seed,
            x_basis,
            Xinv,
        )
        while True:
            yield sampler_state.generate()

    def make_estimator() -> Estimator:
        estimator_state = _ChainEstimatorState(
            packet_bitsize,
            message_bitsize,
            m,
            seed,
            x_basis,
            X,
        )
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


def max_message_bitsize(packet_bitsize: int) -> int:
    return floor_2n_log2_2n_minus_1(packet_bitsize) - 10
