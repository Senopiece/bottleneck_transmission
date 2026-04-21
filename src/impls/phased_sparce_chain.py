import math
import random
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from ._utils.conversions import bool_array_to_uint16, uint16_to_bool_array
from ._interface import Config, Estimator, Message, Protocol, Sampler

# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1.0

_U64_MASK = (1 << 64) - 1
_PEELING_OVERHEAD = 1.43
_EQUATION_CONFIRMATIONS = 2
_FAMILY_CANDIDATES = 8
_FAMILY_SAMPLE_X = 128


def _robust_soliton_cdf(k: int, c: float = 0.1, delta: float = 0.05) -> list[float]:
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

    cdf = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf


def _mix64(value: int) -> int:
    x = (value + 0x9E3779B97F4A7C15) & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    x = x ^ (x >> 31)
    return x & _U64_MASK


def _subset_from_seed(seed: int, k: int, cdf: list[float]) -> list[int]:
    if k <= 0:
        return []

    rng = random.Random(seed)
    draw = rng.random()
    degree = 1
    for idx, threshold in enumerate(cdf):
        if draw <= threshold:
            degree = idx + 1
            break
    degree = max(1, min(degree, k))

    if degree == k:
        return list(range(k))

    subset = rng.sample(range(k), degree)
    subset.sort()
    return subset


def _seed_for_equation(
    x: int, bit_offset: int, k: int, salt: int, family_id: int
) -> int:
    payload = (
        ((x & 0xFFFF) << 32)
        ^ ((bit_offset & 0xFF) << 24)
        ^ ((k & 0xFFFFFF) << 1)
        ^ (salt & 0x1)
        ^ ((family_id & 0xFF) << 48)
    )
    return _mix64(payload)


def _normalize_message(payload: Message, bitsize: int) -> np.ndarray:
    bits = np.array(payload, dtype=np.uint8).reshape(-1) & 1
    if bits.size != bitsize:
        raise ValueError(f"Expected payload of {bitsize} bits, got {bits.size}")
    return bits


def _xor_subset(message_bits: np.ndarray, subset: Iterable[int]) -> int:
    parity = 0
    for idx in subset:
        parity ^= int(message_bits[idx])
    return parity


def _sample_x_points(q: int, sample_count: int, seed: int) -> list[int]:
    if q <= sample_count:
        return list(range(q))
    rng = random.Random(seed)
    chosen = rng.sample(range(q), sample_count)
    chosen.sort()
    return chosen


def _peeling_structural_score(k: int, equations: list[list[int]]) -> tuple[int, int, int]:
    if k <= 0:
        return (0, 0, 0)
    if not equations:
        return (0, 0, 0)

    eq_count = len(equations)
    var_to_eqs: list[list[int]] = [[] for _ in range(k)]
    degrees = np.empty(eq_count, dtype=np.int32)
    eq_unknown = [set(eq) for eq in equations]

    for ei, eq in enumerate(equations):
        degrees[ei] = len(eq)
        for v in eq:
            var_to_eqs[v].append(ei)

    q1: list[int] = [i for i in range(eq_count) if degrees[i] == 1]
    solved = np.zeros(k, dtype=np.bool_)
    solved_count = 0
    singleton_steps = 0

    while q1:
        ei = q1.pop()
        if degrees[ei] != 1:
            continue
        singleton_steps += 1
        v = next(iter(eq_unknown[ei]))
        if solved[v]:
            continue
        solved[v] = True
        solved_count += 1
        for nei in var_to_eqs[v]:
            if degrees[nei] <= 0:
                continue
            if v in eq_unknown[nei]:
                eq_unknown[nei].remove(v)
                degrees[nei] -= 1
                if degrees[nei] == 1:
                    q1.append(nei)

    covered = int(np.count_nonzero([len(v) > 0 for v in var_to_eqs]))
    return (solved_count, covered, singleton_steps)


def _choose_family_id(k: int, z: int, q: int, cdf: list[float], salt: int) -> int:
    if k <= 1:
        return 0

    sampled_x = _sample_x_points(
        q,
        min(q, _FAMILY_SAMPLE_X),
        seed=(0xA51E + 97 * salt + 131 * k + 17 * z + 19 * q),
    )

    best_family = 0
    best_score: tuple[int, int, int] | None = None
    for family_id in range(_FAMILY_CANDIDATES):
        equations: list[list[int]] = []
        for x in sampled_x:
            for bit_offset in range(z):
                seed = _seed_for_equation(x, bit_offset, k, salt, family_id)
                equations.append(_subset_from_seed(seed, k, cdf))

        score = _peeling_structural_score(k, equations)
        if best_score is None or score > best_score:
            best_score = score
            best_family = family_id

    return best_family


class _SparsePeelingDecoder:
    def __init__(self, k: int):
        self.k = k
        self.known = np.full(k, -1, dtype=np.int8)
        self.pending: list[tuple[set[int], int]] = []
        self.inconsistent = False

    def _assign(self, idx: int, value: int) -> bool:
        current = int(self.known[idx])
        v = value & 1
        if current == -1:
            self.known[idx] = v
            return True
        if current != v:
            self.inconsistent = True
        return False

    def _propagate(self):
        if not self.pending:
            return

        progress = True
        while progress:
            progress = False
            next_pending: list[tuple[set[int], int]] = []
            for subset, rhs in self.pending:
                if not subset:
                    if rhs != 0:
                        self.inconsistent = True
                    continue

                reduced_subset = set(subset)
                reduced_rhs = rhs
                for idx in list(reduced_subset):
                    val = int(self.known[idx])
                    if val != -1:
                        reduced_rhs ^= val
                        reduced_subset.remove(idx)

                if not reduced_subset:
                    if reduced_rhs != 0:
                        self.inconsistent = True
                    continue

                if len(reduced_subset) == 1:
                    idx = next(iter(reduced_subset))
                    if self._assign(idx, reduced_rhs):
                        progress = True
                    continue

                next_pending.append((reduced_subset, reduced_rhs))
            self.pending = next_pending

    def add_equation(self, indices: list[int], rhs: int):
        if self.k == 0:
            return

        reduced_rhs = rhs & 1
        unknown: list[int] = []
        for idx in indices:
            val = int(self.known[idx])
            if val == -1:
                unknown.append(idx)
            else:
                reduced_rhs ^= val

        if not unknown:
            if reduced_rhs != 0:
                self.inconsistent = True
            return

        if len(unknown) == 1:
            self._assign(unknown[0], reduced_rhs)
        else:
            self.pending.append((set(unknown), reduced_rhs))
        self._propagate()

    def solved_count(self) -> int:
        if self.k == 0:
            return 0
        return int(np.count_nonzero(self.known != -1))

    def is_solved(self) -> bool:
        return self.solved_count() == self.k

    def result(self) -> np.ndarray:
        if self.k == 0:
            return np.zeros(0, dtype=np.bool_)
        if not self.is_solved():
            raise ValueError("decoder result requested before solving")
        return self.known.astype(np.bool_, copy=False)


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = config.packet_bitsize
    message_bitsize = config.message_bitsize

    N = packet_bitsize
    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2 to reserve a phase bit.")
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0.")

    z = N - 1
    q = 1 << z
    max_half_bits = int((z * q) / _PEELING_OVERHEAD)
    max_bits = 2 * max_half_bits
    if message_bitsize > max_bits:
        raise ValueError(
            f"message_bitsize too large for phased_sparce_chain: max {max_bits} for packet_bitsize={packet_bitsize}"
        )

    half_bits_a = (message_bitsize + 1) // 2
    half_bits_b = message_bitsize - half_bits_a
    cdf_a = _robust_soliton_cdf(half_bits_a) if half_bits_a > 0 else [1.0]
    cdf_b = _robust_soliton_cdf(half_bits_b) if half_bits_b > 0 else [1.0]
    family_a = _choose_family_id(half_bits_a, z, q, cdf_a, salt=0)
    family_b = _choose_family_id(half_bits_b, z, q, cdf_b, salt=1)

    def make_sampler(message: Message) -> Sampler:
        msg = _normalize_message(message, message_bitsize)
        msg_a = msg[:half_bits_a]
        msg_b = msg[half_bits_a:]

        cache_a: Dict[int, List[List[int]]] = {}
        cache_b: Dict[int, List[List[int]]] = {}

        def plans_for_x(
            x: int,
            k: int,
            cdf: list[float],
            salt: int,
            family_id: int,
            cache: Dict[int, List[List[int]]],
        ) -> List[List[int]]:
            found = cache.get(x)
            if found is not None:
                return found
            plan: List[List[int]] = []
            for bit_offset in range(z):
                seed = _seed_for_equation(x, bit_offset, k, salt, family_id)
                plan.append(_subset_from_seed(seed, k, cdf))
            cache[x] = plan
            return plan

        def f_bits(
            x: int,
            bits: np.ndarray,
            cdf: list[float],
            salt: int,
            family_id: int,
            cache: Dict[int, List[List[int]]],
        ) -> int:
            if bits.size == 0:
                return 0
            plan = plans_for_x(x, bits.size, cdf, salt, family_id, cache)
            out = 0
            for bit_offset in range(z):
                out = (out << 1) | _xor_subset(bits, plan[bit_offset])
            return out

        nxt_a = np.empty(q, dtype=np.uint16)
        nxt_b = np.empty(q, dtype=np.uint16)
        for x in range(q):
            nxt_a[x] = np.uint16(f_bits(x, msg_a, cdf_a, 0, family_a, cache_a))
            nxt_b[x] = np.uint16(f_bits(x, msg_b, cdf_b, 1, family_b, cache_b))

        alive_a = np.ones(q, dtype=np.bool_)
        alive_b = np.ones(q, dtype=np.bool_)

        def orbit_path(start: int, start_phase_b: bool):
            seen: set[Tuple[bool, int]] = set()
            path: list[Tuple[np.uint16, bool]] = []
            cur = start
            phase_b = start_phase_b

            while True:
                if phase_b:
                    if (not alive_b[cur]) or ((phase_b, cur) in seen):
                        return path, (np.uint16(cur), phase_b)
                else:
                    if (not alive_a[cur]) or ((phase_b, cur) in seen):
                        return path, (np.uint16(cur), phase_b)

                seen.add((phase_b, cur))
                path.append((np.uint16(cur), phase_b))

                if phase_b:
                    cur = int(nxt_b[cur])
                else:
                    cur = int(nxt_a[cur])
                phase_b = not phase_b

        paths: list[list[Tuple[np.uint16, bool]]] = []
        m = half_bits_a + half_bits_b
        score = 0
        min_score_portion = 0.2
        scaler = 100
        q_total = 2 * q
        target_score = min(max(scaler * m, min_score_portion * q_total), q_total)

        while score < target_score and (np.any(alive_a) or np.any(alive_b)):
            best_path: list[Tuple[np.uint16, bool]] | None = None
            best_terminal: Tuple[np.uint16, bool] = (np.uint16(0), False)
            best_len = -1

            for s in range(q):
                if alive_a[s]:
                    candidate_path, terminal = orbit_path(s, False)
                    candidate_len = len(candidate_path)
                    if candidate_len > best_len:
                        best_len = candidate_len
                        best_path = candidate_path
                        best_terminal = terminal

                if alive_b[s]:
                    candidate_path, terminal = orbit_path(s, True)
                    candidate_len = len(candidate_path)
                    if candidate_len > best_len:
                        best_len = candidate_len
                        best_path = candidate_path
                        best_terminal = terminal

            if best_path is None or best_len <= 0:
                break

            emitted_path = best_path + [best_terminal]
            paths.append(emitted_path)
            score += len(emitted_path) - 1

            for v, v_phase_b in best_path:
                if v_phase_b:
                    alive_b[int(v)] = False
                else:
                    alive_a[int(v)] = False

        if not paths:
            paths = [[(np.uint16(0), False), (np.uint16(0), False)]]

        paths_by_start_phase = {
            False: [p for p in paths if not p[0][1]],
            True: [p for p in paths if p[0][1]],
        }
        phase_cursor = {False: 0, True: 0}

        def next_path(required_start_phase_b: bool):
            primary = paths_by_start_phase[required_start_phase_b]
            if primary:
                idx = phase_cursor[required_start_phase_b] % len(primary)
                phase_cursor[required_start_phase_b] += 1
                return primary[idx]

            fallback_phase_b = not required_start_phase_b
            fallback = paths_by_start_phase[fallback_phase_b]
            if fallback:
                idx = phase_cursor[fallback_phase_b] % len(fallback)
                phase_cursor[fallback_phase_b] += 1
                return fallback[idx]

            return paths[0]

        def phased_output(value: np.uint16, is_phase_b: bool):
            return uint16_to_bool_array(value + (q if is_phase_b else 0), N)

        required_start_phase_b = paths[0][0][1]
        while True:
            path = next_path(required_start_phase_b)
            for i, (state, state_phase_b) in enumerate(path):
                if i > 0 and path[i - 1][1] == state_phase_b:
                    raise RuntimeError(
                        "Sampler path has non-alternating phase sequence."
                    )
                yield phased_output(state, state_phase_b)
            required_start_phase_b = path[-1][1]

    def make_estimator() -> Estimator:
        decoder_a = _SparsePeelingDecoder(half_bits_a)
        decoder_b = _SparsePeelingDecoder(half_bits_b)
        committed_a: Set[tuple[int, tuple[int, ...]]] = set()
        committed_b: Set[tuple[int, tuple[int, ...]]] = set()
        votes_a: Dict[tuple[int, tuple[int, ...]], List[int]] = {}
        votes_b: Dict[tuple[int, tuple[int, ...]], List[int]] = {}

        plan_cache_a: Dict[int, List[List[int]]] = {}
        plan_cache_b: Dict[int, List[List[int]]] = {}
        x: np.uint16 | None = None
        prev_phase = None

        def phase(value: np.ndarray):
            return value[0]

        def plans_for_x(
            x_val: int,
            k: int,
            cdf: list[float],
            salt: int,
            family_id: int,
            cache: Dict[int, List[List[int]]],
        ) -> List[List[int]]:
            found = cache.get(x_val)
            if found is not None:
                return found
            plan: List[List[int]] = []
            for bit_offset in range(z):
                seed = _seed_for_equation(x_val, bit_offset, k, salt, family_id)
                plan.append(_subset_from_seed(seed, k, cdf))
            cache[x_val] = plan
            return plan

        def add_transition(
            decoder: _SparsePeelingDecoder,
            committed: Set[tuple[int, tuple[int, ...]]],
            votes: Dict[tuple[int, tuple[int, ...]], List[int]],
            x_val: int,
            y_val: int,
            cdf: list[float],
            salt: int,
            family_id: int,
            cache: Dict[int, List[List[int]]],
        ):
            if decoder.k == 0:
                return
            plan = plans_for_x(x_val, decoder.k, cdf, salt, family_id, cache)
            for bit_offset in range(z):
                rhs = (y_val >> (z - 1 - bit_offset)) & 1
                subset = plan[bit_offset]
                key = (bit_offset, tuple(subset))
                if key in committed:
                    continue

                key_votes = votes.get(key)
                if key_votes is None:
                    key_votes = [0, 0]
                    votes[key] = key_votes
                key_votes[rhs] += 1

                strong = key_votes[rhs] >= _EQUATION_CONFIRMATIONS
                dominant = key_votes[rhs] > key_votes[rhs ^ 1]
                if strong and dominant:
                    decoder.add_equation(subset, rhs)
                    committed.add(key)
                    del votes[key]

        total_bits = half_bits_a + half_bits_b
        while not (decoder_a.is_solved() and decoder_b.is_solved()):
            solved_bits = decoder_a.solved_count() + decoder_b.solved_count()
            progress = (solved_bits / total_bits) if total_bits > 0 else 1.0
            packet = yield progress

            if packet is None:
                x = None
                prev_phase = None
                continue

            curr_packet_phase = phase(packet)
            if prev_phase is None:
                prev_phase = not curr_packet_phase
            assert prev_phase is not None

            same_phase = curr_packet_phase == prev_phase
            y = bool_array_to_uint16(packet)
            y = np.uint16(y & (q - 1))

            prev_phase_value = prev_phase
            prev_phase = curr_packet_phase

            if same_phase:
                x = y
                continue

            if x is not None:
                x_int = int(x)
                y_int = int(y)
                if (not bool(prev_phase_value)) and bool(curr_packet_phase):
                    add_transition(
                        decoder_a,
                        committed_a,
                        votes_a,
                        x_int,
                        y_int,
                        cdf_a,
                        0,
                        family_a,
                        plan_cache_a,
                    )
                elif bool(prev_phase_value) and (not bool(curr_packet_phase)):
                    add_transition(
                        decoder_b,
                        committed_b,
                        votes_b,
                        x_int,
                        y_int,
                        cdf_b,
                        1,
                        family_b,
                        plan_cache_b,
                    )
            x = y

        if decoder_a.inconsistent or decoder_b.inconsistent:
            raise ValueError("Peeling decoder became inconsistent")

        message = np.concatenate([decoder_a.result(), decoder_b.result()])
        return message.astype(np.bool_, copy=False)

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    zn = packet_bitsize - 1
    q = 1 << zn
    return 2 * int((zn * q) / _PEELING_OVERHEAD)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],  # pGB, pBG, pG, pB
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
