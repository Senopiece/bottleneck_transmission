from collections import deque
from typing import Set, Tuple

import numpy as np

from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.fields import gfp
from ._interface import Config, Protocol, Message, Sampler, Estimator

# Domain:
# skip_probability: [0, 1)
# corruption_probability: 0
# skip_observation: 1.0

SEGMENT_PROBES = 4


def _min_m_q_pow_ge_2p(p: int, q: int) -> int:
    """Smallest m such that q^m >= 2^p."""
    if p <= 0:
        return 0
    if q <= 1:
        raise ValueError("q must be > 1")
    target = 1 << p
    lo, hi = 0, p
    while lo < hi:
        mid = (lo + hi) // 2
        if pow(q, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def sampler_seed(
    message_vector: np.ndarray, packet_bitsize: int, message_bitsize: int
) -> int:
    seed = 1469598103934665603
    fnv_prime = 1099511628211

    seed ^= packet_bitsize & 0xFFFF
    seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    seed ^= message_bitsize & 0xFFFFFFFF
    seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF

    for coeff in message_vector:
        seed ^= int(coeff) + 1
        seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF

    return seed


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = config.packet_bitsize
    message_bitsize = config.message_bitsize

    # ==========================================================================
    # Precomputations
    # ==========================================================================

    N = packet_bitsize
    q, _ = gfp.make_field(N)
    m = _min_m_q_pow_ge_2p(message_bitsize, q)

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        message_vector = make_message_vector(message, m, q)

        def f(x: np.uint16) -> np.uint16:
            return gfp.evaluate_poly(x, message_vector, q)

        nxt = np.empty(q, dtype=np.uint16)
        for x in range(q):
            nxt[x] = f(np.uint16(x))

        window = m
        recent_counts = np.zeros(q, dtype=np.int32)
        recent: deque[int] = deque()
        input_emit_counts = np.zeros(q, dtype=np.int32)

        probes = min(SEGMENT_PROBES, q)
        rng = np.random.default_rng(
            sampler_seed(message_vector, packet_bitsize, message_bitsize)
        )

        def remember(packet_id: int) -> None:
            if window <= 0:
                return
            recent.append(packet_id)
            recent_counts[packet_id] += 1
            if len(recent) > window:
                dropped = recent.popleft()
                recent_counts[dropped] -= 1

        def build_virtual_segment(start: int) -> list[int]:
            visited: set[int] = {start}
            segment = [start]
            cur = start

            while True:
                nxt_id = int(nxt[cur])
                segment.append(nxt_id)
                if recent_counts[nxt_id] > 0 or nxt_id in visited:
                    return segment
                visited.add(nxt_id)
                cur = nxt_id

        def choose_probe_starts() -> np.ndarray:
            min_count = int(np.min(input_emit_counts))
            rare_inputs = np.flatnonzero(input_emit_counts == min_count)
            eligible = rare_inputs[recent_counts[rare_inputs] == 0]
            if eligible.size == 0:
                eligible = rare_inputs

            probe_count = min(probes, int(eligible.size))
            if probe_count <= 0:
                probe_count = 1

            return rng.choice(
                eligible.astype(np.int32), size=probe_count, replace=False
            )

        def segment_score(segment: list[int]) -> tuple[int, float, int]:
            inputs = segment[:-1]
            if not inputs:
                return (0, 0.0, 0)

            min_count = int(np.min(input_emit_counts))
            rare_edges = 0
            weighted_freshness = 0.0
            for x in inputs:
                count = int(input_emit_counts[x])
                if count == min_count:
                    rare_edges += 1
                weighted_freshness += 1.0 / float(count + 1)

            return (rare_edges, weighted_freshness, len(segment))

        delimiter = np.ones(N, dtype=np.bool_)
        while True:
            starts = choose_probe_starts()

            best_segment: list[int] = []
            best_score: tuple[int, float, int] | None = None
            for start in starts:
                candidate = build_virtual_segment(int(start))
                candidate_score = segment_score(candidate)
                if best_score is None or candidate_score > best_score:
                    best_segment = candidate
                    best_score = candidate_score

            if not best_segment:
                best_segment = [0, int(nxt[0])]

            for state in best_segment[:-1]:
                input_emit_counts[state] += 1

            for state in best_segment:
                remember(state)
                yield uint16_to_bool_array(np.uint16(state), N)

            yield delimiter

    # ==========================================================================
    # Estimator fabric
    # ==========================================================================
    def make_estimator() -> Estimator:
        evaluation_examples: Set[Tuple[np.uint16, np.uint16]] = set()
        x: np.uint16 | None = None

        while len(evaluation_examples) < m:
            packet = yield len(evaluation_examples) / m

            if packet is None or np.all(packet == 1):
                x = None
                continue

            y = bool_array_to_uint16(packet)
            if x is not None:
                evaluation_examples.add((x, y))
            x = y

        outputs = np.empty(m, dtype=np.uint16)
        inputs = np.empty(m, dtype=np.uint16)

        for i, (x_val, y_val) in enumerate(evaluation_examples):
            inputs[i] = x_val
            outputs[i] = y_val

        message_vector = gfp.interpolate_poly(outputs, inputs, q)
        message = message_from_message_vector(message_vector, message_bitsize, q)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if packet_bitsize < 2:
        return 0
    q = gfp.largest_prime_below_2n(packet_bitsize)
    # floor(q * log2(q)) = pow(q, q).bit_length() - 1
    return pow(q, q).bit_length() - 1


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
