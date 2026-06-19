from collections import deque
from typing import Set, Tuple

import numpy as np

from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.fields import gf2n
from ._interface import Config, Protocol, Message, Sampler, Estimator
from ._utils.intmath import (
    floor_2n_m1_log2_2n_m1,
    min_m_such_that_2n_minus_1_pow_k_ge_2p,
)

# Domain:
# skip_probability: [0, 1)
# corruption_probability: 0
# skip_observation: 1.0

SEGMENT_PROBES = 4


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

    if not (1 <= packet_bitsize <= 16):
        raise ValueError(
            "GF(2^packet_bitsize) polynomial book integrated only for packet_bitsize in [1,16]."
        )

    # ==========================================================================
    # Precomputations
    # ==========================================================================

    N = packet_bitsize
    q = 1 << N
    z = q - 1
    m = min_m_such_that_2n_minus_1_pow_k_ge_2p(
        message_bitsize,
        packet_bitsize,
        z,
    )
    n, mask, red = gf2n.make_field(N)

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # Message vector is evaluation at [0, m-1] points.
        message_vector = make_message_vector(message, m, z)  # shape (m,)

        # Convert it to falling factorial coeffs.
        coeffs = gf2n.first_points_to_falling_factorial_coeffs(
            message_vector, n, mask, red
        )

        def f(x: np.uint16) -> np.uint16:
            return gf2n.evaluate_poly_falling_factorial(
                np.uint16(x), coeffs, n, mask, red
            )

        delimiter = z  # all ones

        # Precompute transitions for non-delimiter states [0, z-1].
        nxt = np.empty(z, dtype=np.uint16)
        for x in range(z):
            nxt[x] = f(np.uint16(x))

        window = m
        recent_counts = np.zeros(z, dtype=np.int32)
        recent: deque[int] = deque()

        probes = min(SEGMENT_PROBES, z)
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

                # Delimiter is a hard terminal in chain_2n.
                if nxt_id == delimiter:
                    return segment
                if recent_counts[nxt_id] > 0 or nxt_id in visited:
                    return segment

                visited.add(nxt_id)
                cur = nxt_id

        def choose_probe_starts() -> np.ndarray:
            # Start from nodes outside last-m non-delimiter emitted states.
            eligible = np.flatnonzero(recent_counts == 0)
            if eligible.size == 0:
                eligible = np.arange(z, dtype=np.int32)

            probe_count = min(probes, int(eligible.size))
            if probe_count <= 0:
                probe_count = 1

            # Distinct starts when possible.
            return rng.choice(eligible, size=probe_count, replace=False)

        delimiter_packet = np.ones(N, dtype=np.bool_)
        while True:
            starts = choose_probe_starts()

            best_segment: list[int] = []
            for start in starts:
                candidate = build_virtual_segment(int(start))
                if len(candidate) > len(best_segment):
                    best_segment = candidate

            if not best_segment:
                best_segment = [0, int(nxt[0])]

            for state in best_segment:
                if state == delimiter:
                    yield delimiter_packet
                else:
                    remember(state)
                    yield uint16_to_bool_array(np.uint16(state), N)

            # Explicit boundary marker between virtual segments.
            yield delimiter_packet

    # ==========================================================================
    # Estimator fabric
    # ==========================================================================
    def make_estimator() -> Estimator:
        evaluation_examples: Set[Tuple[np.uint16, np.uint16]] = set()
        x: np.uint16 | None = None

        # Collect mapping examples.
        while len(evaluation_examples) < m:
            packet = yield len(evaluation_examples) / m

            # Interrupt streak by delimiter or reset indicator.
            if packet is None or np.all(packet == 1):
                x = None
                continue

            # Record example.
            y = bool_array_to_uint16(packet)
            if x is not None:
                evaluation_examples.add((x, y))
            x = y

        # Reconstruct message.
        outputs = np.empty(m, dtype=np.uint16)
        inputs = np.empty(m, dtype=np.uint16)

        for i, (x_val, y_val) in enumerate(evaluation_examples):
            inputs[i] = x_val
            outputs[i] = y_val

        coeffs = gf2n.interpolate_poly_falling_factorial(outputs, inputs, n, mask, red)
        message_vector = gf2n.evaluate_poly_falling_factorial_first_points(
            coeffs, n, mask, red
        )
        message = message_from_message_vector(message_vector, message_bitsize, z)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    return floor_2n_m1_log2_2n_m1(packet_bitsize) - 30  # TODO: remove this -30


def estimate_packets_until_reconstructed(
    deletion_prob: float, packet_bitsize: int, message_bitsize: int
):
    return None
