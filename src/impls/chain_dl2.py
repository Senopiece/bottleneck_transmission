import math
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

# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1.0


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = config.packet_bitsize
    message_bitsize = config.message_bitsize

    # ==========================================================================
    # Precomputations
    # ==========================================================================

    N = packet_bitsize
    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2 to reserve a phase bit.")

    # One bit is reserved for phase, so each symbol carries (N-1) bits.
    z = N - 1
    n, mask, red = gf2n.make_field(z)
    q = 1 << z
    max_bits = 2 * z * q
    if message_bitsize > max_bits:
        raise ValueError(
            f"message_bitsize too large for chain_dl2: max {max_bits} for packet_bitsize={packet_bitsize}"
        )

    # Split payload in half: f1 encodes first half, f2 encodes second half.
    half_bits_a = (message_bitsize + 1) // 2
    half_bits_b = message_bitsize - half_bits_a

    m1 = math.ceil(half_bits_a / z) if half_bits_a > 0 else 0
    m2 = math.ceil(half_bits_b / z) if half_bits_b > 0 else 0
    if m1 > q or m2 > q:
        raise ValueError(
            f"message_bitsize too large for chain_dl2: requires m1={m1}, m2={m2}, but q={q}"
        )

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        msg_a = message[:half_bits_a]
        msg_b = message[half_bits_a:]

        # Message vectors are directly the polynomial coefficients
        message_vector_a = (
            make_message_vector(msg_a, m1, q) if m1 > 0 else np.empty(0, dtype=np.uint16)
        )
        message_vector_b = (
            make_message_vector(msg_b, m2, q) if m2 > 0 else np.empty(0, dtype=np.uint16)
        )

        def f1(x: np.uint16) -> np.uint16:
            return (
                gf2n.evaluate_poly(x, message_vector_a, n, mask, red)
                if m1 > 0
                else np.uint16(0)
            )

        def f2(x: np.uint16) -> np.uint16:
            return (
                gf2n.evaluate_poly(x, message_vector_b, n, mask, red)
                if m2 > 0
                else np.uint16(0)
            )

        all_states = set(np.uint16(i) for i in range(q))

        # ----------------------------------------------------------------------
        # Precompute tails for each phase domain:
        # tailsA are nodes in A unreachable by f2 outputs (B -> A),
        # tailsB are nodes in B unreachable by f1 outputs (A -> B).
        # ----------------------------------------------------------------------
        tails_a = set(all_states)
        for x in range(q):
            y = f2(np.uint16(x))
            if y in tails_a:
                tails_a.discard(y)

        tails_b = set(all_states)
        for x in range(q):
            y = f1(np.uint16(x))
            if y in tails_b:
                tails_b.discard(y)

        visited_a: Set[np.uint16] = set()
        visited_b: Set[np.uint16] = set()

        def phased_output(value: np.uint16, is_phase_b: bool):
            return uint16_to_bool_array(value + (q if is_phase_b else 0), N)

        # pick a new start state, prioritizing unvisited tails for the current phase
        def reset(is_phase_b: bool) -> np.uint16:
            tails = tails_b if is_phase_b else tails_a
            visited = visited_b if is_phase_b else visited_a

            tails_unvisited = tails - visited
            if tails_unvisited:
                return next(iter(tails_unvisited))

            remaining = all_states - visited
            if remaining:
                return next(iter(remaining))

            visited_a.clear()
            visited_b.clear()
            candidates = tails_b if is_phase_b else tails_a
            return next(iter(candidates)) if candidates else np.uint16(0)

        phase_b = False  # start in A
        curr = reset(phase_b)

        while True:
            # yield, mark visited and compute next
            yield phased_output(curr, phase_b)
            if phase_b:
                visited_b.add(curr)
                nxt = f2(curr)  # B -> A
            else:
                visited_a.add(curr)
                nxt = f1(curr)  # A -> B

            # if next would repeat a visited state, break adjacency with a delimiter and jump
            phase_b = not phase_b
            visited_next = visited_b if phase_b else visited_a
            if nxt in visited_next:
                yield phased_output(nxt, phase_b)
                curr = reset(phase_b)
            else:
                curr = nxt

    # ==========================================================================
    # Estimator fabric
    # ==========================================================================
    def make_estimator() -> Estimator:
        evaluation_examples_a: Set[Tuple[np.uint16, np.uint16]] = set()
        evaluation_examples_b: Set[Tuple[np.uint16, np.uint16]] = set()
        x: np.uint16 | None = None
        prev_phase = None

        def phase(value: np.ndarray):
            return value[0]

        ## Collect mapping examples
        total_m = m1 + m2
        while len(evaluation_examples_a) < m1 or len(evaluation_examples_b) < m2:
            progress = (
                (len(evaluation_examples_a) + len(evaluation_examples_b)) / total_m
                if total_m > 0
                else 1.0
            )
            packet = yield progress

            # interrupt streak by delimiter
            if packet is None:
                x = None
                prev_phase = None
                continue

            curr_packet_phase = phase(packet)

            if prev_phase is None:
                prev_phase = not curr_packet_phase
            assert prev_phase is not None

            # interrupt streak by reset indicator
            same_phase = curr_packet_phase == prev_phase

            # decode packet value (strip the phase bit)
            y = bool_array_to_uint16(packet)
            y = np.uint16(y & (q - 1))

            prev_phase_value = prev_phase
            prev_phase = curr_packet_phase

            if same_phase:
                # Treat as a fresh start: avoid linking across chains,
                # but keep the current node so its outgoing edge is still learnable.
                x = y
                continue

            # record example
            if x is not None:
                if (not bool(prev_phase_value)) and bool(curr_packet_phase):
                    if len(evaluation_examples_a) < m1:
                        evaluation_examples_a.add((x, y))
                elif bool(prev_phase_value) and (not bool(curr_packet_phase)):
                    if len(evaluation_examples_b) < m2:
                        evaluation_examples_b.add((x, y))
            x = y

        ## Reconstruct message
        message_parts = []
        if m1 > 0:
            outputs_a = np.empty(m1, dtype=np.uint16)
            inputs_a = np.empty(m1, dtype=np.uint16)
            for i, (x_val, y_val) in enumerate(evaluation_examples_a):
                inputs_a[i] = x_val
                outputs_a[i] = y_val
            message_vector_a = gf2n.interpolate_poly(outputs_a, inputs_a, n, mask, red)
            message_part_a = message_from_message_vector(
                message_vector_a, half_bits_a, q
            )
            message_parts.append(message_part_a)
        else:
            message_parts.append(np.zeros(0, dtype=np.bool_))

        if m2 > 0:
            outputs_b = np.empty(m2, dtype=np.uint16)
            inputs_b = np.empty(m2, dtype=np.uint16)
            for i, (x_val, y_val) in enumerate(evaluation_examples_b):
                inputs_b[i] = x_val
                outputs_b[i] = y_val
            message_vector_b = gf2n.interpolate_poly(outputs_b, inputs_b, n, mask, red)
            message_part_b = message_from_message_vector(
                message_vector_b, half_bits_b, q
            )
            message_parts.append(message_part_b)
        else:
            message_parts.append(np.zeros(0, dtype=np.bool_))

        message = np.concatenate(message_parts)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    zn = packet_bitsize - 1
    return zn * (1 << packet_bitsize)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],  # pGB, pBG, pG, pB
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
