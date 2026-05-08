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
# skip_probability: [0, 1)
# corruption_probability: 0
# skip_observation: 1.0


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
            make_message_vector(msg_a, m1, q)
            if m1 > 0
            else np.empty(0, dtype=np.uint16)
        )
        message_vector_b = (
            make_message_vector(msg_b, m2, q)
            if m2 > 0
            else np.empty(0, dtype=np.uint16)
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

        # Precompute transitions and greedily pick longest simple paths over
        # phase-state pairs (A/B domains).
        nxt_a = np.empty(q, dtype=np.uint16)  # A -> B via f1
        nxt_b = np.empty(q, dtype=np.uint16)  # B -> A via f2
        for x in range(q):
            xv = np.uint16(x)
            nxt_a[x] = f1(xv)
            nxt_b[x] = f2(xv)

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
        m = m1 + m2
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
    return zn * (1 << packet_bitsize) - 30  # TODO: remove this -30


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],  # pGB, pBG, pG, pB
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
