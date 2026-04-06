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
    m = math.ceil(message_bitsize / z)
    n, mask, red = gf2n.make_field(z)
    q = 1 << z

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # Message vector is directly the polynomial coefficients
        message_vector = make_message_vector(message, m, q)  # shape (m,)

        def f(x: np.uint16) -> np.uint16:
            return gf2n.evaluate_poly(x, message_vector, n, mask, red)

        # Precompute transitions and greedily pick longest simple paths.
        nxt = np.empty(q, dtype=np.uint16)
        for x in range(q):
            nxt[x] = f(np.uint16(x))

        alive = np.ones(q, dtype=np.bool_)

        def orbit_path(start: int) -> Tuple[list[np.uint16], np.uint16]:
            seen: set[int] = set()
            path: list[np.uint16] = []
            cur = start

            while alive[cur] and cur not in seen:
                seen.add(cur)
                path.append(np.uint16(cur))
                cur = int(nxt[cur])

            return path, np.uint16(cur)

        paths: list[list[np.uint16]] = []
        score = 0
        min_score_portion = 0.2
        scaler = 100
        target_score = min(max(scaler * m, min_score_portion * q), q)

        while score < target_score and np.any(alive):
            best_path: list[np.uint16] | None = None
            best_terminal = np.uint16(0)
            best_len = -1

            for s in range(q):
                if not alive[s]:
                    continue
                candidate_path, terminal = orbit_path(s)
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

            for v in best_path:
                alive[int(v)] = False

        if not paths:
            paths = [[np.uint16(0), np.uint16(0)]]

        phase = True

        def phased_output(value: np.uint16):
            return uint16_to_bool_array(value + q if phase else value, N)

        while True:
            for path in paths:
                last_idx = len(path) - 1
                for i, state in enumerate(path):
                    yield phased_output(state)

                    # Keep the phase unchanged after terminal so the next path
                    # starts with same phase and acts as reset marker.
                    if i != last_idx:
                        phase = not phase

    # ==========================================================================
    # Estimator fabric
    # ==========================================================================
    def make_estimator() -> Estimator:
        evaluation_examples: Set[Tuple[np.uint16, np.uint16]] = set()
        x: np.uint16 | None = None

        def phase(value: np.ndarray):
            return value[0]

        ## Collect mapping examples
        prev_packet_phase = None
        while len(evaluation_examples) < m:
            packet = yield len(evaluation_examples) / m

            # interrupt streak by delimiter
            if packet is None:
                x = None
                continue

            curr_packet_phase = phase(packet)

            if prev_packet_phase is None:
                prev_packet_phase = not curr_packet_phase
            assert prev_packet_phase is not None

            # interrupt streak by reset indicator
            same_phase = phase(packet) == prev_packet_phase
            prev_packet_phase = curr_packet_phase

            # decode packet value (strip the phase bit)
            y = bool_array_to_uint16(packet)
            y = np.uint16(y & (q - 1))

            if same_phase:
                # Treat as a fresh start: avoid linking across chains,
                # but keep the current node so its outgoing edge is still learnable.
                x = y
                continue

            # record example
            if x is not None:
                evaluation_examples.add((x, y))
            x = y

        ## Reconstruct message
        outputs = np.empty(m, dtype=np.uint16)
        inputs = np.empty(m, dtype=np.uint16)

        for i, (x_val, y_val) in enumerate(evaluation_examples):
            inputs[i] = x_val
            outputs[i] = y_val

        message_vector = gf2n.interpolate_poly(outputs, inputs, n, mask, red)
        message = message_from_message_vector(message_vector, message_bitsize, q)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    zn = packet_bitsize - 1
    return zn * (1 << zn)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],  # pGB, pBG, pG, pB
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
