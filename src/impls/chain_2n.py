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
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1.0


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
        # Message vector is evaluation at [0, m-1] points
        # It is crucial to define it as evaluations at some points
        # to make sure there are at least m points not touching the 2^n - 1 value
        # - that is the criterion to make polynomial recoverable
        message_vector = make_message_vector(message, m, z)  # shape (m,)

        # Convert it to falling factorial coeffs
        coeffs = gf2n.first_points_to_falling_factorial_coeffs(
            message_vector, n, mask, red
        )

        def f(x: np.uint16) -> np.uint16:
            return gf2n.evaluate_poly_falling_factorial(
                np.uint16(x), coeffs, n, mask, red
            )

        delimiter = np.uint16(z)  # all ones

        # Greedy longest paths over non-delimiter states [0, z-1].
        nxt = np.empty(z, dtype=np.uint16)
        for x in range(z):
            nxt[x] = f(np.uint16(x))

        alive = np.ones(z, dtype=np.bool_)

        def orbit_path(start: int) -> Tuple[list[np.uint16], np.uint16]:
            seen: set[int] = set()
            path: list[np.uint16] = []
            cur = start

            while alive[cur] and cur not in seen:
                seen.add(cur)
                path.append(np.uint16(cur))
                nxt_cur = int(nxt[cur])
                if nxt_cur == z:
                    return path, delimiter
                cur = nxt_cur

            return path, np.uint16(cur)

        paths: list[list[np.uint16]] = []
        score = 0
        min_score_portion = 0.2
        scaler = 100
        target_score = min(max(scaler * m, min_score_portion * q), q)

        while score < target_score and np.any(alive):
            best_path: list[np.uint16] | None = None
            best_terminal = delimiter
            best_len = -1

            for s in range(z):
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

        while True:
            for path in paths:
                for state in path:
                    if state == delimiter:
                        yield np.ones(N, dtype=np.bool_)
                    else:
                        yield uint16_to_bool_array(state, N)
                yield np.ones(N, dtype=np.bool_)

    # ==========================================================================
    # Estimator fabric
    # ==========================================================================
    def make_estimator() -> Estimator:
        evaluation_examples: Set[Tuple[np.uint16, np.uint16]] = set()
        x: np.uint16 | None = None

        ## Collect mapping examples
        while len(evaluation_examples) < m:
            packet = yield len(evaluation_examples) / m

            # interrupt streak by delimiter or reset indicator
            if packet is None or np.all(packet == 1):
                x = None
                continue

            # record example
            y = bool_array_to_uint16(packet)
            if x is not None:
                evaluation_examples.add((x, y))
            x = y

        ## Reconstruct message
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
    return floor_2n_m1_log2_2n_m1(packet_bitsize)


def estimate_packets_until_reconstructed(
    deletion_prob: float, packet_bitsize: int, message_bitsize: int
):
    return None
