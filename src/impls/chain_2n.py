from typing import Set, Tuple

import random
import numpy as np

from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.fields.gf2n import (
    evaluate_poly_falling_factorial,
    interpolate_poly_falling_factorial,
    first_points_to_falling_factorial_coeffs,
    evaluate_poly_falling_factorial_first_points,
    make_feld,
)
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

    m = min_m_such_that_2n_minus_1_pow_k_ge_2p(
        message_bitsize,
        packet_bitsize,
        (1 << packet_bitsize) - 1,
    )

    N = packet_bitsize
    n, mask, red = make_feld(N)

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # Message vector is evaluation at [0, m-1] points
        # It is crucial to define it as evaluations at some points
        # to make sure there are at least m points not touching the 2^n - 1 value
        # - that is the criterion to make polynomial recoverable
        message_vector = make_message_vector(message, N, m)  # shape (m,)

        # Convert it to falling factorial coeffs
        coeffs = first_points_to_falling_factorial_coeffs(message_vector, n, mask, red)

        # Yield samples
        Nstates = 1 << n
        tails = set(np.uint16(i) for i in range(Nstates - 1))
        for x in range(Nstates - 1):
            y = evaluate_poly_falling_factorial(np.uint16(x), coeffs, n, mask, red)
            tails.discard(y)

        visited = {mask}  # use mask as the delimiter - that is 2^n - 1 aka all ones
        all_states = set(np.uint16(i) for i in range(Nstates))
        prev = np.uint16(mask)
        curr = np.uint16(0)

        def reset():
            nonlocal prev, curr
            prev = np.uint16(mask)
            tails_unvisited = tails - visited
            remaining = all_states - visited
            curr = list(tails_unvisited)[0] if tails_unvisited else list(remaining)[0]

        reset()

        while True:
            if curr in visited:
                if curr != mask and prev != mask:
                    yield uint16_to_bool_array(curr, N)

                yield np.ones(n, dtype=np.bool_)

                if len(visited) == Nstates:
                    visited = {mask}

                reset()

            yield uint16_to_bool_array(curr, N)
            visited.add(curr)
            prev = curr
            curr = evaluate_poly_falling_factorial(curr, coeffs, n, mask, red)

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

        coeffs = interpolate_poly_falling_factorial(outputs, inputs, n, mask, red)
        message_vector = evaluate_poly_falling_factorial_first_points(
            coeffs, n, mask, red
        )
        message = message_from_message_vector(message_vector, N, message_bitsize)

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
