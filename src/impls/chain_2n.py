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

    m = min_m_such_that_2n_minus_1_pow_k_ge_2p(
        message_bitsize,
        packet_bitsize,
        (1 << packet_bitsize) - 1,
    )

    N = packet_bitsize
    n, mask, red = gf2n.make_field(N)

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
        coeffs = gf2n.first_points_to_falling_factorial_coeffs(
            message_vector, n, mask, red
        )

        def f(x: np.uint16) -> np.uint16:
            return gf2n.evaluate_poly_falling_factorial(
                np.uint16(x), coeffs, n, mask, red
            )

        q = 1 << n
        all_states = set(np.uint16(i) for i in range(q))

        # ----------------------------------------------------------------------
        # Precompute tails = nodes with indegree 0 in the functional graph of f
        # (i.e., values not attained as f(x) for any x).
        # This is sampler-side only and does not depend on the channel.
        # ----------------------------------------------------------------------
        tails = set(np.uint16(i) for i in range(q - 1))
        for x in range(q - 1):
            y = f(np.uint16(x))
            tails.discard(y)

        delimiter = np.uint16((1 << N) - 1)  # all ones
        visited = {delimiter}

        # pick a new start state, prioritizing unvisited tails
        def reset():
            tails_unvisited = tails - visited
            if tails_unvisited:
                return next(iter(tails_unvisited))

            remaining = all_states - visited
            if remaining:
                return next(iter(remaining))

            visited.clear()
            visited.add(delimiter)
            tails_unvisited = tails - visited

            return next(iter(tails_unvisited)) if tails_unvisited else np.uint16(0)

        curr = reset()

        while True:
            # yield, mark visited and compute next
            yield uint16_to_bool_array(curr, N)
            visited.add(curr)
            nxt = f(curr)

            # if next would repeat a visited valid state, yield it then reset
            # if the next is delimiter - a force reset occurs, make sure not to yield delimiter twice
            if nxt in visited:
                if nxt != delimiter:
                    yield uint16_to_bool_array(nxt, N)
                yield np.ones(n, dtype=np.bool_)  # delimiter
                curr = reset()
            else:
                curr = nxt

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
