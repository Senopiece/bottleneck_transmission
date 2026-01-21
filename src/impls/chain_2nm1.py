from typing import Set, Tuple

import numpy as np

from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.fields.gf2nm1 import (
    evaluate_poly,
    interpolate_poly,
    make_feld,
)
from ._interface import Config, Protocol, Message, Sampler, Estimator
from ._utils.intmath import (
    ispowprime_1_15,
    min_m_such_that_2n_minus_1_pow_k_ge_2p,
    floor_2n_m1_log2_2n_m1,
)

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
    m = min_m_such_that_2n_minus_1_pow_k_ge_2p(
        message_bitsize,
        packet_bitsize,
        (1 << packet_bitsize) - 1,
    )
    n, mask = make_feld(N)

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # Message vector is directly the polynomial coefficients
        message_vector = make_message_vector(message, N, m)  # shape (m,)

        # Yield samples
        Nstates = (1 << N) - 1  # GF(2^n - 1) states
        tails = set(np.uint16(i) for i in range(Nstates))
        for x in range(Nstates):
            y = evaluate_poly(np.uint16(x), message_vector, n, mask)
            tails.discard(y)

        visited: Set[np.uint16] = set()
        all_states = set(np.uint16(i) for i in range(Nstates))
        curr: np.uint16 = np.uint16(1)

        def reset():
            nonlocal curr
            tails_unvisited = tails - visited
            remaining = all_states - visited
            curr = list(tails_unvisited)[0] if tails_unvisited else list(remaining)[0]

        reset()

        while True:
            if curr in visited:
                yield uint16_to_bool_array(curr, N)
                yield np.ones(N, dtype=np.bool_)

                if len(visited) == Nstates:
                    visited.clear()

                reset()

            yield uint16_to_bool_array(curr, N)
            visited.add(curr)
            curr = evaluate_poly(curr, message_vector, n, mask)

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

        message_vector = interpolate_poly(outputs, inputs, n, mask)
        message = message_from_message_vector(message_vector, N, message_bitsize)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if ispowprime_1_15(packet_bitsize):
        return floor_2n_m1_log2_2n_m1(packet_bitsize)
    return 0
