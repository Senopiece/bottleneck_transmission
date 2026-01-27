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

        all_states = set(np.uint16(i) for i in range(q))

        # ----------------------------------------------------------------------
        # Precompute tails = nodes with indegree 0 in the functional graph of f
        # (i.e., values not attained as f(x) for any x).
        # This is sampler-side only and does not depend on the channel.
        # ----------------------------------------------------------------------
        tails = set(all_states)
        for x in range(q):
            y = f(np.uint16(x))
            if y in tails:
                tails.discard(y)

        visited: Set[np.uint16] = set()

        # pick a new start state, prioritizing unvisited tails
        def reset() -> np.uint16:
            tails_unvisited = tails - visited
            if tails_unvisited:
                return next(iter(tails_unvisited))

            remaining = all_states - visited
            if remaining:
                return next(iter(remaining))

            visited.clear()
            candidates = tails

            return next(iter(candidates)) if candidates else np.uint16(0)

        curr = reset()
        phase = True

        def phased_output(value: np.uint16):
            return uint16_to_bool_array(value + q if phase else value, N)

        while True:
            # yield, mark visited and compute next
            yield phased_output(curr)
            visited.add(curr)
            nxt = f(curr)

            # if next would repeat a visited state, break adjacency with a delimiter and jump
            phase = not phase
            if nxt in visited:
                yield phased_output(nxt)
                curr = reset()
            else:
                curr = nxt

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
