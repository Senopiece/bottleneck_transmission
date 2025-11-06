"""Simple impl: send a zero packet then transmit the binary representation of `index`.

Protocol:
 - First packet: all zeros (marker).
 - Next packets: chunks of up to N bits each (LSB-first within the whole bitstream).
 - After all bits are sent, producer returns zeros indefinitely.

Recoverer listens for a zero packet to start a fresh message, then collects bits
from subsequent non-zero packets until it has enough bits to reconstruct the
original integer (uses ceil(log2(D)) bits, minimum 1).
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


class Producer:
    def __init__(self, index: int, n: int, d: int):
        """index: integer in [0, d-1]; n: packet length; d: domain size.
        Prepare a sequence: zero marker + bit-chunks (LSB-first)."""
        self.n = n
        self.d = d
        self.index = int(index)

        # number of bits to represent values in [0, d-1]; at least 1 bit
        self.bits_needed = max(1, math.ceil(math.log2(d)) if d > 1 else 1)

        # produce bit list (LSB first)
        bits: List[int] = [(self.index >> i) & 1 for i in range(self.bits_needed)]

        # pack bits into packets of length n
        self.packets: List[np.ndarray] = []
        # zero marker first
        self.packets.append(np.zeros(self.n, dtype=np.uint8))

        for i in range(0, len(bits), self.n):
            chunk = bits[i : i + self.n]
            # pad chunk to length n with zeros
            if len(chunk) < self.n:
                chunk = chunk + [0] * (self.n - len(chunk))
            arr = np.array(chunk, dtype=np.uint8)
            self.packets.append(arr)

        # after consuming packets, stay idle returning zeros
        self._pos = 0

    def generate(self) -> np.ndarray:
        if self._pos < len(self.packets):
            out = self.packets[self._pos]
            self._pos += 1
            return out.copy()
        # idle zeros afterwards
        return np.zeros(self.n, dtype=np.uint8)


class Recoverer:
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self.bits_needed = max(1, math.ceil(math.log2(d)) if d > 1 else 1)
        self.collecting = False
        self.received_bits: List[int] = []
        self.recovered_value: int | None = None

    def feed(self, data: np.ndarray | None) -> int | None:
        # None acts as an interrupt / reset
        if data is None:
            self.collecting = False
            self.received_bits = []
            return self.recovered_value

        # Zero packet marks the start of a new message
        if np.all(data == 0):
            # Start fresh collection
            self.collecting = True
            self.received_bits = []
            self.recovered_value = None
            return None

        # treat the packet as bits (0/1) in order 0..n-1 (matching producer packing)
        bits = [int(b) & 1 for b in data.tolist()]

        if self.collecting:
            self.received_bits.extend(bits)

            if len(self.received_bits) >= self.bits_needed:
                # reconstruct using LSB-first ordering
                value = 0
                for i in range(self.bits_needed):
                    value |= (self.received_bits[i] & 1) << i

                self.recovered_value = int(value)
                # finished; stop collecting until next zero marker
                self.collecting = False
                return self.recovered_value

        return None


def producer_constructor(index: int, n: int, d: int) -> Producer:
    return Producer(index, n, d)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    return Recoverer(n, d)
