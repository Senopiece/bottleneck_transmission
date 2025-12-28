from typing import Callable, Generator, Iterator, NamedTuple
from numpy.typing import NDArray
import numpy as np


BitArray = NDArray[np.bool_]

Packet = BitArray
Payload = BitArray

Deletion = None  # send a special delimiter when deletion occurs

Sampler = Iterator[Packet]  # yields Packets
Estimator = Generator[
    float, Packet | Deletion, Payload
]  # receives Packets, yields % done [0-1], returns reconstructed Payload


SamplerFactory = Callable[[Payload], Sampler]
EstimatorFactory = Callable[[], Estimator]


class Config(NamedTuple):
    packet_bitsize: int
    payload_bitsize: int


class Protocol(NamedTuple):
    make_sampler: SamplerFactory
    make_estimator: EstimatorFactory


ProtocolFactory = Callable[[Config], Protocol]

# Each impl is implementing:
# max_payload_bitsize : int -> int # maximum payload size for a given packet size (all in bits)
# create_protocol : ProtocolFactory
