from typing import Callable, Generator, Iterator, NamedTuple
from numpy.typing import NDArray
import numpy as np


BitArray = NDArray[np.bool_]

Packet = BitArray
Message = BitArray

Deletion = None  # send a special delimiter when deletion occurs

Sampler = Iterator[Packet]  # yields Packets
Estimator = Generator[
    float, Packet | Deletion, Message
]  # receives Packets, yields % done [0-1], returns reconstructed Message


SamplerFactory = Callable[[Message], Sampler]
EstimatorFactory = Callable[[], Estimator]


class Config(NamedTuple):
    packet_bitsize: int
    message_bitsize: int


class Protocol(NamedTuple):
    make_sampler: SamplerFactory
    make_estimator: EstimatorFactory


ProtocolFactory = Callable[[Config], Protocol]

# Each impl is implementing:
# max_message_bitsize : int -> int # maximum message size for a given packet size (all in bits)
# create_protocol : ProtocolFactory
