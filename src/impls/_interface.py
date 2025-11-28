from abc import ABC, abstractmethod
import numpy as np


class Producer:
    @abstractmethod
    def generate(self) -> np.ndarray:
        """returns ndarray of shape (N,)"""


class GeneratorProducer(Producer):
    """
    Wrap any Python generator so it behaves like a Producer:
        producer.generate() -> next value of the generator
    """

    def __init__(self, generator):
        """
        Parameters
        ----------
        generator : generator or iterable
            A Python generator object or anything implementing __next__().
        """
        self._gen = generator

    def generate(self):
        """Return the next item from the wrapped generator."""
        return next(self._gen)


class Recoverer(ABC):
    @abstractmethod
    def feed(self, data: np.ndarray | None) -> int | None:
        """
        input: ndarray of shape (N,) or None for interrupt signal
        output: recovered data: number from 0 to D-1 or None
        """
