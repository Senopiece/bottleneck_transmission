from abc import ABC, abstractmethod
import numpy as np


class Producer:
    @abstractmethod
    def generate(self) -> np.ndarray:
        """returns ndarray of shape (N,)"""


class Recoverer(ABC):
    @abstractmethod
    def feed(self, data: np.ndarray | None) -> int | None:
        """
        input: ndarray of shape (N,) or None for interrupt signal
        output: recovered data: number from 0 to D-1 or None
        """
