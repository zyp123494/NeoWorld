from abc import ABC, abstractmethod
from typing import *


class Sampler(ABC):
    """
    A base class for samplers.
    """

    @abstractmethod
    def sample(self, model, **kwargs):
        """
        Sample from a model.
        """
        pass
