"""
This module contains the abstract Trajectory class, which serves as a base of all trajectory types.
"""

from abc import ABC, abstractmethod
from typing import Any


class Trajectory(ABC):
    """
    Base class for all trajectories.

    Trajectories may have different outputs, but they will always be organized in a dictionary, to be able to look up
    by velocity, position, etc.
    """
    def __init__(self):
        """
        Constructor left empty due to great variance in actual implementations.
        """
        super().__init__()

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, Any]:
        """
        Concrete subclasses must implement this method. Calculates the desired setpoint with given inputs. The signature
        is left flexible to allow multiple kinds of trajectories to map onto it.

        Returns:
            dict[str, Any]: The desired setpoint at the provided timestamp.
        """
        pass

