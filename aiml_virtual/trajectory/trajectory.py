"""
This module contains the abstract Trajectory class, which serves as a base of all trajectory types.
"""

from abc import ABC, abstractmethod
from typing import Any


class Trajectory(ABC):  # move this to a separate file, and make it abstract base
    """
    Base class for all trajectories.

    Trajectories may have different outputs, but they will always be organized in a dictionary, to be able to look up
    by velocity, position, etc.
    """
    def __init__(self):
        """
        Constructor left empty due to great variance in actual implementations.
        """
        pass

    @abstractmethod
    def evaluate(self, time: float) -> dict[str, Any]:
        """
        Concrete subclasses must implement this method. Calculates the desired setpoint at a given time.

        Args:
            time (float): The timestamp at which to evaluate the trajectory.

        Returns:
            dict[str, Any]: The desired setpoint at the provided timestamp.
        """
        pass

