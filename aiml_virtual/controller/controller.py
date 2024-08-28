"""
This module houses the Controller class, which serves as a base for all controller types.

Classes:
    Controller
"""

from abc import ABC, abstractmethod
from typing import Any


class Controller(ABC):
    """
    Base class for all controllers.

    Controllers may vary widely in actual implementation: they may produce different outputs (thrust+torque, or
    maybe rpy rate, etc.) and also take different inputs (e.g. one may take only position, velocity and acceleration
    into account, another one may also look at jerk). To this end, this base class is relatively plain, leaving the
    concrete implementation in the hands of the actual subclasses.
    """
    def __init__(self):
        """
        Constructor left empty due to great variance in actual implementations.
        """
        pass

    @abstractmethod
    def compute_control(self, *args, **kwargs) -> Any:
        """
        Concrete subclasses must implement this method: it's going to get called once per control loop.
        """
        pass
