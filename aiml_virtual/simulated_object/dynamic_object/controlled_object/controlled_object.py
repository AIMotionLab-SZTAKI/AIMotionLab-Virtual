"""
This module contains the base class for controlled SimulatedObjects.
"""

from typing import Optional
from abc import ABC, abstractmethod

from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.controller import controller
from aiml_virtual.trajectory import trajectory


class ControlledObject(dynamic_object.DynamicObject, ABC):
    """
    Base class for objects in the simulation that have a controller implemented in python, and therefore need a python
    representation (to interact with the controller).
    """

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return None

    def __init__(self):
        super().__init__()
        self.controller: Optional[controller.Controller] = None  #: The controller in use currently.
        self.trajectory: Optional[trajectory.Trajectory] = None  #: The trajectory to follow.

    @abstractmethod
    def set_default_controller(self) -> None:
        """
        For ease, all controlled objects must have a default controller type. This method instantiates it, and sets it
        for use.
        """
        pass

