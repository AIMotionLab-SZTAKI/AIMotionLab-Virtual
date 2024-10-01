"""
This module contains the base class for controlled SimulatedObjects.
"""

from typing import Optional
from abc import ABC

from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.controller import controller
from aiml_virtual.trajectory import trajectory


class ControlledObject(dynamic_object.DynamicObject, ABC):
    """
    Base class for objects in the simulation that have a controller implemented in python, and therefore need a python
    representation (to interact with the controller).
    """
    def __init__(self):
        super().__init__()
        self.controllers: list[controller.Controller] = []  # storage for containers to switch between
        self.controller: Optional[controller.Controller] = None
        self.trajectory: Optional[trajectory.Trajectory] = None

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return None
