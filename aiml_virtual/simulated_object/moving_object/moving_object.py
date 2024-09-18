"""
This module contains the base class for controlled SimulatedObjects.
"""

from typing import Optional, Union
import numpy as np

from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.controller import controller
from aiml_virtual.trajectory import trajectory


class MovingObject(simulated_object.SimulatedObject):
    """
    Base class for objects i the simulation that have a controller implemented in python, and therefore need a python
    representation (to interact with the controller).
    """
    def __init__(self):
        super().__init__()
        self.controllers: list[controller.Controller] = []  # storage for containers to switch between
        self.controller: Optional[controller.Controller] = None
        self.trajectory: Optional[trajectory.Trajectory] = None

    @property
    def mass(self) -> Union[None, float, np.array]:
        """
        Property to look up the mass of the drone in the mjModel.
        """
        if self.model:
            return self.model.body(self.name).mass
        else:
            return None

    @property
    def inertia(self) -> Union[None, np.ndarray]:
        """
        Property to look up the diagonal inertia of the drone in the mjModel.
        """
        if self.model:
            return self.model.body(self.name).inertia
        else:
            return None

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Overrides method in SimulatedObject to specify whether to check for aliases when parsing an XML.

        Returns:
            Optional[list[str]]: None, to opt out of parsing.
        """
        return None

