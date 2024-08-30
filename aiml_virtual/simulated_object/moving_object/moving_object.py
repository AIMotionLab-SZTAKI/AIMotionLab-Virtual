"""
This module contains the base class for controlled SimulatedObjects.
"""

import xml.etree.ElementTree as ET
from abc import abstractmethod
import mujoco
from typing import Optional

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

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Overrides method in SimulatedObject to specify whether to check for aliases when parsing an XML.

        Returns:
            Optional[list[str]]: None, to opt out of parsing.
        """
        return None

