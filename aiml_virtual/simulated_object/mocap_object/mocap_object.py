"""
This module contains the base class for SimulatedObjects that don't have a controller. Instead they receive their pose
data from a motion capture system (in our case, most likely Optitrack).
"""

import xml.etree.ElementTree as ET
from abc import abstractmethod
import mujoco
from typing import Optional

from aiml_virtual.simulated_object import simulated_object


class MocapObject(simulated_object.SimulatedObject):
    """
    Base class for objects in the simulation that don't have a controller, but still need to be represented by a python
    class.
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Overrides method in SimulatedObject to specify whether to check for aliases when parsing an XML.

        Returns:
            Optional[list[str]]: None, to opt out of parsing.
        """
        return None
