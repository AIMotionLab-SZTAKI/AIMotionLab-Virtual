"""
This module contains the base class for SimulatedObjects that don't have a controller. Instead they receive their pose
data from a motion capture system (in our case, most likely Optitrack).
"""

import xml.etree.ElementTree as ET
from abc import abstractmethod
import mujoco
from typing import Optional

import numpy as np

from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.simulated_object.mocap_object.mocap_source import mocap_source

class MocapObject(simulated_object.SimulatedObject):
    """
    Base class for objects in the simulation that don't have a controller, but still need to be represented by a python
    class.
    TODO: incorporate this link into the docstring: https://mujoco.readthedocs.io/en/stable/modeling.html#cmocap
    """
    def __init__(self, mocap: mocap_source.MocapSource):
        super().__init__()
        self.mocap: mocap_source.MocapSource = mocap

    @property
    def mocapid(self) -> Optional[int]:
        if self.model is not None:
            _id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.name)
            return self.model.body_mocapid[_id]
        else:
            return None

    def update(self, time: float) -> None:
        # TODO: note that this is one level of abstraction higher than in moving object
        frame = self.mocap.get_mocap_by_name(self.name)
        # TODO: NEED SEPARATE MOCAP NAME AND MUJOCO NAME
        if frame is not None:
            self.data.mocap_pos[self.mocapid] = frame[0]
            self.data.mocap_quat[self.mocapid] = frame[1]
        else:
            print(f"Obj {self.name} not in mocap")

    def bind_to_data(self, data: mujoco.MjData) -> None:
        self.data = data

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Overrides method in SimulatedObject to specify whether to check for aliases when parsing an XML.

        Returns:
            Optional[list[str]]: None, to opt out of parsing.
        """
        return None

