# TODO: DOCSTRINGS AND COMMENTS

"""
This module contains the base class for SimulatedObjects that don't have a controller. Instead they receive their pose
data from a motion capture system (in our case, most likely Optitrack).
"""

import mujoco
from typing import Optional

from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.mocap import mocap_source
from aiml_virtual.utils import utils_general

warning = utils_general.warning

class MocapObject(simulated_object.SimulatedObject):
    """
    Base class for objects in the simulation that don't have a controller, but still need to be represented by a python
    class.
    TODO: incorporate this link into the docstring: https://mujoco.readthedocs.io/en/stable/modeling.html#cmocap
    """
    def __init__(self, source: mocap_source.MocapSource, mocap_name: Optional[str]=None):
        super().__init__()
        self.mocap_name = mocap_name if mocap_name is not None else self.name
        self.source: mocap_source.MocapSource = source

    @property
    def mocapid(self) -> Optional[int]:
        if self.model is not None:
            return self.model.body_mocapid[self.id]
        else:
            return None

    def update(self, time: float) -> None:
        # TODO: note that this is one level of abstraction higher than in moving object
        mocap_frame = self.source.data
        if self.mocap_name in mocap_frame:
            self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid] = mocap_frame[self.mocap_name]
        else:
            warning(f"Obj {self.name} not in mocap")

    def bind_to_data(self, data: mujoco.MjData) -> None:
        if self.model is None:
            raise RuntimeError
        self.data = data

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Overrides method in SimulatedObject to specify whether to check for aliases when parsing an XML.

        Returns:
            Optional[list[str]]: None, to opt out of parsing.
        """
        return None

