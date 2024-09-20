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
    The corresponding mujoco documentation can be found here: https://mujoco.readthedocs.io/en/stable/modeling.html#cmocap
    Each MocapObject has to have exactly one Mocap Source associated with it: this is where the object will get its pose
    info.
    """
    def __init__(self, source: mocap_source.MocapSource, mocap_name: Optional[str]=None):
        super().__init__()
        self.mocap_name: str = mocap_name if mocap_name is not None else self.name  #: The name in the mocap dictionary (it may differ from the mujoco model name).
        self.source: mocap_source.MocapSource = source  #: The source for the poses of the object.

    @property
    def mocapid(self) -> Optional[int]:
        """
        Property to look up the mocap id of the object in the mujoco model. This is different from the regular id of
        the model, made evident by the fact that only mocap objects will have it.
        """
        if self.model is not None:
            return self.model.body_mocapid[self.id]
        else:
            return None

    def update(self, time: float) -> None:
        """
        Overrides SimulatedObject.update. Checks the mocap source to update its pose.

        Args:
            time (float): The time elapsed in the simulation.

        .. note::
            This function is one of level of abstraction higher in MocapObject than in MovingObject. That is to say, in
            the case of MoingObject, update is left abstract and the subclasses must implement it. MovingObject type
            objects implement update in vastly different ways, however, in the case of MocapObjects, it is pretty much
            always the case that all we need to do is copy the data in the mocap source to the mujoco data. If the
            subclass needs additional operations, it may override this function (this is what MocapDrones do, which
            need to spin their propellers in addition to saving their mocap data).
        """
        mocap_frame = self.source.data
        if self.mocap_name in mocap_frame:
            self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid] = mocap_frame[self.mocap_name]
        else:
            pass
            warning(f"Obj {self.name} not in mocap")

    def bind_to_data(self, data: mujoco.MjData) -> None:
        """
        Overrides SimulatedObject.bind_to_data. Saves the reference of the mujoco data.

        .. note::
            This function (like update) is also one level of abstraction higher in MocapObject than in MovingObject,
            since MocapObject type objects are more similar than MovingObjects.

        Args:
            data (mujoco.MjData): The data of the simulation run.
        """
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

