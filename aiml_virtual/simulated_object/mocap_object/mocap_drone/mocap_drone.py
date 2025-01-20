"""
This module contains the class that encapsulates the common behaviour of mocap drones (such as MocapCrazyflie or
MocapBumblebee)
"""

from typing import Optional
import mujoco
from abc import ABC

from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.drone import Propeller
from aiml_virtual.mocap import mocap_source


class MocapDrone(mocap_object.MocapObject, ABC):
    """
    Class encapsulation behaviour common to all mocap drones (which are MocapObjects).
    """
    def __init__(self, source: Optional[mocap_source.MocapSource] = None, mocap_name: Optional[str] = None):
        super().__init__(source, mocap_name)
        self.propellers: list[Propeller] = [Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE),
                                            Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE)]  #: List of propellers; first and third are ccw, second and fourth are cw

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return None

    def spin_propellers(self) -> None:
        """
        Updates the display of the propellers, to make it look like they are spinning.
        """
        if self.xpos[2] > 0.15:  # only start spinning if the mocap_drone has taken flight
            for propeller in self.propellers:
                propeller.spin()

    def update(self) -> None:
        """
        Overrides MocapObject.update: spins propellers in addition to writing its pose data.
        """
        super().update()
        self.spin_propellers()  # update how the propellers look

    def bind_to_data(self, data: mujoco.MjData) -> None:
        """
        Overrides MocapObject.bind_to_data. In addition to saving a reference to the data (in the superclass'
        bind call), it also saves references to propeller data in order to be able to spin them visually.

        Args:
             data (mujoco.MjData): The data of the simulation (as opposed to the *model*).
        """
        super().bind_to_data(data)
        for i, propeller in enumerate(self.propellers):
            prop_joint = self.data.joint(f"{self.name}_prop{i}")
            propeller.qpos = prop_joint.qpos
            propeller.angle = propeller.qpos[0]