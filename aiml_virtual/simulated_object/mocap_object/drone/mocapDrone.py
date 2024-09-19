# TODO: DOCSTRINGS AND COMMENTS

from typing import Optional
import mujoco

from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.simulated_object.moving_object.drone.drone import Propeller
from aiml_virtual.mocap import mocap_source


class MocapDrone(mocap_object.MocapObject):
    def __init__(self, source: mocap_source.MocapSource, mocap_name: Optional[str]=None):
        super().__init__(source, mocap_name)
        self.propellers: list[Propeller] = [Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE),
                                            Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE)]  #: List of propellers; first and third are ccw, second and fourth are cw

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return None

    def spin_propellers(self) -> None:
        """
        Updates the display of the propellers, to make it look like they are spinning.
        """
        if self.xpos[2] > 0.015:  # only start spinning if the drone has taken flight
            for propeller in self.propellers:
                propeller.spin()

    def update(self, time: float) -> None:
        super().update(time)
        self.spin_propellers()  # update how the propellers look

    def bind_to_data(self, data: mujoco.MjData) -> None:
        super().bind_to_data(data)
        for i, propeller in enumerate(self.propellers):
            prop_joint = self.data.joint(f"{self.name}_prop{i}")
            propeller.qpos = prop_joint.qpos
            propeller.angle = propeller.qpos[0]