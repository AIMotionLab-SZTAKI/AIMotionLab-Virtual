import mujoco
from typing import Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod

from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.simulated_object.moving_object.drone.drone import Propeller
from aiml_virtual.simulated_object.mocap_object.mocap_source import mocap_source

class MocapDrone(mocap_object.MocapObject):
    def __init__(self, mocap: mocap_source.MocapSource):
        super().__init__(mocap)
        self.propellers: list[Propeller] = [Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE),
                                            Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE)]  #: List of propellers; first and third are ccw, second and fourth are cw

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return None
