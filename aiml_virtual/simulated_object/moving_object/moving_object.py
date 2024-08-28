import xml.etree.ElementTree as ET
from abc import abstractmethod
import mujoco
from typing import Optional

from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.controller import controller
from aiml_virtual.trajectory import trajectory


class MovingObject(simulated_object.SimulatedObject):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self):
        super().__init__()
        self.controllers: list[controller.Controller] = []  # storage for containers to switch between
        self.controller: Optional[controller.Controller] = None
        self.trajectory: Optional[trajectory.Trajectory] = None

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        # returning None opts out of XML parsing
        return None

    @abstractmethod
    def bind_to_data(self, data: mujoco.MjData):
        pass

    @abstractmethod
    def update(self, mj_step_count: int, step: float) -> None:
        pass

    @abstractmethod
    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        pass

