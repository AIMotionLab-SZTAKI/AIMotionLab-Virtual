from xml.etree import ElementTree as ET
import mujoco
from typing import Optional
from abc import ABC
from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.mocap import mocap_source
from aiml_virtual.utils import utils_general
warning = utils_general.warning

class MocapSkeleton(simulated_object.SimulatedObject, ABC):
    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return None

    def __init__(self, source: Optional[mocap_source.MocapSource] = None):
        super().__init__()
        self.source: mocap_source.MocapSource = source
        self.mocap_objects: list[mocap_object.MocapObject] = []

    def bind_to_data(self, data: mujoco.MjData) -> None:
        if self.model is None:
            raise RuntimeError
        else:
            for o in self.mocap_objects:
                o.bind_to_data(data)
            self.data = data

    def bind_to_model(self, model: mujoco.MjModel) -> None:
        for o in self.mocap_objects:
            o.bind_to_model(model)
        self.model = model

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        merged_dict: dict[str, list[ET.Element]] = {}
        xml_dicts = [obj.create_xml_element(pos, quat, color) for obj in self.mocap_objects]
        for xml_dict in xml_dicts:
            for key, value in xml_dict.items():
                merged_dict[key] = merged_dict.get(key, []) + value
        return merged_dict

