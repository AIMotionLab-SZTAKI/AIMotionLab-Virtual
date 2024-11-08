from xml.etree import ElementTree as ET
import mujoco
from typing import Optional, Type
from abc import ABC
from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.mocap import mocap_source
from aiml_virtual.simulated_object.mocap_object.mocap_object import MocapObject
from aiml_virtual.utils import utils_general
warning = utils_general.warning

class MocapSkeleton(mocap_object.MocapObject, ABC):
    """
    Base class for objects that receive their pose data fro a motion capture system, and are made up of several
    individually rigid motion capture objects. For example, the hooked mocap drone has two individually rigid parts:
    the drone and the hook. Since mujoco mocap objects cannot have joints, these objects each have their own top-level
    rigid mocap body. Of these rigid bodies, one has to be the 'owner' of the rest, and operations which require a
    single name/body/etc will use the respective methods of that body. In the case of a hooked bumblebee, for example,
    the owner is the MocapBumblebee part. For example, the mocapid property returns the mocapid of the corresponding
    mocap body in the xml.
    Their update shall be constructed in a way to keep the constraints between the bodies valid.

    .. todo::
        Check if these can be read from XML. Also check if we event want to be able to read Mocap Bodies from XML.

    """
    configurations: dict[str, list[tuple[str, Type[MocapObject]]]]  #: The recognized combinations for a given subclass.

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return None

    def __init__(self, source: mocap_source.MocapSource, mocap_name: str):
        super().__init__(source, mocap_name)
        self.mocap_objects: list[mocap_object.MocapObject] = []  # is made up of several mocap objects
        for part_name, cls in self.configurations[mocap_name]:
            self.mocap_objects.append(cls(source, part_name))

    @property
    def mocapid(self) -> Optional[int]:
        if self.model is not None and len(self.mocap_objects) > 0:
            return self.mocap_objects[0].mocapid
        else:
            return None

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

