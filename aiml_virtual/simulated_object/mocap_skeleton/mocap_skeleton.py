from xml.etree import ElementTree as ET
import mujoco
from typing import Optional, Type, cast
from abc import ABC
import os
import json

from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.mocap.mocap_source import MocapSource
from aiml_virtual.simulated_object.mocap_object.mocap_object import MocapObject
from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.utils import utils_general
import aiml_virtual
warning = utils_general.warning


class MocapSkeleton(mocap_object.MocapObject, ABC):
    """
    Base class for objects that receive their pose data from a motion capture system, and are made up of several
    individually rigid motion capture objects. For example, the hooked mocap drone has two individually rigid parts:
    the drone and the hook. Since mujoco mocap objects cannot have joints, these objects each have their own top-level
    rigid mocap body. Of these rigid bodies, one has to be the 'owner' of the rest, and operations which require a
    single name/body/etc will use the respective methods of that body. In the case of a hooked bumblebee, for example,
    the owner is the MocapBumblebee part. For example, the mocapid property returns the mocapid of the corresponding
    mocap body in the xml.
    Their update shall be constructed in a way to keep the constraints between the bodies valid.
    """
    configurations: dict[str, list[tuple[str, Type[MocapObject]]]] = {} #: The recognized combinations for a given subclass.

    def __init_subclass__(cls, **kwargs):
        """
        MocapSkeleton subclasses will each have their own configuration, read from the config file. This method
        will run each time  anew MocapSkeleton subclass is defined, and read the config.
        """
        super().__init_subclass__(**kwargs)
        cls.configurations = {}
        config_path = os.path.join(aiml_virtual.resource_directory, "mocap_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file '{config_path}' not found.")
        with open(config_path, "r") as file:
            all_configs = json.load(file)
        identifier = cls.get_identifier()
        if identifier not in all_configs or "configurations" not in all_configs[identifier]:
            raise KeyError(f"Configuration for '{identifier}.configurations' not found in config file.")
        configurations = all_configs[identifier]["configurations"]
        for key, value in configurations.items():
            # Value is a list, where each element is again a list, of length 2. For example:
            # "bb3": [["bb3", "MocapBumblebee"], ["hook12", "MocapHook"]]
            # value[0]=["bb3", "MocapBumblebee"]
            # value[1]=["hook12", "MocapHook"]
            # the first element is the optitrack name of an element in the mocap skeleton, the second is the identifier
            # of the class corresponding to the mocap object
            # in other words, the value above defines a mocap skeleton, whose parent mocap object is called "bb3" in
            # motive, and its first element (the parent) is a mocap bumblebee, the second element is a mocap hook
            lst = []
            for element in value:
                name = element[0]  # e.g. "hook12"
                obj_type = simulated_object.SimulatedObject.xml_registry[element[1]] # e.g. "MocapHook" -> MocapHook (as type)
                lst.append((name, cast(Type[MocapObject], obj_type)))
            cls.configurations[key] = lst

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return None

    def __init__(self, source: MocapSource, mocap_name: str):
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

