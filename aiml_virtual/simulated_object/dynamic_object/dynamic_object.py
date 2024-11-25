"""
This module contains classes for SimulatedObjects that adhere to mujoco's rules of physics (gravity, etc.).
"""

from typing import Optional, Union
from xml.etree import ElementTree as ET
from abc import ABC
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from aiml_virtual.simulated_object import simulated_object


class DynamicObject(simulated_object.SimulatedObject, ABC):
    """
    Base class for objects that follow the rules of physics. This includes simple objects without actuators such as a
    dynamic payload (as opposed to a mocap payload), as well as actuated objects such as a drone.
    """
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        """
        Overrides method in MocapObject to specify whether to check for aliases when parsing an XML. A None returns
        signals that this class opts out of parsing. This usually also means that it's an abstract class (ABC).

        .. todo::
            Will need to reconcile identifiers in passive mocap objects vs passive moving objects.

        Returns:
            Optional[str]: The alias for objects belonging to this class.
        """

        return None

    @property
    def mass(self) -> Union[None, float, np.array]:
        """
        Property to look up the mass of the object in the mjModel.
        """
        if self.model:
            return self.model.body(self.name).mass
        else:
            return None

    @property
    def inertia(self) -> Union[None, np.ndarray]:
        """
        Property to look up the diagonal inertia of the object in the mjModel.
        """
        if self.model:
            return self.model.body(self.name).inertia
        else:
            return None

    def bind_to_data(self, data: mujoco.MjData) -> None:
        self.data = data

class DynamicPayload(DynamicObject):
    """
    Class for handling a dynamic payload that is subject to physics. Not to be confused with a mocap payload, which is
    what we use to track a payload in optitrack.
    """
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "DynamicPayload"

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        load_mass = "0.07"  # I'm pretty sure it's something like 70g
        segment_mass = "0.0001"
        black = "0 0 0 1"
        body = ET.Element("body", name=self.name, pos=pos, quat=quat)
        ET.SubElement(body, "joint", name=self.name, type="free")
        ET.SubElement(body, "geom", name=self.name, type="mesh", mesh="payload_simplified", pos="0 0 0.0405",
                      rgba="0 0 0 1", euler="1.57 0 0", mass=segment_mass)
        ET.SubElement(body, "geom", name=f"{self.name}_bottom", type="box", size=".016 .016 .02",
                      pos="0 0 0.0175", mass=load_mass, rgba="1.0 1.0 1.0 0.0")
        ET.SubElement(body, "geom", type="capsule", pos="0 0 0.075", size="0.004 0.027", rgba=black,
                      mass=segment_mass)
        ET.SubElement(body, "geom", type="capsule", pos="0.01173 0 0.10565", euler="0 1.12200 0",
                      size="0.004 0.01562", rgba=black, mass=segment_mass)
        ET.SubElement(body, "geom", type="capsule", pos="0.01061 0 0.10439", euler="0 1.17810 0",
                      size="0.004 0.01378", rgba=black, mass=segment_mass)
        ET.SubElement(body, "geom", type="capsule", pos="0.02561 0 0.11939", euler="0 0.39270 0",
                      size="0.004 0.01378", rgba=black, mass=segment_mass)
        ET.SubElement(body, "geom", type="capsule", pos="-0.02561 0 0.14061", euler="0 0.39270 0",
                      size="0.004 0.005", rgba=black, mass=segment_mass)
        ET.SubElement(body, "geom", type="capsule", pos="-0.01061 0 0.15561", euler="0 1.17810 0",
                      size="0.004 0.01378", rgba=black, mass=segment_mass)
        ET.SubElement(body, "geom", type="capsule", pos="0.01061 0 0.15561", euler="0 1.96350 0",
                      size="0.004 0.01378", rgba=black, mass=segment_mass)
        ET.SubElement(body, "geom", type="capsule", pos="0.02561 0 0.14061", euler="0 2.74889 0",
                      size="0.004 0.008", rgba=black, mass=segment_mass)
        ET.SubElement(body, "site", name="load_contact_point", pos="0 0 0.16", type="sphere", size="0.002", rgba=black)
        ET.SubElement(body, "site", name="load_hook_center_point", pos="0 0 0.1275", type="sphere", size="0.001", rgba=black)
        return {"worldbody": [body]}

    def update(self) -> None:
        pass