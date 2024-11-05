"""
This module contains classes for SimulatedObjects that don't adhere to mujoco's rules of physics. Instead they
receive their pose data from a motion capture system (in our case, most likely Optitrack).
"""

import mujoco
from typing import Optional
from xml.etree import ElementTree as ET
from abc import ABC

from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.mocap import mocap_source
from aiml_virtual.utils import utils_general
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import hooked_bumblebee

warning = utils_general.warning

class MocapObject(simulated_object.SimulatedObject, ABC):
    """
    Base class for objects in the simulation that receive their data from a motion capture system.
    The corresponding mujoco documentation can be found here: https://mujoco.readthedocs.io/en/stable/modeling.html#cmocap
    Each MocapObject has to have exactly one Mocap Source associated with it: this is where the object will get its pose
    info.
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return None

    def __init__(self, source: Optional[mocap_source.MocapSource]=None, mocap_name: Optional[str]=None):
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

    def update(self) -> None:
        """
        Overrides SimulatedObject.update. Checks the mocap source to update its pose. The mocap source updates its frame
        in a different thread, this function merely copies the data found there.
        """
        if self.source is not None:
            mocap_frame = self.source.data
            if self.mocap_name in mocap_frame:
                self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid] = mocap_frame[self.mocap_name]
        else:
            return
            warning(f"Obj {self.name} not in mocap")

    def bind_to_data(self, data: mujoco.MjData) -> None:
        """
        Overrides SimulatedObject.bind_to_data. Saves the reference of the mujoco data.

        Args:
            data (mujoco.MjData): The data of the simulation run.
        """
        if self.model is None:
            raise RuntimeError
        self.data = data

class Airport(MocapObject):
    """
    Mocap object to display a piece of paper with the airport sign on it.
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["Airport"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(body, "geom", name=self.name, type="plane", pos="0 0 .05",
                      size="0.105 0.105 .05", material="mat-airport")
        return {"worldbody": [body]}

class ParkingLot(MocapObject):
    """
    Mocap object to display a piece of paper with the parking lot sign on it.
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["ParkingLot"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(body, "geom", name=self.name, type="plane", pos="0 0 .05",
                      size="0.105 0.105 .05", material="mat-parking_lot")
        return {"worldbody": [body]}

class LandingZone(MocapObject):
    """
    Mocap object to display the white pads to be put under crazyflies for landing.
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["LandingZone"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(body, "geom", {"class": "landing_zone"})
        return {"worldbody": [body]}

class Pole(MocapObject):
    """
    Mocap object to display the pool noodles we use as obstacles.
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["Pole", "obst"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        # note: cannot do ET.SubElement(body, "geom", class="pole_top"), because class is a keyword!
        ET.SubElement(body, "geom", {"class": "pole_top"})
        ET.SubElement(body, "geom", {"class": "pole_bottom1"})
        ET.SubElement(body, "geom", {"class": "pole_bottom2"})
        return {"worldbody": [body]}

class Hospital(MocapObject):
    """
    Mocap object to display the Hospital building (currently assumed to be bu1 in optitrack).
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["Hospital", "bu1"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(body, "geom", name=self.name, type="box", pos="0 0 0.445", size="0.1275 0.13 0.445",
                      material="mat-hospital")
        return {"worldbody": [body]}

class PostOffice(MocapObject):
    """
    Mocap object to display the Post Office building (currently assumed to be bu2 in optitrack).
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["bu2", "PostOffice"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(body, "geom", name=self.name, type="box", pos="0 0 0.205", size="0.1275 0.1275 0.205",
                      material="mat-post_office")
        return {"worldbody": [body]}

class Sztaki(MocapObject):
    """
    Mocap object to display the Sztaki building (currently assumed to be bu3 in optitrack).
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["Sztaki", "bu3"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(body, "geom", name=self.name, type="box", pos="0 0 0.0925", size="0.105 0.105 0.0925",
                      rgba="0.8 0.8 0.8 1.0", material="mat-sztaki")
        return {"worldbody": [body]}

class MocapPayload(MocapObject):
    """
    Mocap object to display a tear shaped Payload.
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["payload", "MocapPayload"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        black = "0 0 0 1"
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(body, "geom", name=self.name, type="mesh", mesh="payload_simplified", pos="0 0 0.0405",
                      rgba=black, euler="1.57 0 0")
        ET.SubElement(body, "geom", type="capsule", pos="0 0 0.075", size="0.004 0.027", rgba=black)
        ET.SubElement(body, "geom", type="capsule", pos="0 0.01173 0.10565", euler="-1.12200 0 0",
                      size="0.004 0.01562", rgba=black)
        ET.SubElement(body, "geom", type="capsule", pos="0 0.01061 0.10439", euler="-1.17810 0 0",
                      size="0.004 0.01378", rgba=black)
        ET.SubElement(body, "geom", type="capsule", pos="0 0.02561 0.11939", euler="-0.39270 0 0",
                      size="0.004 0.01378", rgba=black)
        ET.SubElement(body, "geom", type="capsule", pos="0 0.02561 0.14061", euler="0.39270 0 0",
                      size="0.004 0.01378", rgba=black)
        ET.SubElement(body, "geom", type="capsule", pos="0 0.01061 0.15561", euler="1.17810 0 0",
                      size="0.004 0.01378", rgba=black)
        ET.SubElement(body, "geom", type="capsule", pos="0 -0.01061 0.15561", euler="1.96350 0 0",
                      size="0.004 0.01378", rgba=black)
        ET.SubElement(body, "geom", type="capsule", pos="0 -0.02561 0.14061", euler="2.74889 0 0",
                      size="0.004 0.008", rgba=black)
        return {"worldbody": [body]}

class MocapHook(MocapObject):
    """
    Mocap object to display a drone hook.
    """

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["hook"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        L = hooked_bumblebee.HookedBumblebee1DOF.ROD_LENGTH
        # The hook structure consists of the rod (which is a geom in the xml), and the head of the hook (which is a body
        # in the xml, although it is welded to the parent body). The head of the hook in turn consists of other geoms.
        # The rationale is that although we could have the hook structure be only geoms, it's simpler to define the
        # geoms of the hook's 'head' in relation to the end of the rod
        hook = ET.Element("body", name=self.name, pos="0.0085 0 0", mocap="true")  # parent body of rod+hook_head
        ET.SubElement(hook, "geom", name=f"{self.name}_rod", type="cylinder", fromto=f"0 0 0  0 0 -{L}",
                      size="0.005", mass="0.0")
        # this is the body for the "head" of the hook, which is welded to the parent body
        hook_head = ET.SubElement(hook, "body", name=f"{self.name}_head", pos=f"0 0 -{L}", euler="0 3.141592 0")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0 0.02", size="0.003 0.003 0.02")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0.019 0.054", euler="-0.92 0 0", size="0.003 0.003 0.026")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0.02 0.0825", euler="0.92 0 0", size="0.003 0.003 0.026")
        ET.SubElement(hook_head, "geom", type="box", pos="0 -0.018 0.085", euler="-1.0472 0 0",
                      size="0.003 0.003 0.026")
        return {"worldbody": [hook]}