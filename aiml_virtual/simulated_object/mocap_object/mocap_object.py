"""
This module contains the base class for SimulatedObjects that don't have a controller. Instead they receive their pose
data from a motion capture system (in our case, most likely Optitrack).
"""

import mujoco
from typing import Optional
from xml.etree import ElementTree as ET

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
    MocapObject is a concrete class: instances of it can be created. If the user creates  an instance of MocapObject
    directly (as opposed to a MocapDrone for example), then that object will have no  actuators/controllers/moving
    parts. It will be a passive mocap object. For example, one may use this class to keep track of buildings/obstacles
    in optitrack, but not mocap drones, as those have moving propellers.
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
            return
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

        .. todo::
            Will need to reconcile identifiers in passive mocap objects vs passive moving objects.

        Returns:
            Optional[list[str]]: None, to opt out of parsing.
        """
        return ["bu", "building", "Building", "obst", "obstacle", "Obstacle", "pole", "Pole", "parking_lot",
                "landing_zone", "airport", "Airport", "hospital", "Hospital", "post_office", "sztaki", "Sztaki",
                "payload", "Payload"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        """
        Overrides method in SimulatedObject. Generates all the necessary XML elements for the model. Generates a
        different XML based on the mocap name. E.g.: if the mocap name is "hospital" it will generate a hospital
        building, but if the mocap name is "bu1", it will generate a pole.

        Args:
            pos (str): The position of the object in the scene, x-y-z separated by spaces. E.g.: "0 1 -1"
            quat (str): The quaternion orientation of the object in the scene, w-x-y-z separated by spaces.
            color (str): The base color of the object in th scene, r-g-b-opacity separated by spaces, scaled 0.0  to 1.0

        Returns:
            dict[str, list[ET.Element]]: A dictionary where the keys are tags of XML elements in the MJCF file, and the
            values are lists of XML elements to be appended as children to those XML elements.
        """
        # note: need separate bodies and geoms, because the mocap parameter can only be given to a body, but
        # only a geom may be defined as a "plane". So there are a few single-geom bodies here.
        if "airport" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            ET.SubElement(body, "geom", name=self.name, type="plane", pos="0 0 .05",
                          size="0.105 0.105 .05", material="mat-airport")
            return {"worldbody": [body]}
        if "parking_lot" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            ET.SubElement(body, "geom", name=self.name, type="plane", pos="0 0 .05",
                          size="0.105 0.105 .05", material="mat-parking_lot")
            return {"worldbody": [body]}
        if "landing_zone" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            ET.SubElement(body, "geom", {"class" : "landing_zone"})
            return {"worldbody": [body]}
        if "pole" in self.mocap_name.lower() or "obst" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            # note: cannot do ET.SubElement(body, "geom", class="pole_top"), because class is a keyword!
            ET.SubElement(body, "geom", {"class": "pole_top"})
            ET.SubElement(body, "geom", {"class": "pole_bottom1"})
            ET.SubElement(body, "geom", {"class": "pole_bottom2"})
            return {"worldbody": [body]}
        if "hospital" in self.mocap_name.lower() or "bu1" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            ET.SubElement(body, "geom", name=self.name, type="box", pos="0 0 0.445", size="0.1275 0.13 0.445",
                          material="mat-hospital")
            return {"worldbody": [body]}
        if "post_office" in self.mocap_name.lower() or "bu2" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            ET.SubElement(body, "geom", name=self.name, type="box", pos="0 0 0.205", size="0.1275 0.1275 0.205",
                          material="mat-post_office")
            return {"worldbody": [body]}
        if "sztaki" in self.mocap_name.lower() or "bu3" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            ET.SubElement(body, "geom", name=self.name, type="box", pos="0 0 0.0925", size="0.105 0.105 0.0925",
                          rgba="0.8 0.8 0.8 1.0", material="mat-sztaki")
            return {"worldbody": [body]}
        if "payload" in self.mocap_name.lower():
            body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
            ET.SubElement(body, "geom", name=self.name, type="mesh", mesh="payload_simplified", pos="0 0 0.0405",
                          rgba="0 0 0 1", euler="1.57 0 0")
            ET.SubElement(body, "geom", type="capsule", pos="0 0 0.075", size="0.004 0.027", rgba="0 0 0 1")
            ET.SubElement(body, "geom", type="capsule", pos="0 0.01173 0.10565", euler="-1.12200 0 0",
                          size="0.004 0.01562", rgba="0 0 0 1")
            ET.SubElement(body, "geom", type="capsule", pos="0 0.01061 0.10439", euler="-1.17810 0 0",
                          size="0.004 0.01378", rgba="0 0 0 1")
            ET.SubElement(body, "geom", type="capsule", pos="0 0.02561 0.11939", euler="-0.39270 0 0",
                          size="0.004 0.01378", rgba="0 0 0 1")
            ET.SubElement(body, "geom", type="capsule", pos="0 0.02561 0.14061", euler="0.39270 0 0",
                          size="0.004 0.01378", rgba="0 0 0 1")
            ET.SubElement(body, "geom", type="capsule", pos="0 0.01061 0.15561", euler="1.17810 0 0",
                          size="0.004 0.01378", rgba="0 0 0 1")
            ET.SubElement(body, "geom", type="capsule", pos="0 -0.01061 0.15561", euler="1.96350 0 0",
                          size="0.004 0.01378", rgba="0 0 0 1")
            ET.SubElement(body, "geom", type="capsule", pos="0 -0.02561 0.14061", euler="2.74889 0 0",
                          size="0.004 0.008", rgba="0 0 0 1")
            return {"worldbody": [body]}

        raise RuntimeError(f"No such PassiveObject recognized: {self.mocap_name}")
