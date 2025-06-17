"""
Module that contains bumblebees with hooks attached.
"""

from typing import Optional
from xml.etree import ElementTree as ET

import mujoco

from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import bumblebee

class HookedBumblebee1DOF(bumblebee.Bumblebee):
    """
    Class that extends the Bumblebee to have a 1-DOF hook on it.
    """
    ROD_LENGTH = 0.4

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        L = HookedBumblebee1DOF.ROD_LENGTH
        # grab the parent's xml to augment with our own
        bumblebee_xml = super().create_xml_element(pos, quat, color)
        drone_body = bumblebee_xml["worldbody"][0]
        actuators = bumblebee_xml["actuator"]
        sensors = bumblebee_xml["sensor"]
        hook_name = f"{self.name}_hook"
        site_name = self.name + "_hook_base"
        # The hook structure consists of the rod (which is a geom in the xml), and the head of the hook (which is a body
        # in the xml, although it is welded to the parent body). The head of the hook in turn consists of other geoms.
        # The rationale is that although we could have the hook structure be only geoms, it's simpler to define the
        # geoms of the hook's 'head' in relation to the end of the rod
        hook = ET.SubElement(drone_body, "body", name=hook_name, pos="0.0085 0 0")  # parent body of rod+hook_head
        # this site is at the end of the rod, which is the same as the base of the head of the hook
        ET.SubElement(hook, "site", name=site_name, pos=f"0 0 {L}", type="sphere", size="0.002")
        # this is the 1DOF joint that allows movement of the hook relative to the drone
        ET.SubElement(hook, "joint", name=hook_name, axis="0 1 0", pos="0 0 0", damping="0.001")
        ET.SubElement(hook, "geom", name=f"{self.name}_rod", type="cylinder", fromto=f"0 0 0  0 0 -{L}",
                      size="0.005", mass="0.0")
        # this is the body for the "head" of the hook, which is welded to the parent body
        hook_head = ET.SubElement(hook, "body", name=f"{hook_name}_head", pos=f"0 0 -{L}", euler="0 3.141592 0")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0 0.02", size="0.003 0.003 0.02", mass="0.02")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0.019 0.054", euler="-0.92 0 0", size="0.003 0.003 0.026",
                      mass="0.0001")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0.02 0.0825", euler="0.92 0 0", size="0.003 0.003 0.026",
                      mass="0.0001")
        ET.SubElement(hook_head, "geom", type="box", pos="0 -0.018 0.085", euler="-1.0472 0 0", size="0.003 0.003 0.026",
                      mass="0.0001")

        #sensors.append(ET.Element("jointpos", joint=hook_name, name=f"{self.name}_joint_pos"))
        #sensors.append(ET.Element("jointvel", joint=hook_name, name=f"{self.name}_joint_vel"))
        sensors.append(ET.Element("framepos", objtype="site", objname=site_name, name=f"{self.name}_hook_pos"))
        sensors.append(ET.Element("framelinvel", objtype="site", objname=site_name, name=f"{self.name}_hook_vel"))
        sensors.append(ET.Element("framequat", objtype="site", objname=site_name, name=f"{self.name}_hook_quat"))
        sensors.append(ET.Element("frameangvel", objtype="site", objname=site_name, name=f"{self.name}_hook_ang_vel"))
        return bumblebee_xml

    def bind_to_data(self, data: mujoco.MjData) -> None:
        super().bind_to_data(data)
        #self.sensors["hook_joint_pos"] = self.data.sensor(f"{self.name}_joint_pos")
        #self.sensors["hook_joint_vel"] = self.data.sensor(f"{self.name}_joint_vel")
        self.sensors["hook_pos"] = self.data.sensor(f"{self.name}_hook_pos")
        self.sensors["hook_vel"] = self.data.sensor(f"{self.name}_hook_vel")
        self.sensors["hook_quat"] = self.data.sensor(f"{self.name}_hook_quat")
        self.sensors["hook_ang_vel"] = self.data.sensor(f"{self.name}_hook_ang_vel")

class HookedBumblebee2DOF(bumblebee.Bumblebee):
    """
    Class that extends the Bumblebee to have a 2-DOF hook on it.
    """
    ROD_LENGTH = 0.4

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        L = HookedBumblebee1DOF.ROD_LENGTH
        # grab the parent's xml to augment with our own
        bumblebee_xml = super().create_xml_element(pos, quat, color)
        drone_body = bumblebee_xml["worldbody"][0]
        actuators = bumblebee_xml["actuator"]
        sensors = bumblebee_xml["sensor"]
        hook_name = f"{self.name}_hook"
        site_name = self.name + "_hook_base"
        # The hook structure consists of the rod (which is a geom in the xml), and the head of the hook (which is a body
        # in the xml, although it is welded to the parent body). The head of the hook in turn consists of other geoms.
        # The rationale is that although we could have the hook structure be only geoms, it's simpler to define the
        # geoms of the hook's 'head' in relation to the end of the rod
        hook = ET.SubElement(drone_body, "body", name=hook_name, pos="0.0085 0 0")  # parent body of rod+hook_head
        # this site is at the end of the rod, which is the same as the base of the head of the hook
        ET.SubElement(hook, "site", name=site_name, pos=f"0 0 {L}", type="sphere", size="0.002")
        # In order to make a 2DOF joint, we just stack two 1DOF joints, as opposed to a ball joint, which would allow
        # rotation.
        ET.SubElement(hook, "joint", name=f"{hook_name}_y", axis="0 1 0", pos="0 0 0", damping="0.001")
        ET.SubElement(hook, "joint", name=f"{hook_name}_x", axis="1 0 0", pos="0 0 0", damping="0.001")
        ET.SubElement(hook, "geom", name=f"{self.name}_rod", type="cylinder", fromto=f"0 0 0  0 0 -{L}",
                      size="0.005", mass="0.0")
        # this is the body for the "head" of the hook, which is welded to the parent body
        hook_head = ET.SubElement(hook, "body", name=f"{hook_name}_head", pos=f"0 0 -{L}", euler="0 3.141592 0")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0 0.02", size="0.003 0.003 0.02", mass="0.02")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0.019 0.054", euler="-0.92 0 0", size="0.003 0.003 0.026",
                      mass="0.0001")
        ET.SubElement(hook_head, "geom", type="box", pos="0 0.02 0.0825", euler="0.92 0 0", size="0.003 0.003 0.026",
                      mass="0.0001")
        ET.SubElement(hook_head, "geom", type="box", pos="0 -0.018 0.085", euler="-1.0472 0 0", size="0.003 0.003 0.026",
                      mass="0.0001")

        sensors.append(ET.Element("framepos", objtype="site", objname=site_name, name=f"{self.name}_hook_pos"))
        sensors.append(ET.Element("framelinvel", objtype="site", objname=site_name, name=f"{self.name}_hook_vel"))
        sensors.append(ET.Element("framequat", objtype="site", objname=site_name, name=f"{self.name}_hook_quat"))
        sensors.append(ET.Element("frameangvel", objtype="site", objname=site_name, name=f"{self.name}_hook_ang_vel"))
        return bumblebee_xml

    def bind_to_data(self, data: mujoco.MjData) -> None:
        super().bind_to_data(data)
        self.sensors["hook_pos"] = self.data.sensor(f"{self.name}_hook_pos")
        self.sensors["hook_vel"] = self.data.sensor(f"{self.name}_hook_vel")
        self.sensors["hook_quat"] = self.data.sensor(f"{self.name}_hook_quat")
        self.sensors["hook_ang_vel"] = self.data.sensor(f"{self.name}_hook_ang_vel")
        self.sensors["qpos"] = self.data.qpos
        self.sensors["qvel"] = self.data.qvel
