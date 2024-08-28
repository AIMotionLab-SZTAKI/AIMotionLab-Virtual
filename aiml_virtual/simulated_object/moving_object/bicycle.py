import xml.etree.ElementTree as ET
import mujoco
from typing import Any, Optional

from aiml_virtual.simulated_object.moving_object import moving_object
from aiml_virtual.controller import bicycle_controller

BicycleController = bicycle_controller.BicycleController


class Bicycle(moving_object.MovingObject):

    def __init__(self):
        super().__init__()
        self.controller: BicycleController = BicycleController()
        self.actr: Any = None  # TODO: type
        self.ctrl: Any = None  # TODO: type
        self.sensor: Any = None  # TODO: type

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        # the identifiers to look for in the XML
        return ["Bicycle", "bicycle"]

    # todo: types
    def update(self, mj_step_count: int, step: float) -> None:
        if self.controller:
            self.ctrl[0] = self.controller.compute_control()

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        name = self.name
        site_name = f"{name}_cog"

        bike_frame = ET.Element("body", name=name, pos=pos, quat=quat)
        ET.SubElement(bike_frame, "inertial", pos="0 0 0", diaginertia=".01 .01 .01", mass="1.0")
        ET.SubElement(bike_frame, "joint", name=name, type="free")
        ET.SubElement(bike_frame, "site", name=site_name, pos="0 0 0")
        ET.SubElement(bike_frame, "geom", name=name + "_crossbar", type="box", size=".06 .015 .02", pos="0 0 0",
                      rgba=color)

        front_wheel_name = name + "_wheelf"
        wheelf = ET.SubElement(bike_frame, "body", name=front_wheel_name)
        ET.SubElement(wheelf, "joint", name=front_wheel_name, type="hinge", pos="0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")
        ET.SubElement(wheelf, "geom", name=front_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="0.1 0 0", euler="1.571 0 0", material="material_check")

        rear_wheel_name = name + "_wheelr"
        wheelr = ET.SubElement(bike_frame, "body", name=rear_wheel_name)
        ET.SubElement(wheelr, "joint", name=rear_wheel_name, type="hinge", pos="-0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")
        ET.SubElement(wheelr, "geom", name=rear_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="-0.1 0 0", euler="1.571 0 0", material="material_check")
        actuator = ET.Element("motor", name=name+"_actr", joint=rear_wheel_name)
        sensor = ET.Element("velocimeter", site=site_name, name=name+"_velocimeter")
        ret = {"worldbody": [bike_frame],
               "actuator": [actuator],
               "sensor": [sensor]}
        return ret

    def bind_to_data(self, data: mujoco.MjData):
        self.data = data
        self.actr = data.actuator(self.name + "_actr")
        self.ctrl = self.actr.ctrl
        self.sensor = data.sensor(self.name + "_velocimeter").data

