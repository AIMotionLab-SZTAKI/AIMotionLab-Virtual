"""
This module contains the implementation of a very basic ControlledObject: a bike with two wheels.
"""
import xml.etree.ElementTree as ET
import mujoco
from typing import Optional
import numpy as np

from aiml_virtual.simulated_object.dynamic_object.controlled_object import controlled_object
from aiml_virtual.controller import bicycle_controller

BicycleController = bicycle_controller.BicycleController


class Bicycle(controlled_object.ControlledObject):
    """
    A simple controlled object mostly used for testing. Consists of two wheels and an actuator, as well
    as a rectangular body.
    """

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "Bicycle"

    def __init__(self):
        super().__init__()
        self.controller: Optional[BicycleController] = None
        self.ctrl: Optional[np.ndarray] = None
        self.sensor: Optional[np.ndarray] = None

    def update(self) -> None:
        """
        Overrides method in SimulatedObject. Sets the control input (the torque of the actuated wheel).
        """
        if self.controller:
            self.ctrl[0] = self.controller.compute_control()

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        name = self.name
        site_name = f"{name}_cog"

        bike = ET.Element("body", name=name, pos=pos, quat=quat)  # the top level body
        ET.SubElement(bike, "inertial", pos="0 0 0", diaginertia=".01 .01 .01", mass="1.0")
        ET.SubElement(bike, "joint", name=name, type="free")  # the freejoint that allows the bike to move
        ET.SubElement(bike, "site", name=site_name, pos="0 0 0")  # center of gravity
        ET.SubElement(bike, "geom", name=name + "_crossbar", type="box", size=".06 .015 .02", pos="0 0 0",
                      rgba=color)  # the geom that would be the bike frame in an actual bike: connects the two wheels

        front_wheel_name = name + "_wheelf"
        front_wheel = ET.SubElement(bike, "body", name=front_wheel_name)
        ET.SubElement(front_wheel, "joint", name=front_wheel_name, type="hinge", pos="0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")  # front wheel axis
        ET.SubElement(front_wheel, "geom", name=front_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="0.1 0 0", euler="1.571 0 0", material="material_check")

        rear_wheel_name = name + "_wheelr"
        rear_wheel = ET.SubElement(bike, "body", name=rear_wheel_name)
        ET.SubElement(rear_wheel, "joint", name=rear_wheel_name, type="hinge", pos="-0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")  # rear wheel axis
        ET.SubElement(rear_wheel, "geom", name=rear_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="-0.1 0 0", euler="1.571 0 0", material="material_check")
        actuator = ET.Element("motor", name=name+"_actr", joint=rear_wheel_name)  # only rear wheel is actuated
        sensor = ET.Element("velocimeter", site=site_name, name=name+"_velocimeter")
        ret = {"worldbody": [bike],
               "actuator": [actuator],
               "sensor": [sensor]}
        return ret

    def bind_to_data(self, data: mujoco.MjData) -> None:
        """
        Overrides DynamicObject.bind_to_data. In addition to saving a reference to the MjData, it also saves a
        reference to its actuator and sensor.

        Args:
            data (mujoco.MjData): The data of the simulation (as opposed to the *model*).
        """
        if self.model is None:
            raise RuntimeError
        if self.controller is None:
            self.set_default_controller()
        self.data = data
        self.ctrl = data.actuator(self.name + "_actr").ctrl
        self.sensor = data.sensor(self.name + "_velocimeter").data

    def set_default_controller(self) -> None:
        self.controller: BicycleController = BicycleController()