"""
This module contains the implementation of a very basic MovingObject: a bike with two wheels.
"""
import xml.etree.ElementTree as ET
import mujoco
from typing import Any, Optional

from aiml_virtual.simulated_object.moving_object import moving_object
from aiml_virtual.controller import bicycle_controller

BicycleController = bicycle_controller.BicycleController


class Bicycle(moving_object.MovingObject):
    """
    A simple controlled object mostly used for testing. Consists of two wheels and an actuator, as well
    as a rectangular body.
    """
    def __init__(self):
        super().__init__()
        self.controller: BicycleController = BicycleController()
        self.actr: Any = None  # TODO: type
        self.ctrl: Any = None  # TODO: type
        self.sensor: Any = None  # TODO: type

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Overrides method in MovingObject to specify whether to check for aliases when parsing an XML.

        Returns:
            Optional[list[str]]: The list of aliases for objects belonging to this class.
        """
        return ["Bicycle", "bicycle"]

    def update(self, time: float) -> None:
        """
        Overrides method in SimulatedObject. Sets the control input (the torque of the actuated wheel).

        Args:
            time (float): The elapsed time in the simulation.
        """
        if self.controller:
            self.ctrl[0] = self.controller.compute_control()

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        """
        Overrides method in SimulatedObject. Generates all the necessary XML elements for the model of a Bicycle.

        Args:
            pos (str): The position of the object in the scene, x-y-z separated by spaces. E.g.: "0 1 -1"
            quat (str): The quaternion orientation of the object in the scene, w-x-y-z separated by spaces.
            color (str): The base color of the object in th scene, r-g-b-opacity separated by spaces, scaled 0.0  to 1.0

        Returns:
            dict[str, list[ET.Element]]: A dictionary where the keys are tags of XML elements in the MJCF file, and the
            values are lists of XML elements to be appended as children to those XML elements.
        """
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

    def bind_to_data(self, data: mujoco.MjData):
        self.data = data
        self.actr = data.actuator(self.name + "_actr")
        self.ctrl = self.actr.ctrl
        self.sensor = data.sensor(self.name + "_velocimeter").data

