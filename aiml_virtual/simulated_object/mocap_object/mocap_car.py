"""
This module contains classes relating to mocap representation of the Car (Trailer, etc.).
It also relies on some constants defined in the dynamic car's module.
"""

from typing import Optional
from xml.etree import ElementTree as ET
import numpy as np
from aiml_virtual.mocap.mocap_source import MocapSource

from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import Wheel, TRAILER

class MocapCar(mocap_object.MocapObject):
    """
    Class that encapsulates a mocap car (which has no actuators or dynamics).
    """
    WHEEL_X: float = 0.16113  #: distance between axles and center of gravity
    WHEEL_Y: float = 0.122385  #: distance between center line and wheels
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "MocapCar"

    def __init__(self, source: Optional[MocapSource] = None, mocap_name: Optional[str] = None):
        super().__init__(source, mocap_name)
        self.wheels: dict[str, Wheel] = {
            "front_left": Wheel(self.name + Wheel.FRONT_LEFT, pos=f"{MocapCar.WHEEL_X} {MocapCar.WHEEL_Y} 0"),
            "rear_left": Wheel(self.name + Wheel.REAR_LEFT, pos=f"-{MocapCar.WHEEL_X} {MocapCar.WHEEL_Y} 0"),
            "front_right": Wheel(self.name + Wheel.FRONT_RIGHT, pos=f"{MocapCar.WHEEL_X} -{MocapCar.WHEEL_Y} 0"),
            "rear_right": Wheel(self.name + Wheel.REAR_RIGHT, pos=f"-{MocapCar.WHEEL_X} -{MocapCar.WHEEL_Y} 0")
        }  #: Object to access the relevant mjdata for Wheels.
        self.offset = np.array([0, 0, -0.097])

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        car = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")  # top level mocap body
        ######################### ADDING ALL THE GEOMS THAT MAKE UP THE MAIN BODY OF THE CAR ###########################
        ET.SubElement(car, "geom", name=self.name + "_chassis_b", type="box", size=".10113 .1016 .02", pos="-.06 0 0",
                      rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_chassis_f", type="box", size=".06 .07 .02", pos=".10113 0 0",
                      rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_front", type="box", size=".052388 .02 .02", pos=".2135 0 0",
                      rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_back", type="box", size=".062488 .02 .02", pos="-.2236 0 0",
                      rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_front_bumper", type="box", size=".005 .09 .02",
                      pos=".265888 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_back_bumper", type="box", size=".005 .08 .02",
                      pos="-.2861 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_number", type="cylinder", size=".01984 .03", pos=".12 0 .05",
                      rgba="0.1 0.1 0.1 1.0")
        ET.SubElement(car, "geom", name=self.name + "_camera", type="box", size=".012 .06 0.02", pos=".18 0 .08")
        ET.SubElement(car, "geom", name=self.name + "_camera_holder", type="box", size=".012 .008 .02", pos=".18 0 .04")
        ET.SubElement(car, "geom", name=self.name + "_circuits", type="box", size=".08 .06 .03", pos="-.05 0 .05",
                      rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_antennal", type="box", size=".007 .004 .06", pos="-.16 -.01 .105",
                      euler="0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=self.name + "_antennar", type="box", size=".007 .004 .06", pos="-.16 .01 .105",
                      euler="-0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=self.name + "_antenna_holder", type="box", size=".008 .008 .02",
                      pos="-.16 0 .04", rgba=".1 .1 .1 1.0")
        ################################################ WHEELS ########################################################
        for wheel in self.wheels.values():
            ET.SubElement(car, "geom", name=wheel.name, type="cylinder", size=Wheel.SIZE, pos=wheel.pos,
                          rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")
        ret = {"worldbody": [car]}
        return ret


class MocapTrailer(mocap_object.MocapObject):
    """
    Class that encapsulates a mocap trailer (which in this approximation is not attached to a car!).
    """
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "MocapTrailer"

    def __init__(self, source: MocapSource, mocap_name: str):
        super().__init__(source, mocap_name)
        self.offset = np.array([0, 0.02, -0.08])

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        trailer = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(trailer, "geom", type="cylinder", size=f"0.0025 {TRAILER.DRAWBAR_LENGTH / 2}",  # drawbar geom
                      euler="0 1.571 0", pos=f"-{TRAILER.DRAWBAR_LENGTH / 2} 0 0")
        front_axle = ET.SubElement(trailer, "body", name=self.name + "_trailer_front_structure",
                                   pos=f"{-TRAILER.DRAWBAR_LENGTH} 0 0")  # the body for the front axle
        ET.SubElement(front_axle, "geom", type="box", size=f"0.0075 {TRAILER.TRACK_DISTANCE / 2} 0.0075")
        # front left wheel
        trailer_wheelfl = ET.SubElement(front_axle, "body", pos=f"0 {TRAILER.TRACK_DISTANCE / 2} 0",
                                        name=self.name + "_trailer_wheelfl")
        ET.SubElement(trailer_wheelfl, "geom", type="cylinder", size=".0315 .005", rgba="0.1 0.1 0.1 1.0",
                      euler="1.571 0 0", name=self.name + "_trailer_wheelfl")
        # front right wheel
        trailer_wheelfr = ET.SubElement(front_axle, "body", pos=f"0 {-TRAILER.TRACK_DISTANCE / 2} 0",
                                        name=self.name + "_trailer_wheelfr")
        ET.SubElement(trailer_wheelfr, "geom", type="cylinder", size=".0315 .005", rgba="0.1 0.1 0.1 1.0",
                      euler="1.571 0 0", name=self.name + "_trailer_wheelfr")
        # rear structure consists of the axle, the plates, holders and screws
        rear_structure = ET.SubElement(front_axle, "body", name=self.name + "_trailer_rear_structure")
        # top plate
        ET.SubElement(rear_structure, "geom", type="box", size=".25 .1475 .003", pos="-.21 0 .08",
                      rgba="0.7 0.6 0.35 1.0", euler="0 " + TRAILER.TOP_TILT + " 0")
        # rear axle
        ET.SubElement(rear_structure, "geom", type="box", size=f"0.0075 {TRAILER.TRACK_DISTANCE / 2} 0.0075",
                      pos=f"{-TRAILER.AXLE_DISTANCE} 0 0")
        # bottom plate
        ET.SubElement(rear_structure, "geom", type="box",
                      size=f"{(TRAILER.AXLE_DISTANCE + 0.05) / 2} {(TRAILER.TRACK_DISTANCE - 0.05) / 2} 0.0025",
                      pos=f"{-TRAILER.AXLE_DISTANCE / 2} 0 0.014", rgba="0.9 0.9 0.9 0.2",
                      euler="0 " + TRAILER.TOP_TILT + " 0")
        # 5 screws
        ET.SubElement(rear_structure, "geom", type="cylinder", size="0.0025 0.035", pos=".01 0 0.05",
                      euler="0 " + TRAILER.TOP_TILT + " 0")
        ET.SubElement(rear_structure, "geom", type="cylinder", size="0.0025 0.035", pos=".01 -0.03 0.05",
                      euler="0 " + TRAILER.TOP_TILT + " 0")
        ET.SubElement(rear_structure, "geom", type="cylinder", size="0.0025 0.035", pos=".01 0.03 0.05",
                      euler="0 " + TRAILER.TOP_TILT + " 0")
        ET.SubElement(rear_structure, "geom", type="cylinder", size="0.0025 0.035", pos="-.01 -0.015 0.05",
                      euler="0 " + TRAILER.TOP_TILT + " 0")
        ET.SubElement(rear_structure, "geom", type="cylinder", size="0.0025 0.035", pos="-.01 0.015 0.05",
                      euler="0 " + TRAILER.TOP_TILT + " 0")
        # rear holder
        ET.SubElement(rear_structure, "geom", type="cylinder", size="0.008 0.032",
                      pos=f"{-TRAILER.AXLE_DISTANCE + 0.02} 0 0.045", rgba="0.1 0.1 0.1 1.0",
                      euler="0 " + TRAILER.TOP_TILT + " 0")
        # rear left wheel
        trailer_wheelrl = ET.SubElement(rear_structure, "body",
                                        pos=f"{-TRAILER.AXLE_DISTANCE} {TRAILER.TRACK_DISTANCE / 2} 0",
                                        name=self.name + "_trailer_wheelrl")
        ET.SubElement(trailer_wheelrl, "geom", type="cylinder", size=".0315 .005", rgba="0.1 0.1 0.1 1.0",
                      euler="1.571 0 0", name=self.name + "_trailer_wheelrl")
        # rear right wheel
        trailer_wheelrr = ET.SubElement(rear_structure, "body",
                                        pos=f"{-TRAILER.AXLE_DISTANCE} {-TRAILER.TRACK_DISTANCE / 2} 0",
                                        name=self.name + "_trailer_wheelrr")
        ET.SubElement(trailer_wheelrr, "geom", type="cylinder", size=".0315 .005", rgba="0.1 0.1 0.1 1.0",
                      euler="1.571 0 0", name=self.name + "_trailer_wheelrr")
        ret = {"worldbody": [trailer]}
        return ret

