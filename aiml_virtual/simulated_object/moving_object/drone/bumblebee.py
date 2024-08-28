import math
import xml.etree.ElementTree as ET
import mujoco
from typing import Optional, Any
from enum import Enum
import numpy as np

from aiml_virtual.simulated_object.moving_object.drone import drone
from aiml_virtual.controller import drone_geom_controller
from aiml_virtual import utils


class Bumblebee(drone.Drone):
    #static variables that are the same for every bumblebee, TODO: fix question marks
    OFFSET_X1 = "0.074"  # distance of the motor axies from the center of mass in the negative? x direction
    OFFSET_X2 = "0.091"  # distance of the motor axies from the center of mass in the positive? x direction
    OFFSET_Y = "0.087"  # distance of the motor axies from the center of mass in the y direction
    OFFSET_Z = "0.036"  # height of the motors  # ratio of Z torque to thrust?
    MOTOR_PARAM = "0.5954"  # ratio of Z torque to thrust?
    MAX_THRUST = "15"  # the maximum thrust each motor can generate
    MASS = "0.605"
    DIAGINERTIA = "1.5e-3 1.45e-3 2.66e-3"
    COG = "0.0085 0.0 0.0"
    PROP_COLOR = "0.1 0.02 0.5 1.0"

    def __init__(self):
        super().__init__()

    @property
    def input_matrix(self) -> np.ndarray:
        Lx1 = float(Bumblebee.OFFSET_X1)
        Lx2 = float(Bumblebee.OFFSET_X2)
        Ly = float(Bumblebee.OFFSET_Y)
        motor_param = float(Bumblebee.MOTOR_PARAM)
        return np.array([[1 / 4, -1 / (4 * Ly), -1 / (4 * Lx2), 1 / (4 * motor_param)],
                         [1 / 4, -1 / (4 * Ly), 1 / (4 * Lx1), -1 / (4 * motor_param)],
                         [1 / 4, 1 / (4 * Ly), 1 / (4 * Lx1), 1 / (4 * motor_param)],
                         [1 / 4, 1 / (4 * Ly), -1 / (4 * Lx2), -1 / (4 * motor_param)]])

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        # the identifiers to look for in the XML
        return ["Bumblebee", "bb", "bumblebee"]

    def set_default_controller(self) -> None:
        self.controller = drone_geom_controller.GeomControl(self.mass, self.inertia, self.model.opt.gravity,
                                                            k_r=4, k_v=2, k_R=1.7, k_w=0.2)

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        name = self.name
        mass = Bumblebee.MASS
        diaginertia = Bumblebee.DIAGINERTIA
        Lx1 = Bumblebee.OFFSET_X1
        Lx2 = Bumblebee.OFFSET_X2
        Ly = Bumblebee.OFFSET_Y
        Lz = Bumblebee.OFFSET_Z
        motor_param = Bumblebee.MOTOR_PARAM
        max_thrust = Bumblebee.MAX_THRUST
        cog = Bumblebee.COG
        drone = ET.Element("body", name=name, pos=pos, quat=quat)
        # need to rotate the body mesh to match optitrack orientation
        quat_mesh = utils.quaternion_from_euler(0, 0, math.radians(270))
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])
        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str,
                      mesh="bumblebee_body", rgba=color)
        ret = {"worldbody": [drone],
               "actuator": [],
               "sensor": []}
        # TODO: safety sphere?
        # we give the inertia by hand instead of it auto-computing based on geoms
        ET.SubElement(drone, "inertial", pos=cog, diaginertia=diaginertia, mass=mass)
        ET.SubElement(drone, "joint", name=name, type="free")  # the free joint that allows this to move freely
        site_name = name + "_cog"
        ET.SubElement(drone, "site", name=site_name, pos="0 0 0", size="0.005")  # center of gravity
        prop_site_size = "0.0001"
        prop_mass = "0.00001"  # mass of the propeller is approximately zero it seems
        prop_pos = [f"{Lx2} -{Ly} {Lz}",
                    f"-{Lx1} -{Ly} {Lz}",
                    f"-{Lx1} {Ly} {Lz}",
                    f"{Lx2} {Ly} {Lz}"]
        for i, propeller in enumerate(self.propellers):
            prop_name = f"{name}_prop{i}"
            prop_body = ET.SubElement(drone, "body", name=prop_name)
            ET.SubElement(prop_body, "joint", name=prop_name, axis="0 0 1", pos=prop_pos[i])
            mesh = f"bumblebee_{propeller.dir_mesh}_prop"
            ET.SubElement(prop_body, "geom", name=prop_name, type="mesh", mesh=mesh, mass=prop_mass,
                          pos=prop_pos[i], rgba=Bumblebee.PROP_COLOR)
            ET.SubElement(drone, "site", name=prop_name, pos=prop_pos[i], size=prop_site_size)
            actuator = ET.Element("general", site=prop_name, name=f"{name}_actr{i}",
                                  gear=f"0 0 1 0 0 {propeller.dir_str}{motor_param}",
                                  ctrllimited="true", ctrlrange=f"0 {max_thrust}")
            ret["actuator"].append(actuator)
        ret["sensor"].append(ET.Element("gyro", site=site_name, name=name + "_gyro"))
        ret["sensor"].append(ET.Element("framelinvel", objtype="site", objname=site_name, name=name + "_velocimeter"))
        ret["sensor"].append(ET.Element("accelerometer", site=site_name, name=name + "_accelerometer"))
        ret["sensor"].append(ET.Element("framepos", objtype="site", objname=site_name, name=name + "_posimeter"))
        ret["sensor"].append(ET.Element("framequat", objtype="site", objname=site_name, name=name + "_orimeter"))
        ret["sensor"].append(
            ET.Element("frameangacc", objtype="site", objname=site_name, name=name + "_ang_accelerometer"))
        return ret






