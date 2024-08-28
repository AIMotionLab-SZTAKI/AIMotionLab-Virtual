import xml.etree.ElementTree as ET
import mujoco
from typing import Optional, Any
from enum import Enum
import numpy as np

from aiml_virtual.simulated_object.moving_object.drone import drone
from aiml_virtual.controller import drone_geom_controller


class Crazyflie(drone.Drone):
    # static variables that are the same for every crazyflie
    OFFSET = "0.03275"  # distance of the motor axies from the center of mass in each dimension
    OFFSET_Z = "0.0223"  # height of the motors
    MOTOR_PARAM = "0.02514"  # ratio of Z torque to thrust? TODO
    MAX_THRUST = "0.16"  # the maximum thrust each motor can generate
    MASS = "0.028"
    DIAGINERTIA = "1.4e-5 1.4e-5 2.17e-5"
    COG = "0.0 0.0 0.0"
    PROP_COLOR = "0.1 0.1 0.1 1.0"

    def __init__(self):
        super().__init__()

    @property
    def input_matrix(self) -> np.ndarray:
        Lx = float(Crazyflie.OFFSET)
        Ly = Lx
        motor_param = float(Crazyflie.MOTOR_PARAM)
        return np.array([[1/4, -1/(4*Ly), -1/(4*Lx), 1/(4*motor_param)],
                         [1/4, -1/(4*Ly), 1/(4*Lx), -1/(4*motor_param)],
                         [1/4, 1/(4*Ly), 1/(4*Lx), 1/(4*motor_param)],
                         [1/4, 1/(4*Ly), -1/(4*Lx), -1/(4*motor_param)]])

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        # the identifiers to look for in the XML
        return ["Crazyflie", "cf", "crazyflie"]

    def set_default_controller(self) -> None:
        self.controller = drone_geom_controller.GeomControl(self.mass, self.inertia, self.model.opt.gravity)

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        # TODO: separate crazyflie and bumblebee elements and common parts (for simplicity, just doing crazyflie for now)
        # Also todo: comment this whole thing cause i don't understand the half of it...
        # NOTE: In the original version, the mass of the drone and the mass of the prop are getting confused and
        # weirdly overwritten. The add_drone_common_parts function wants a 'mass' parameter, and uses this mass
        # parameter to set the inertial element of the XML, which makes sense. HOWEVER, the actual value passed to
        # this mass parameter is CRAZYFLIE_PROP.MASS.value, which doesn't make sense. It is then overwritten to be
        # "0.00001", which is what is used to set the mass of the propellers, instead of CRAZYFLIE_PROP.MASS.value
        name = self.name
        mass = Crazyflie.MASS
        diaginertia = Crazyflie.DIAGINERTIA
        Lx1 = Crazyflie.OFFSET
        Lx2 = Crazyflie.OFFSET
        Ly = Crazyflie.OFFSET
        Lz = Crazyflie.OFFSET_Z
        motor_param = Crazyflie.MOTOR_PARAM
        max_thrust = Crazyflie.MAX_THRUST
        cog = Crazyflie.COG
        drone = ET.Element("body", name=name, pos=pos, quat=quat)  # this is the parent element
        # this is the main body of the crazyflie (from mesh)
        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="crazyflie_body", rgba=color)
        # this is a geom created by making 4 motormounts placed in the right position
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="crazyflie_4_motormounts",
                      rgba=color)
        # this is a geom created by making 4 motors placed in the right position
        ET.SubElement(drone, "geom", name=name + "_4_motors", type="mesh", mesh="crazyflie_4_motors", rgba=color)
        ret = {"worldbody": [drone],
               "actuator": [],
               "sensor": []}
        # TODO: safety sphere?
        # ET.SubElement(drone, "geom", type="sphere", name=name + "_sphere", size="1.0", rgba=color, contype="0",
        #               conaffinity="0")
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
            mesh = f"crazyflie_{propeller.dir_mesh}_prop"
            ET.SubElement(prop_body, "geom", name=prop_name, type="mesh", mesh=mesh, mass=prop_mass,
                          pos=prop_pos[i], rgba=Crazyflie.PROP_COLOR)
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
        ret["sensor"].append(ET.Element("frameangacc", objtype="site", objname=site_name, name=name + "_ang_accelerometer"))
        return ret