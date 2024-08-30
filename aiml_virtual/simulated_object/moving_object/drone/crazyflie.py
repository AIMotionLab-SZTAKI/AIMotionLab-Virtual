"""
This module contains the class encapsulating the crazyflie drone.
"""

import xml.etree.ElementTree as ET
from typing import Optional, Any
import numpy as np

from aiml_virtual.simulated_object.moving_object.drone import drone
from aiml_virtual.controller import drone_geom_controller


class Crazyflie(drone.Drone):
    """
    Class that specializes the Drone superclass for a crazyflie.
    """
    # static variables that are the same for every crazyflie
    OFFSET: str = "0.03275"  #: **classvar** | Distance of the motor axies from the center of mass in each dimension.
    OFFSET_Z: str = "0.0223"  #: **classvar** | Height of the propellers.
    MOTOR_PARAM: str = "0.02514"  #: **classvar** | Ratio of Z torque to thrust.
    MAX_THRUST: str = "0.16"  #: **classvar** | The maximum thrust each motor can generate.
    MASS: str = "0.028"  #: **classvar** | Mass of a crazyflie.
    DIAGINERTIA: str = "1.4e-5 1.4e-5 2.17e-5"  #: **classvar** | Diagonal inertia components of a crazyflie.
    COG: str = "0.0 0.0 0.0"  #: **classvar** | Location of the center of mass.
    PROP_COLOR: str = "0.1 0.1 0.1 1.0"  #: **classvar** | Color of the (old style) propellers.

    def __init__(self):
        super().__init__()

    @property
    def input_matrix(self) -> np.ndarray:
        """
        Overrides (implements) Drone.input_matrix. Property to grab the input matrix for use in input allocation: it
        shows the connection between the control outputs (thrust-toruqeX-torqueY-torqueZ) and the individual motor
        thrusts.
        """
        Lx = float(Crazyflie.OFFSET)
        Ly = Lx
        motor_param = float(Crazyflie.MOTOR_PARAM)
        return np.array([[1/4, -1/(4*Ly), -1/(4*Lx), 1/(4*motor_param)],
                         [1/4, -1/(4*Ly), 1/(4*Lx), -1/(4*motor_param)],
                         [1/4, 1/(4*Ly), 1/(4*Lx), 1/(4*motor_param)],
                         [1/4, 1/(4*Ly), -1/(4*Lx), -1/(4*motor_param)]])

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Overrides method in MovingObject to specify whether to check for aliases when parsing an XML.

        Returns:
            Optional[list[str]]: The list of aliases for objects belonging to this class.
        """
        return ["Crazyflie", "cf", "crazyflie"]

    def set_default_controller(self) -> None:
        """
        Sets the GeomControl as the controller (default in this package).
        """
        self.controller = drone_geom_controller.GeomControl(self.mass, self.inertia, self.model.opt.gravity)

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

        .. note::
            In the original version, the mass of the drone and the mass of the prop are getting confused and
            weirdly overwritten. The add_drone_common_parts function wants a 'mass' parameter, and uses this mass
            parameter to set the inertial element of the XML, which makes sense. HOWEVER, the actual value passed to
            this mass parameter is CRAZYFLIE_PROP.MASS.value, which doesn't make sense. It is then overwritten to be
            "0.00001", which is what is used to set the mass of the propellers, instead of CRAZYFLIE_PROP.MASS.value.
            Here the distinction is more clear: mass only ever refers to the drone, prop_mass only ever refers to the
            propeller.
        """
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
        drone = ET.Element("body", name=name, pos=pos, quat=quat)  # the top level body
        # this is the main body of the crazyflie (from mesh):
        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="crazyflie_body", rgba=color)
        # this is a geom created by making 4 motormounts placed in the right position:
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="crazyflie_4_motormounts",
                      rgba=color)
        # this is a geom created by making 4 motors placed in the right position:
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
            prop_body = ET.SubElement(drone, "body", name=prop_name)  # propeller body in the kinematic chain
            ET.SubElement(prop_body, "joint", name=prop_name, axis="0 0 1", pos=prop_pos[i])  # moves around z axis
            mesh = f"crazyflie_{propeller.dir_mesh}_prop"
            ET.SubElement(prop_body, "geom", name=prop_name, type="mesh", mesh=mesh, mass=prop_mass,
                          pos=prop_pos[i], rgba=Crazyflie.PROP_COLOR)  # geom initialized from the mesh
            ET.SubElement(drone, "site", name=prop_name, pos=prop_pos[i], size=prop_site_size)
            # the line under here is notable: All actuators exert force to generate lift in the same way. However, due
            # to their drag, they will also exert a torque around the Z axis. The direction of this torque depends on
            # the direction of the spin, its relation to the propeller thrush is roughly described by motor_param.
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