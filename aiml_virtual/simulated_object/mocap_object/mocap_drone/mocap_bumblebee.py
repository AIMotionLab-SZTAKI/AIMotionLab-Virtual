"""
This module contains the class encapsulating mocap bumblebee drones.
"""

from typing import Optional
import xml.etree.ElementTree as ET
import math

from aiml_virtual.simulated_object.mocap_object.mocap_drone import mocap_drone
from aiml_virtual.utils import utils_general

class MocapBumblebee(mocap_drone.MocapDrone):
    """
    Class that specializes the MocapDrone superclass for a bumblebee.
    """
    # static variables that are the same for every bumblebee
    OFFSET_X1 = "0.074"  #: **classvar** | Distance of the motor axies from the center of mass in the negative x direction.
    OFFSET_X2 = "0.091"  #: **classvar** | Distance of the motor axies from the center of mass in the positive x direction.
    OFFSET_Y = "0.087"  # #: **classvar** | Distance of the motor axies from the center of mass in the y direction.
    OFFSET_Z = "0.036"  #: **classvar** | Height of the propellers.
    MOTOR_PARAM = "0.5954"  #: **classvar** | Ratio of Z torque to thrust.
    MAX_THRUST = "15"  #: **classvar** | The maximum thrust each motor can generate.
    MASS = "0.605"  #: **classvar** | Mass of a crazyflie.
    DIAGINERTIA = "1.5e-3 1.45e-3 2.66e-3"  #: **classvar** | Diagonal inertia components of a bumblebee.
    COG = "0.0085 0.0 0.0"  #: **classvar** | Location of the center of mass.
    PROP_COLOR = "0.1 0.02 0.5 1.0"  #: **classvar** | Color of the propellers.

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["mocapBumblebee", "MocapBumblebee", "mocapbumblebee", "bb"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        """
        Overrides method in SimulatedObject. Generates all the necessary XML elements for the model.

        Args:
            pos (str): The position of the object in the scene, x-y-z separated by spaces. E.g.: "0 1 -1"
            quat (str): The quaternion orientation of the object in the scene, w-x-y-z separated by spaces.
            color (str): The base color of the object in th scene, r-g-b-opacity separated by spaces, scaled 0.0  to 1.0

        Returns:
            dict[str, list[ET.Element]]: A dictionary where the keys are tags of XML elements in the MJCF file, and the
            values are lists of XML elements to be appended as children to those XML elements.

        .. todo::
            Determine whether a mocap object needs inertial and mass parameters. They are omitted in the old
            aiml-virtual. Also, pos and quat may not be neccessary?
        """
        name = self.name
        Lx1 = MocapBumblebee.OFFSET_X1
        Lx2 = MocapBumblebee.OFFSET_X2
        Ly = MocapBumblebee.OFFSET_Y
        Lz = MocapBumblebee.OFFSET_Z
        drone = ET.Element("body", name=name, pos=pos, quat=quat, mocap="true")  # the top level body
        quat_mesh = utils_general.quaternion_from_euler(0, 0, math.radians(270))
        # this is the main body of the crazyflie (from mesh):
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])
        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str,
                      mesh="bumblebee_body", rgba=color)
        ret = {"worldbody": [drone]}
        prop_pos = [f"{Lx2} -{Ly} {Lz}",
                    f"-{Lx1} -{Ly} {Lz}",
                    f"-{Lx1} {Ly} {Lz}",
                    f"{Lx2} {Ly} {Lz}"]
        for i, propeller in enumerate(self.propellers):
            prop_name = f"{name}_prop{i}"
            prop_body = ET.SubElement(drone, "body", name=prop_name)  # propeller body in the kinematic chain
            ET.SubElement(prop_body, "joint", name=prop_name, axis="0 0 1", pos=prop_pos[i])  # moves around z axis
            mesh = f"bumblebee_{propeller.dir_mesh}_prop"
            ET.SubElement(prop_body, "geom", name=prop_name, type="mesh", mesh=mesh, pos=prop_pos[i],
                          rgba=MocapBumblebee.PROP_COLOR)  # geom initialized from the mesh
        return ret