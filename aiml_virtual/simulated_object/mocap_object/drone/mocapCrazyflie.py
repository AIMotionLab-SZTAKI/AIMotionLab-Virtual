from typing import Optional
import xml.etree.ElementTree as ET

from aiml_virtual.simulated_object.mocap_object.drone import mocapDrone


class MocapCrazyflie(mocapDrone.MocapDrone):
    # static variables that are the same for every crazyflie
    OFFSET: str = "0.03275"  #: **classvar** | Distance of the motor axies from the center of mass in each dimension.
    OFFSET_Z: str = "0.0223"  #: **classvar** | Height of the propellers.
    MOTOR_PARAM: str = "0.02514"  #: **classvar** | Ratio of Z torque to thrust.
    MAX_THRUST: str = "0.16"  #: **classvar** | The maximum thrust each motor can generate.
    MASS: str = "0.028"  #: **classvar** | Mass of a crazyflie.
    DIAGINERTIA: str = "1.4e-5 1.4e-5 2.17e-5"  #: **classvar** | Diagonal inertia components of a crazyflie.
    COG: str = "0.0 0.0 0.0"  #: **classvar** | Location of the center of mass.
    PROP_COLOR: str = "0.1 0.1 0.1 1.0"  #: **classvar** | Color of the (old style) propellers.

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["mocapCrazyflie", "MocapCrazyflie", "mocapcrazyflie", "cf"]

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
        Lx1 = MocapCrazyflie.OFFSET
        Lx2 = MocapCrazyflie.OFFSET
        Ly = MocapCrazyflie.OFFSET
        Lz = MocapCrazyflie.OFFSET_Z
        drone = ET.Element("body", name=name, pos=pos, quat=quat, mocap="true")  # the top level body
        # this is the main body of the crazyflie (from mesh):
        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="crazyflie_body", rgba=color)
        # this is a geom created by making 4 motormounts placed in the right position:
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="crazyflie_4_motormounts",
                      rgba=color)
        # this is a geom created by making 4 motors placed in the right position:
        ET.SubElement(drone, "geom", name=name + "_4_motors", type="mesh", mesh="crazyflie_4_motors", rgba=color)
        ret = {"worldbody": [drone]}
        prop_pos = [f"{Lx2} -{Ly} {Lz}",
                    f"-{Lx1} -{Ly} {Lz}",
                    f"-{Lx1} {Ly} {Lz}",
                    f"{Lx2} {Ly} {Lz}"]
        for i, propeller in enumerate(self.propellers):
            prop_name = f"{name}_prop{i}"
            prop_body = ET.SubElement(drone, "body", name=prop_name)  # propeller body in the kinematic chain
            ET.SubElement(prop_body, "joint", name=prop_name, axis="0 0 1", pos=prop_pos[i])  # moves around z axis
            mesh = f"crazyflie_{propeller.dir_mesh}_prop"
            ET.SubElement(prop_body, "geom", name=prop_name, type="mesh", mesh=mesh, pos=prop_pos[i],
                          rgba=MocapCrazyflie.PROP_COLOR)  # geom initialized from the mesh
        return ret