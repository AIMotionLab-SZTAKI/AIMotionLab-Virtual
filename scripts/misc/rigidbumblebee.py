"""
This script shows how dynamic objects work.
"""
import os
import sys
import pathlib
import numpy as np
import math
import xml.etree.ElementTree as ET
from typing import Optional
import mujoco

# make sure imports work by adding the necessary folders to the path:
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())
xml_directory = os.path.join(project_root.resolve().as_posix(), "xml_models")
project_root = project_root.resolve().as_posix()
from aiml_virtual import scene, simulator
from aiml_virtual.trajectory import dummy_drone_trajectory, skyc_trajectory
from aiml_virtual.utils import utils_general
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import bumblebee


Bumblebee = bumblebee.Bumblebee

class FixedPropBumblebee(Bumblebee):
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "FixedPropBumblebee"

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
        drone = ET.Element("body", name=name, pos=pos, quat=quat)  # the top level body
        # need to rotate the body mesh to match optitrack orientation
        quat_mesh = utils_general.quaternion_from_euler(0, 0, math.radians(270))
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])
        # this is the main body of the crazyflie (from mesh):
        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str,
                      mesh="bumblebee_body", rgba=color)
        ret = {"worldbody": [drone],
               "actuator": [],
               "sensor": []}
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
            mesh = f"bumblebee_{propeller.dir_mesh}_prop"
            ET.SubElement(prop_body, "geom", name=prop_name, type="mesh", mesh=mesh, mass=prop_mass,
                          pos=prop_pos[i], rgba=Bumblebee.PROP_COLOR)  # geom initialized from the mesh
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
        ret["sensor"].append(
            ET.Element("frameangacc", objtype="site", objname=site_name, name=name + "_ang_accelerometer"))
        return ret

    def bind_to_data(self, data: mujoco.MjData) -> None:
        if self.model is None:
            raise RuntimeError
        if self.controller is None:
            self.set_default_controller()
        self.data = data
        self.sensors["ang_vel"] = self.data.sensor(self.name + "_gyro").data
        self.sensors["vel"] = self.data.sensor(self.name + "_velocimeter").data
        self.sensors["acc"] = self.data.sensor(self.name + "_accelerometer").data
        self.sensors["pos"] = self.data.sensor(self.name + "_posimeter").data
        self.sensors["quat"] = self.data.sensor(self.name + "_orimeter").data
        self.sensors["ang_acc"] = self.data.sensor(self.name + "_ang_accelerometer").data
        for i, propeller in enumerate(self.propellers):
            propeller.ctrl = self.data.actuator(f"{self.name}_actr{i}").ctrl
            propeller.actr_force = self.data.actuator(f"{self.name}_actr{i}").force

if __name__ == "__main__":
    # As mentioned in 02_build_scene.py, we can simulate physics using DynamicObjects. So far we've only seen dynamic
    # objects that had no actuators. Let's change that, and build a scene with dynamic objects based on the empty
    # checkerboard scene base!
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=f"example_scene_3.xml")
    bb = FixedPropBumblebee()
    scn.add_object(bb, "0 0 1.5", "1 0 0 0", "0.5 0.5 0.5 1")
    bb.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([0, 0, 0.5]))
    sim = simulator.Simulator(scn)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses
