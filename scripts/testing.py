"""
testing.py
========================

Docstring needs to be written!
"""

import os
import pathlib
import numpy as np

import aiml_virtual.scene as scene
import aiml_virtual.simulated_object.moving_object.bicycle as bicycle
import aiml_virtual.simulator as simulator
import aiml_virtual.simulated_object.moving_object.drone.crazyflie as cf
import aiml_virtual.simulated_object.moving_object.drone.bumblebee as bb
import aiml_virtual.simulated_object.mocap_object.drone.mocapCrazyflie as mcf
from aiml_virtual.trajectory import dummy_drone_trajectory, skyc_trajectory
from aiml_virtual.simulated_object.mocap_object.mocap_source import dummy_mocap_source
from aiml_virtual.utils.dummy_mocap_server import DummyRigidBody

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parents[1].resolve().as_posix()
    xml_directory = os.path.join(project_root, "xml_models")
    scene = scene.Scene(os.path.join(xml_directory, "bicycle.xml"))
    bike1 = bicycle.Bicycle()
    scene.add_object(bike1, "0 1 0", "1 0 0 0", "0.5 0.5 0.5 1")
    bike2 = bicycle.Bicycle()
    scene.add_object(bike2, "0 -1 0", "1 0 0 0", "0.5 0.5 0.5 1")
    cf0 = cf.Crazyflie()
    traj = skyc_trajectory.SkycTrajectory("skyc_example.skyc")
    cf0.trajectory = traj
    scene.add_object(cf0, "0 0 0", "1 0 0 0", "0.5 0.5 0.5 1")
    bb0 = bb.Bumblebee()
    bb0.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([-1, 0, 1]))
    scene.add_object(bb0, "-1 0 0.5", "1 0 0 0", "0.5 0.5 0.5 1")

    scene.add_object(bb0, "-1 0 0.5", "1 0 0 0", "0.5 0.5 0.5 1")

    mocap = dummy_mocap_source.DummyMocapSource("localhost", 9999, 50)
    mcf0 = mcf.MocapCrazyflie(mocap)
    scene.add_object(mcf0, "1 0 0", "1 0 0 0", "0.5 0.0 0.0 1")
    mcf1 = mcf.MocapCrazyflie(mocap)
    scene.add_object(mcf1, "-1 0 0", "1 0 0 0", "0.5 0.0 0.0 1")



    sim = simulator.Simulator(scene, control_freq=500, target_fps=100)
    with sim.launch_viewer():
        while sim.viewer.is_running():
            sim.tick()

