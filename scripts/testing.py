"""
testing.py
========================

Docstring needs to be written!
"""

import os
import sys
import pathlib
import numpy as np

# make sure imports work by adding the aiml_virtual directory to path:
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(file_dir, '..'))

import aiml_virtual.scene as scene
import aiml_virtual.simulated_object.dynamic_object.controlled_object.bicycle as bicycle
import aiml_virtual.simulator as simulator
import aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie as cf
import aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.bumblebee as bb
from aiml_virtual.trajectory import dummy_drone_trajectory, skyc_trajectory
from aiml_virtual.simulated_object.dynamic_object import dynamic_object

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parents[1].resolve().as_posix()
    xml_directory = os.path.join(project_root, "xml_models")
    scene = scene.Scene(os.path.join(xml_directory, "bicycle.xml"))
    bike1 = bicycle.Bicycle()
    scene.add_object(bike1, "0 1 0", "1 0 0 0", "0.5 0.5 0.5 1")
    bike2 = bicycle.Bicycle()
    scene.add_object(bike2, "0 -1 0", "1 0 0 0", "0.5 0.5 0.5 1")
    cf0 = cf.Crazyflie()
    traj = skyc_trajectory.SkycTrajectory("misc/skyc_example.skyc")
    cf0.trajectory = traj
    scene.add_object(cf0, "0 0 0", "1 0 0 0", "0.5 0.5 0.5 1")
    bb0 = bb.Bumblebee()
    bb0.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([-1, 0, 1]))
    scene.add_object(bb0, "-1 0 0.5", "1 0 0 0", "0.5 0.5 0.5 1")

    payload = dynamic_object.DynamicPayload()
    scene.add_object(payload, "0 0 1.0")
    # dummy_mocap_start_poses = {
    #     "cf0": (np.array([1, -1, 1]), np.array([0, 0, 0, 1])),
    #     "cf1": (np.array([1, 1, 1]), np.array([0, 0, 0, 1])),
    #     "bb0": (np.array([-1, 1, 1]), np.array([0, 0, 0, 1])),
    #     "bb1": (np.array([-1, -1, 1]), np.array([0, 0, 0, 1]))
    # }
    # mocap1_framegen = partial(dummy_mocap_source.generate_circular_paths, start_poses=dummy_mocap_start_poses, T=5)
    # mocap1 = dummy_mocap_source.DummyMocapSource(frame_generator=mocap1_framegen, fps=120)
    # mcf0 = mcf.MocapCrazyflie(mocap1, "cf0")
    # scene.add_object(mcf0, "1 0 0", "1 0 0 0", "0.5 0.0 0.0 1")
    # mbb0 = mbb.MocapBumblebee(mocap1, "bb0")
    # scene.add_object(mbb0, "-1 0 0", "1 0 0 0", "0.5 0.0 0.0 1")
    # scene.add_mocap_objects(mocap1, color="0 0.5 0 1")
    # scene.remove_object(scene.simulated_objects[-1])
    #
    # mocap2 = optitrack_mocap_source.OptitrackMocapSource(ip="192.168.2.141")
    # mcf2 = mcf.MocapCrazyflie(mocap2, "cf9")
    # scene.add_object(mcf2, color="0.5 0 0 1")

    sim = simulator.Simulator(scene, update_freq=500, target_fps=100)
    with sim.launch_viewer():
        while sim.viewer.is_running():
            sim.tick()

