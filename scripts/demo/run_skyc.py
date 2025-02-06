import os
import sys
import pathlib
import numpy as np
from skyc_utils.skyc_maker import write_skyc, XYZYaw, Trajectory, TrajectoryType
from skyc_utils.skyc_inspector import get_traj_data

# The lines under here are intended to make sure imports work, by adding parent folders to the path (i.e. the list
# of folders where the interpreter will look for a given package when you try to import it). This is to account for
# differences in what the interpreter identifies as your current working directory when launching these scripts
# from the command line as regular scripts vs with the -m option vs from PyCharm, as well as the script being placed
# in any depth of sub-sub-subfolder.
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())

import aiml_virtual
xml_directory = aiml_virtual.xml_directory
from aiml_virtual.scene import Scene
from aiml_virtual.simulator import Simulator
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie import Crazyflie
from aiml_virtual.trajectory.skyc_trajectory import SkycTrajectory

def generate_skyc():
    z = 0.75
    d = 0.75
    offsets = [np.array([d, d, 0, 0]),
               np.array([-d, d, 0, 0]),
               np.array([d, -d, 0, 0]),
               np.array([-d, -d, 0, 0])]
    points = [np.array([0, 0, z, 0]),
              np.array([d, 0, z, 90]),
              np.array([d, d, z, 180]),
              np.array([-d, d, z, 270]),
              np.array([-d, -d, z, 360]),
              np.array([0, 0, z, 0]),
              np.array([0, 0, 0, 0])]
    trajType = TrajectoryType.COMPRESSED
    trajectories = [Trajectory(trajType), Trajectory(trajType), Trajectory(trajType), Trajectory(trajType)]
    for i, trajectory in enumerate(trajectories):
        offset = offsets[i]
        trajectory.set_start(XYZYaw(offset))
        for point in points:
            trajectory.add_goto(XYZYaw(point + offset), dt=3)
    write_skyc(trajectories)

def initialize_from_skyc(skyc_file: str, base: str, save_filename: str) -> Scene:
    scn = Scene(os.path.join(xml_directory, base), save_filename=save_filename)
    traj_data = get_traj_data(skyc_file)
    for i in range(len(traj_data)):
        cf = Crazyflie()
        traj = SkycTrajectory(skyc_file, i)
        cf.trajectory = traj
        start_pos = traj.evaluate(0.0).get("target_pos")
        scn.add_object(cf, f"{start_pos[0]} {start_pos[1]} {start_pos[2]}", "1 0 0 0", "1 0 0 1")
    return scn


if __name__ == "__main__":
    generate_skyc()
    scene = initialize_from_skyc("run_skyc.skyc", "scene_base.xml", "run_skyc.xml")
    simulator = Simulator(scene)
    with simulator.launch():
        while not simulator.display_should_close():
            simulator.tick()




