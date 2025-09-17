import os

import aiml_virtual
from aiml_virtual.scene import Scene
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie import Crazyflie
from aiml_virtual.trajectory.skyc_trajectory import SkycTrajectory, extract_trajectories
from aiml_virtual.utils.utils_general import quaternion_from_euler
from aiml_virtual.simulator import Simulator

# TODO: comments and docstrings
class SkycViewer:
    def __init__(self, skyc_file: str):
        self.trajectories: list[SkycTrajectory] = extract_trajectories(skyc_file)

    def play(self, delay: float = 1.0):
        scn = Scene(os.path.join(aiml_virtual.xml_directory, "empty_checkerboard.xml"))
        for traj in self.trajectories:
            cf = Crazyflie()
            traj.time_offset = delay
            cf.trajectory = traj
            start = traj.traj.start
            quat = quaternion_from_euler(0, 0, start.yaw)
            scn.add_object(
                cf,
                f"{start.x} {start.y} {start.z}",
                f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                "0.5 0.5 0.5 1"
            )
        sim = Simulator(scn)
        with sim.launch():
            while not sim.display_should_close():
                sim.tick()  # tick steps the simulator, including all its subprocesses

