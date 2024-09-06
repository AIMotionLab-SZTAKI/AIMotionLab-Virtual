"""
This module contains the classes responsible for reading and evaluating a trajectory read from a skyc file.
"""

from typing import Any
import bisect

from scipy import interpolate
import numpy as np
from scipy.interpolate import BPoly
from skyc_utils import skyc_inspector

from aiml_virtual.trajectory import trajectory

BezierCurve = skyc_inspector.BezierCurve
TrajEvaluator = skyc_inspector.TrajEvaluator

class SkycTrajectory(trajectory.Trajectory):
    """
    Class that evaluates a trajectory read from a skyc file. A skyc file may hold several trajectories, so when
    initializing a SkycTrajectory, one may supply the ID of the trajectory in question (default is 0). Skyc trajectories
    consist of Bezier curves, represented in the appropriate trajectory.json file, selected by the aforementioned ID.
    In this trajectory.json there will be a "points" list, where each element describes a Bezier curve like so:

    - A float: this is the end time of the curve.
    - A list of  four floats (x-y-z-yaw): this is the physical end of the curve.
    - A variable length list of elements like the previous one: these are the (optional) extra control points.

    A Bezier curve is described by its parameter and its control points. The first and last control points are the
    beginning and end of said curve, so in a piecewise representation, if we stored each point for each individual
    segment, the end time and end pose of the Nth semgnet matches the start time and pose of the N-1th segment, leading
    to redundancy in the representation. Removing this redundancy is the rationale behind the described trajectory.json.
    This means that if "points" is of length N, there will be only N-1 segments in the trajectory. To this end, the 0th
    segment is special (it has to be, since it has no previous segment to describe its starting time/point): it shall
    have no extra control points.

    .. note::
        The fact that functions such as select_curve and evaluate and the helper class BezierCurve have to be written
        is testament to the fact that the skyc_utils package leaves a lot to be desired. It should be reworked.
    """
    def __init__(self, skyc_file: str, traj_id: int = 0):
        super().__init__()
        traj_data: list[dict] = skyc_inspector.get_traj_data(skyc_file)
        self.traj_data = traj_data[traj_id]  #: The dictionary that can be read from trajectory.json.
        self.evaluator = TrajEvaluator(self.traj_data)

    def evaluate(self, time: float) -> dict[str, Any]:
        """
        Overrides (implements) superclass' evaluate. Returns a setpoint according to the trajectory.json that was
        selected by the skyc file, ID, and the timestamp, or if the timestamp is after the trajectory, returns the end
        of the trajectory (with the derivatives set to 0).

        Args:
            time (float): The timestamp when the trajectory must be evaluated.

        Returns:
            dict[str, Any]: The desired setpoint at the provided timestamp.
        """
        curve = self.evaluator.select_curve(time)
        t_land = self.traj_data["landingTime"]
        if time <= t_land:  # If the trajectory isn't supposed to be over yet.
            retval = {
                "load_mass": 0.0,
                "target_pos": np.array([curve.x(time), curve.y(time), curve.z(time)]),
                "target_vel": np.array([curve.x(time, nu=1), curve.y(time, nu=1), curve.z(time, nu=1)]),
                "target_acc": np.array([curve.x(time, nu=2), curve.y(time, nu=2), curve.z(time, nu=2)]),
                "target_rpy": np.array([0, 0, np.deg2rad(curve.yaw(time))]),
                "target_ang_vel": np.array([0, 0, np.deg2rad(curve.yaw(time, nu=1))])
            }
        else:
            retval = {
                "load_mass": 0.0,
                "target_pos": np.array([curve.x(t_land), curve.y(t_land), curve.z(t_land)]),
                "target_vel": np.zeros(3),  # We've stopped: speed should be 0.
                "target_acc": np.zeros(3),  # We've stopped: traj should be 0.
                "target_rpy": np.array([0, 0, curve.yaw(t_land)]),
                "target_ang_vel": np.zeros(3)  # we've stopped: angular velocity should be 0.
            }
        return retval
