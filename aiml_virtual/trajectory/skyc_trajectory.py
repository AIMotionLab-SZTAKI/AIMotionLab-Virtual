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

class BezierCurve:
    """
    Convenience class to encapsulate a single bezier curve (as opposed to a series of bezier curves like in a skyc file).
    """
    def __init__(self, points: list[float], start_time: float, end_time: float):
        self.points: list[float] = points  #: The control points of the curve (the coefficients).
        self.start_time: float = start_time  #: The start of the curve.
        self.end_time: float = end_time  #: The end of the curve.
        coeffs: list[tuple[float, ...]] = list(zip(*points))  #: This is equivalent to transposing the matrix formed by the points.
        self.BPolys: list[BPoly] = [BPoly(np.array(coeffs).reshape(len(coeffs), 1), np.array([start_time, end_time]))
                                    for coeffs in coeffs]  #: The interpolate.BPoly objects formed from the points.

    def x(self, time: float, nu: int = 0) -> np.ndarray:
        """
        Wrapper around the __call__ function of the appropriate BPoly object; it evaluates the x Bezier Polynomial.

        Args:
            time (float): The timestamp at which to evaluate.
            nu (int): The number of derivative to evaluate.

        Returns:
            np.ndarray: The "nu"th derivative at "time" time.
        """
        return self.BPolys[0](time, nu)

    def y(self, time: float, nu: int = 0) -> np.ndarray:
        """
        Wrapper around the __call__ function of the appropriate BPoly object; it evaluates the y Bezier Polynomial.

        Args:
            time (float): The timestamp at which to evaluate.
            nu (int): The number of derivative to evaluate.

        Returns:
            np.ndarray: The "nu"th derivative at "time" time.
        """
        return self.BPolys[1](time, nu)

    def z(self, time: float, nu: int = 0) -> np.ndarray:
        """
        Wrapper around the __call__ function of the appropriate BPoly object; it evaluates the z Bezier Polynomial.

        Args:
            time (float): The timestamp at which to evaluate.
            nu (int): The number of derivative to evaluate.

        Returns:
            np.ndarray: The "nu"th derivative at "time" time.
        """
        return self.BPolys[2](time, nu)

    def yaw(self, time: float, nu: int = 0) -> np.ndarray:
        """
        Wrapper around the __call__ function of the appropriate BPoly object; it evaluates the yaw Bezier Polynomial.

        Args:
            time (float): The timestamp at which to evaluate.
            nu (int): The number of derivative to evaluate.

        Returns:
            np.ndarray: The "nu"th derivative at "time" time.
        """
        return self.BPolys[3](time, nu)


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
    """
    def __init__(self, skyc_file: str, traj_id: int = 0):
        super().__init__()
        traj_data: list[dict] = skyc_inspector.get_traj_data(skyc_file)
        self.traj_data = traj_data[traj_id]  #: The dictionary that can be read from trajectory.json.
        self.bezier_curves: list[BezierCurve] = []  #: The list of individual segments.
        segments = self.traj_data.get("points")
        assert segments is not None
        for i in range(1, len(segments)):
            prev_segment = segments[i - 1]
            start_point = prev_segment[1]  # The current segment's start pose is the end pose of the previous one.
            start_time = prev_segment[0]  # The current segment starts when the previous ends.
            segment = segments[i]
            end_point = segment[1]
            end_time = segment[0]
            ctrl_points = segment[2]  # The "extra" control points, which aren't physically on the curve.
            # points will contain all points of the bezier curve, including the start and end, unlike in trajectory.json
            points = [start_point, *ctrl_points, end_point] if ctrl_points else [start_point, end_point]
            self.bezier_curves.append(BezierCurve(points, start_time, end_time))


    def select_curve(self, time: float) -> BezierCurve:
        """
        Calculates which Bezier segment the timestamp falls under.

        Args:
            time (float): The time at which we're investigating the trajectory.

        Returns:
            BezierCurve: The segment which will be traversed at the given timestamp.
        """
        if time < self.traj_data["takeoffTime"]:  # this should never happen, but in case it does we return the 0th
            return self.bezier_curves[0]
        # This loop breaks if we find a match, meaning that it relies on the segments being sorted in increasing
        # timestamp order.
        for bezier_curve in self.bezier_curves:
            if bezier_curve.start_time <= time <= bezier_curve.end_time:
                return bezier_curve
        return self.bezier_curves[-1]

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
        curve = self.select_curve(time)
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
