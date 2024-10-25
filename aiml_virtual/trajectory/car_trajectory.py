"""
This module contains two trajectory implementations to be used with a Car.

.. note::
       The classes in this module, as well as the car trajectories were carried over from
       the previous aiml_virtual with as minimal changes as possible. I was not
       comfortable enough with the code and the working of the
       car to refactor it with confidence. Because of this, the docstrings and
       comments in this module (and the car_lpv_controller.py) are rather sparse
       and the code is in dire need of refactoring.

.. todo::
    Refactor this module.
"""

from typing import Any
import numpy as np
from scipy.interpolate import splrep, splprep, splev
from scipy.integrate import quad

from aiml_virtual.trajectory import trajectory
from aiml_virtual.utils import utils_general
from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import Car

normalize = utils_general.normalize
clamp = utils_general.clamp


class CarTrajectory(trajectory.Trajectory):
    """
    Class implementation of (time-parametrized) BSpline-based trajectories for autonomous ground vehicles
    """
    def __init__(self) -> None:
        super().__init__()
        self.output: dict[str, Any] = {}  #: Storage for the last return of evaluate
        self.pos_tck: list = []  #: the spline for x-y-z parametrized by arc length
        self.evol_tck: tuple = () #: the spline for the arc length parametrized by time
        self.t_end: float = 0  #: duration of the trajectory
        self.length: float = 0  #: length of the trajectory
        self.reversed: bool = False  #: whether the trajectory should be traversed forward or backward

    def build_from_points_smooth_const_speed(self, path_points: np.ndarray, path_smoothing: float, path_degree: int,
                                             virtual_speed: float) -> None:
        """
        Builds a trajectory using a set of reference waypoints and a virtual reference velocity with smoothed start
         and end reference.

        :param path_points: Reference points of the trajectory
        :type path_points: np.ndarray
        :param path_smoothing: Smoothing factor used for the spline interpolation
        :type path_smoothing: float
        :param path_degree: Degree of the fitted Spline
        :type path_degree: int
        :param virtual_speed: The virtual constant speed for the vehicle. In practice the designer smoothens the start and end of the trajectory
        :type virtual_speed: float
        """

        # handle negative speeds (reversing motion)
        if virtual_speed < 0:
            virtual_speed = abs(virtual_speed)
            self.reversed = True
        else:
            self.reversed = False

        x_points = path_points[:, 0].tolist()
        y_points = path_points[:, 1].tolist()

        # fit spline and evaluate
        tck, u, *_ = splprep([x_points, y_points], k=path_degree, s=path_smoothing)
        XY = splev(u, tck)

        # calculate arc length
        def integrand(x):
            dx, dy = splev(x, tck, der=1)
            return np.sqrt(dx ** 2 + dy ** 2)

        self.length, _ = quad(integrand, u[0], u[-1])

        # build spline for the path parameter
        self.pos_tck, _, *_ = splprep(XY, k=path_degree, s=0, u=np.linspace(0, self.length, len(XY[0])))

        # calculate travel time
        self.t_end = self.length / virtual_speed

        # construct third order s(t) poly, then refit it with splines
        A = np.linalg.pinv(np.array([[self.t_end, self.t_end ** 2, self.t_end ** 3],
                                     [0, 2 * self.t_end, 3 * self.t_end ** 2]])) @ np.array([[self.length], [0]])

        t_evol = np.linspace(0, self.t_end, 101)
        s_evol = A[0] * t_evol + A[1] * t_evol ** 2 + A[2] * t_evol ** 3

        self.evol_tck = splrep(t_evol, s_evol, k=1, s=0)

    def build_from_points_const_speed(self, path_points: np.ndarray, path_smoothing: float, path_degree: int,
                                      const_speed: float) -> None:
        """
        Builds a trajectory using a set of reference waypoints and a constant reference velocity.

        :param path_points: Reference points of the trajectory
        :type path_points: np.ndarray
        :param path_smoothing: Smoothing factor used for the spline interpolation
        :type path_smoothing: float
        :param path_degree: Degree of the fitted Spline
        :type path_degree: int
        :param const_speed: Constant speed of the vehicle
        :type const_speed: float
        """

        # handle negative speeds (reversing motion)
        if const_speed < 0:
            const_speed = abs(const_speed)
            self.reversed = True
        else:
            self.reversed = False

        x_points = path_points[:, 0].tolist()
        y_points = path_points[:, 1].tolist()

        # fit spline and evaluate
        tck, u, *_ = splprep([x_points, y_points], k=path_degree, s=path_smoothing)
        XY = splev(u, tck)

        # calculate arc length
        def integrand(x):
            dx, dy = splev(x, tck, der=1)
            return np.sqrt(dx ** 2 + dy ** 2)

        self.length, _ = quad(integrand, u[0], u[-1])

        # build spline for the path parameter
        self.pos_tck, _, *_ = splprep(XY, k=path_degree, s=0, u=np.linspace(0, self.length, len(XY[0])))

        # build constant speed profile
        self.t_end = self.length / const_speed

        t_evol = np.linspace(0, self.t_end, 101)
        s_evol = np.linspace(0, self.length, 101)

        self.evol_tck = splrep(t_evol, s_evol, k=1,
                               s=0)  # NOTE: This is completely overkill for constant veloctities but can be easily adjusted for more complex speed profiles

    def export_to_time_dependent(self) -> np.ndarray:
        """Exports the trajectory to a time dependent representation
        """
        t_eval = np.linspace(0, self.t_end, 100)

        s = splev(t_eval, self.evol_tck)
        (x, y) = splev(s, self.pos_tck)

        tck, t, *_ = splprep([x, y], k=5, s=0, u=t_eval)

        return tck

    def project_to_closest(self, pos: np.ndarray, param_estimate: float, projetion_window: float,
                           projection_step: float) -> float:
        """
        Projects the vehicle position onto the given path and returns the path parameter.
        The path parameter is the curvilinear abscissa along the path as the Bspline that represents the path is
        arc length parameterized.

        :param pos: Vehicle x,y position
        :type pos: np.ndarray
        :param param_estimate: Estimated value of the path parameter
        :type param_estimate: float
        :param projetion_window: Length of the projection window along the path
        :type projetion_window: float
        :param projection_step: Precision of the projection in the window
        :type projection_step: float
        :return: The path parameter
        :rtype: float

        """

        # calulate the endpoints of the projection window
        floored = clamp(param_estimate - projetion_window / 2, [0, self.length])
        ceiled = clamp(param_estimate + projetion_window / 2, [0, self.length])

        # create a grid on the window with the given precision
        window = np.linspace(floored, ceiled, round((ceiled - floored) / projection_step))

        # evaluate the path at each instance
        path_points = np.array(splev(window, self.pos_tck)).T

        # find & return the closest points
        deltas = path_points - pos
        indx = np.argmin(np.einsum("ij,ij->i", deltas, deltas))
        return floored + indx * projection_step

    def evaluate(self, time: float, state: dict[str, float]) -> dict[str, Any]:
        """
        Evaluates the trajectory based on the vehicle state & time. Checks for the closest point along the path
        to the position read from state, and calculates the setpoint accordingly.

        Args:
            time (float): Current time.
            state (dict[str, float]): The state of the car, consisting of pose, velocity, and yaw rate.

        Returns:
            dict[str, Any]: A dictionary containing information about the trajectory at the point closest to the state.
            The dictionary has the following fields:

                - 'running' (bool): Whether the trajectory has finished.
                - 'ref_pos' (list[float]): The target position.
                - 's0' (list[float]): Base vector of the moving coordinate frame.
                - 'z0' (list[float]): Another base vector of the moving coordinate frame.
                - 's' (float): The path parameter at the target position.
                - 's_ref' (float): The estimated path parameter at the target position.
                - 'v_ref' (float): The target velocity.
        """

        if self.pos_tck is None or self.evol_tck is None:  # check if data has already been provided
            raise ValueError("Trajectory must be defined before evaluation")

        pos = np.array([state["pos_x"], state["pos_y"]])

        # get the path parameter (position along the path)
        s_ref = splev(time, self.evol_tck)
        try:
            s = self.project_to_closest(pos=pos, param_estimate=s_ref, projetion_window=0.5,
                                        projection_step=0.005)  # the projection parameters cound be refined/not hardcoded
        except:
            self.output["running"] = False
            return self.output  # if the projection fails, stop the controller

        # check if the retrievd is satisfies the boundary constraints & the path is not completed
        if time >= self.t_end or s > self.length - 0.01:  # substraction is required because of the projection step
            self.output["running"] = False  # stop the controller
        else:
            self.output["running"] = True  # the goal has not been reached, evaluate the trajectory

        # get path data at the parameter s
        (x, y) = splev(s, self.pos_tck)
        (x_, y_) = splev(s, self.pos_tck, der=1)
        (x__, y__) = splev(s, self.pos_tck, der=2)

        # calculate base vectors of the moving coordinate frame
        s0 = np.array(
            [x_ / np.sqrt(x_ ** 2 + y_ ** 2), y_ / np.sqrt(x_ ** 2 + y_ ** 2)]
        )
        z0 = np.array(
            [-y_ / np.sqrt(x_ ** 2 + y_ ** 2), x_ / np.sqrt(x_ ** 2 + y_ ** 2)]
        )

        # calculate path curvature
        c = abs(x__ * y_ - x_ * y__) / ((x_ ** 2 + y_ ** 2) ** (3 / 2))

        # get speed reference
        v_ref = splev(time, self.evol_tck, der=1)

        self.output["ref_pos"] = np.array([x, y])
        self.output["s0"] = s0
        self.output["z0"] = z0
        self.output["c"] = c

        self.output["s"] = s
        self.output["s_ref"] = s_ref  # might be more effective if output["s"] & self.s are combined
        self.output["v_ref"] = v_ref

        return self.output


class CarTrajectorySpatial(trajectory.Trajectory):
    """
    Class implementation of parametrized BSpline-based trajectories for autonomous ground vehicles
    """
    def __init__(self):
        super().__init__()
        self.output: dict[str, Any] = {}  #: Storage for the last return of evaluate
        self.tck: list = []  #: the spline for x-y-z parametrized by arc length
        self.s_bounds: list[float] = [0, 0]  #: the range of the path parameter (0 to length)
        self.s: float = 0  #: running path parameter
        self.s_ref: float = 0  #: the reference (target) path parameter
        self.start_delay: float = 0  #: the initial start delay
        self.speed_tck: tuple = ()  #: the spline describing the velocity parametrized by the arc length

    def build_from_points_const_speed(self, path_points: np.ndarray, path_smoothing: float, path_degree: int,
                                      const_speed: float, start_delay: float = 0):
        """
        Builds a trajectory using a set of reference waypoints and a constant reference velocity.

        :param path_points: Reference points of the trajectory
        :type path_points: np.ndarray
        :param path_smoothing: Smoothing factor used for the spline interpolation
        :type path_smoothing: float
        :param path_degree: Degree of the fitted Spline
        :type path_degree: int
        :param const_speed: Constant speed of the vehicle
        :type const_speed: float
        :param start_delay: the initial delay of the trajectory
        :type start_delay: float
        """

        self.start_delay = start_delay
        x_points = path_points[:, 0].tolist()
        y_points = path_points[:, 1].tolist()

        if (
                path_points[0, 0] == path_points[-1, 0]
                and path_points[0, 1] == path_points[-1, 1]
        ):
            tck, *rest = splprep([x_points, y_points], k=3, s=path_smoothing)  # closed
        elif len(x_points):
            tck, *rest = splprep([x_points, y_points], k=1, s=path_smoothing)  # line
        else:
            tck, *rest = splprep([x_points, y_points], k=2, s=path_smoothing)  # curve

        u = np.arange(0, 1.001, 0.001)
        path = splev(u, tck)

        (X, Y) = path
        s = np.cumsum(np.sqrt(np.sum(np.diff(np.array((X, Y)), axis=1) ** 2, axis=0)))

        par = np.linspace(0, s[-1], 1001)
        par = np.reshape(par, par.size)

        self.tck, u, *rest = splprep([X, Y], k=path_degree, s=0.1, u=par)

        # set parameter bounds
        self.s_bounds = [u[0], u[-1]]

        # set the running parameter to the start of the path
        self.s = u[0]

        # set reference position too
        self.s_ref = u[0]

        # create spline reference speed
        self.speed_tck = splrep(u, const_speed * np.ones(len(u)), s=0.1)  # currently constant speed profile is given

    def set_trajectory_splines(self, path_tck: tuple, speed_tck: tuple, param_bounds: tuple) -> None:
        """Sets the Spline parameters of the trajectory directly.
           Note that the Spline knot points and coefficients should be in scipy syntax and the Spline should be arc length parametereized

        Args:
            path_tck (tuple): a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline representing the path
            speed_tck (tuple): a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline representing the speed prfile
            param_bounds (tuple): a tuple containing the lower and upper bounds of the parameter; normally (0, arc length) but it can also be shifted e.g. (p, p + arc length)
        """

        self.tck = path_tck
        self.speed_tck = speed_tck
        self.s_bounds = param_bounds
        self.s = param_bounds[0]

    def project_to_closest(self, pos: np.ndarray, param_estimate: float, projetion_window: float,
                           projection_step: float) -> float:
        """
        Projects the vehicle position onto the ginven path and returns the path parameter.
           The path parameter is the curvilinear abscissa along the path as the Bspline that represents the path is arc length parameterized

        Args:
            pos (np.ndarray): Vehicle x,y position
            param_estimate (float): Estimated value of the path parameter
            projetion_window (float): Length of the projection window along the path
            projection_step (float): Precision of the projection in the window

        Returns:
            float: The path parameter
        """

        # calulate the endpoints of the projection window
        floored = clamp(param_estimate - projetion_window / 2, self.s_bounds)
        ceiled = clamp(param_estimate + projetion_window / 2, self.s_bounds)

        # create a grid on the window with the given precision
        window = np.linspace(floored, ceiled, round((ceiled - floored) / projection_step))

        # evaluate the path at each instance
        path_points = np.array(splev(window, self.tck)).T

        # find & return the closest points
        deltas = path_points - pos
        indx = np.argmin(np.einsum("ij,ij->i", deltas, deltas))
        return floored + indx * projection_step

    def evaluate(self, time: float, state: dict[str, float], control_step: float = 1/Car.CTRL_FREQ) -> dict[str, Any]:
        """
        Evaluates the trajectory based on the vehicle state & time. Checks for the closest point along the path
        to the position read from state, and calculates the setpoint accordingly.

        Args:
            time (float): Current time.
            state (dict[str, float]): The state of the car, consisting of pose, velocity, and yaw rate.
            control_step (float): The control step of the car.

            .. note::
                There is no logical reason a trajectory would need to know about the control frequency/step of the
                controller. Will need to examine this

        Returns:
            dict[str, Any]: A dictionary containing information about the trajectory at the point closest to the state.
            The dictionary has the following fields:

                - 'running' (bool): Whether the trajectory has finished.
                - 'ref_pos' (list[float]): The target position.
                - 's0' (list[float]): Base vector of the moving coordinate frame.
                - 'z0' (list[float]): Another base vector of the moving coordinate frame.
                - 's' (float): The path parameter at the target position.
                - 's_ref' (float): The estimated path parameter at the target position.
                - 'v_ref' (float): The target velocity.
        """

        if time < self.start_delay:
            self.output["running"] = False
            return self.output

        if self.tck is None:
            raise ValueError("Trajectory must be defined before evaluation")

        pos = np.array([state["pos_x"], state["pos_y"]])

        # get the path parameter (position along the path)
        s_est = self.s + float(state["long_vel"]) * control_step
        self.s = self.project_to_closest(pos=pos, param_estimate=s_est, projetion_window=2,
                                         projection_step=0.005)  # the projection parameters could be refined/not hardcoded (TODO)

        # check if the retrievd is satisfies the boundary constraints & the path is not completed
        if self.s < self.s_bounds[0] or self.s >= self.s_bounds[
            1] - 0.01:  # substraction is required because of the projection step
            self.output["running"] = False  # stop the controller
        else:
            self.output["running"] = True  # the goal has not been reached, evaluate the trajectory

        # get path data at the parameter s
        (x, y) = splev(self.s, self.tck)
        (x_, y_) = splev(self.s, self.tck, der=1)
        (x__, y__) = splev(self.s, self.tck, der=2)

        # calculate base vectors of the moving coordinate frame
        s0 = np.array(
            [x_ / np.sqrt(x_ ** 2 + y_ ** 2), y_ / np.sqrt(x_ ** 2 + y_ ** 2)]
        )
        z0 = np.array(
            [-y_ / np.sqrt(x_ ** 2 + y_ ** 2), x_ / np.sqrt(x_ ** 2 + y_ ** 2)]
        )

        # calculate path curvature
        c = abs(x__ * y_ - x_ * y__) / ((x_ ** 2 + y_ ** 2) ** (3 / 2))

        # get speed reference
        self.s_ref += splev(self.s, self.speed_tck) * control_step

        self.output["ref_pos"] = np.array([x, y])
        self.output["s0"] = s0
        self.output["z0"] = z0
        self.output["c"] = c

        self.output["s"] = self.s
        self.output["s_ref"] = self.s_ref  # might be more effective if output["s"] & self.s are combined
        self.output["v_ref"] = splev(self.s, self.speed_tck)

        return self.output