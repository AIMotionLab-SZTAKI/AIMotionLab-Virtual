from typing import Union, Any
import numpy as np
from scipy.interpolate import splrep, splprep, splev
from scipy.integrate import quad

from aiml_virtual.trajectory import trajectory

class CarTrajectory(trajectory.Trajectory):
    def __init__(self, trajectory_ID="") -> None:
        """Class implementation of BSpline-based trajectories for autonomous ground vehicles
        """
        super().__init__()
        self.trajectory_ID = trajectory_ID
        self.output = {}
        self.pos_tck = None
        self.evol_tck = None
        self.t_end = None
        self.length = None
        self.reversed = None

    def build_from_points_smooth_const_speed(self, path_points: np.ndarray, path_smoothing: float, path_degree: int,
                                             virtual_speed: float):
        """Builds a trajectory using a set of reference waypoints and a virtual reference velocity with smoothed start and end reference

        :param path_points: Reference points of the trajectory
        :type path_points: np.ndarray
        :param path_smoothing: Smoothing factor used for the spline interpolation
        :type path_smoothing: float
        :param path_degree: Degree of the fitted Spline
        :type path_degree: int
        :param virtual_speed: The virtual constant speed for the vehicle. In practice the designer smoothens the start and end of the trajectory
        :type virsual_speed: float

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
                                      const_speed: float):
        """Object responsible for storing the reference trajectory data.

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

    def export_to_time_dependent(self):
        """Exports the trajectory to a time dependent representation
        """
        t_eval = np.linspace(0, self.t_end, 100)

        s = splev(t_eval, self.evol_tck)
        (x, y) = splev(s, self.pos_tck)

        tck, t, *_ = splprep([x, y], k=5, s=0, u=t_eval)

        return tck

    def _project_to_closest(self, pos: np.ndarray, param_estimate: float, projetion_window: float,
                            projection_step: float) -> float:
        """Projects the vehicle position onto the ginven path and returns the path parameter.
           The path parameter is the curvilinear abscissa along the path as the Bspline that represents the path is arc length parameterized


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
        floored = self._clamp(param_estimate - projetion_window / 2, [0, self.length])
        ceiled = self._clamp(param_estimate + projetion_window / 2, [0, self.length])

        # create a grid on the window with the given precision
        window = np.linspace(floored, ceiled, round((ceiled - floored) / projection_step))

        # evaluate the path at each instance
        path_points = np.array(splev(window, self.pos_tck)).T

        # find & return the closest points
        deltas = path_points - pos
        indx = np.argmin(np.einsum("ij,ij->i", deltas, deltas))
        return floored + indx * projection_step

    def evaluate(self, time: float, state) -> dict[str, Any]:
        """Evaluates the trajectory based on the vehicle state & time

        :param state: Vehicle state
        :type state: dict
        :param time: Current time
        :type time: float
        """

        if self.pos_tck is None or self.evol_tck is None:  # check if data has already been provided
            raise ValueError("Trajectory must be defined before evaluation")

        pos = np.array([state["pos_x"], state["pos_y"]])

        # get the path parameter (position along the path)
        s_ref = splev(time, self.evol_tck)
        try:
            s = self._project_to_closest(pos=pos, param_estimate=s_ref, projetion_window=0.5,
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

    @staticmethod
    def _clamp(value: Union[float, int], bound: Union[int, float, list, tuple, np.ndarray]) -> float:
        """Helper function that clamps the given value with the specified bounds


        :param value: The value to clamp
        :type value: Union[float,int]
        :param bound: If int | float the function constrains the value into [-bound,bound]
                      If tuple| list | np.ndarray the value is constained into the range of [bound[0],bound[1]]
        :type bound: Union[int,float,list,tuple,np.ndarray]
        :return: The clamped value
        :rtype: float
        """
        if isinstance(bound, int) or isinstance(bound, float):
            if value < -bound:
                return float(-bound)
            elif value > bound:
                return float(bound)
            return float(value)
        elif isinstance(bound, tuple) or isinstance(bound, list) or isinstance(bound, np.ndarray):
            if value < bound[0]:
                return float(bound[0])
            elif value > bound[1]:
                return float(bound[1])
            return float(value)

    @staticmethod
    def _normalize(angle: float) -> float:
        """Normalizes the given angle into the [-pi/2, pi/2] range

        :param angle: Input angle
        :type angle: float
        :return: Normalized angle
        :rtype: float
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi

        return angle

    def get_traj(self):
        t_eval = np.linspace(0, self.t_end, 100)

        s = splev(t_eval, self.evol_tck)
        (x, y) = splev(s, self.pos_tck)

        return x, y