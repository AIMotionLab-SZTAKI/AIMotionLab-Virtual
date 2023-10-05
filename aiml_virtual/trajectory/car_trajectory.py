from typing import Union
import numpy as np
from scipy.interpolate import splrep, splprep, splev
import matplotlib.pyplot as plt
from aiml_virtual.trajectory.trajectory_base import TrajectoryBase


class CarTrajectory(TrajectoryBase):
    def __init__(self):
        """Class implementation of BSpline-based trajectories for autonomous ground vehicles
        """
        super().__init__()

        self._tck = None # BSpline vector knots/coeffs/degree
        self.s_bounds = None # path parameter vector
        self.s=None # running path parameter


    def build_from_points_const_speed(self, path_points: np.ndarray, path_smoothing: float, path_degree: int, const_speed: float):
        """Object responsible for storing the reference trajectory data.

        Args:
            path_points (numpy.ndarray): Reference points of the trajectory
            smoothing (float): Smoothing factor used for the spline interpolation
            degree (int): Degree of the fitted Spline
        """

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

        self._tck, u, *rest = splprep([X, Y], k=path_degree, s=0.1, u=par)

        # set parameter bounds
        self.s_bounds=[u[0], u[-1]]
    
        # set the running parameter to the start of the path
        self.s=u[0]

        # set reference position too
        self.s_ref=u[0]

        # create spline reference speed
        self._speed_tck = splrep(u, const_speed*np.ones(len(u)), s=0.1) # currently constant speed profile is given


    def set_trajectory_splines(self, path_tck: tuple, speed_tck: tuple, param_bounds: tuple) -> None:
        """Sets the Spline parameters of the trajectory directly. 
           Note that the Spline knot points and coefficients should be in scipy syntax and the Spline should be arc length parametereized

        Args:
            path_tck (tuple): a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline representing the path
            speed_tck (tuple): a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline representing the speed prfile
            param_bounds (tuple): a tuple containing the lower and upper bounds of the parameter; normally (0, arc length) but it can also be shifted e.g. (p, p + arc length)
        """

        self._tck=path_tck
        self._speed_tck=speed_tck
        self.s_bounds=param_bounds
        self.s=param_bounds[0]


    def _project_to_closest(self, pos: np.ndarray, param_estimate: float, projetion_window: float, projection_step: float) -> float:
        """Projects the vehicle position onto the ginven path and returns the path parameter.
           The path parameter is the curvilinear abscissa along the path as the Bspline that represents the path is arc length parameterized

        Args:
            pos (np.ndarray): Vehicle x,y position
            param_esimate: Estimated value of the path parameter
            projetion_window (float): Length of the projection window along the path
            projection_step (float): Precision of the projection in the window

        Returns:
            float: The path parameter
        """

        
        # calulate the endpoints of the projection window
        floored = self._clamp(param_estimate - projetion_window / 2, self.s_bounds)
        ceiled = self._clamp(param_estimate + projetion_window/2, self.s_bounds)


        # create a grid on the window with the given precision
        window = np.linspace(floored, ceiled, round((ceiled - floored) / projection_step))


        # evaluate the path at each instance
        path_points = np.array(splev(window, self._tck)).T

        # find & return the closest points
        deltas = path_points - pos
        indx = np.argmin(np.einsum("ij,ij->i", deltas, deltas))
        return floored + indx * projection_step



    def evaluate(self, state, i, time, control_step) -> dict:
        """Evaluates the trajectory based on the vehicle state & time"""
        if self._tck is None:
            raise ValueError("Trajectory must be defined before evaluation")
        
        pos=np.array([state["pos_x"],state["pos_y"]])

        # get the path parameter (position along the path)
        s_est=self.s+float(state["long_vel"])*control_step
        self.s=self._project_to_closest(pos=pos, param_estimate=s_est, projetion_window=2, projection_step=0.005) # the projection parameters cound be refined/not hardcoded (TODO)

        # check if the retrievd is satisfies the boundary constraints & the path is not completed
        if self.s < self.s_bounds[0] or self.s >= self.s_bounds[1]-0.01: # substraction is required because of the projection step
            self.output["running"]=False # stop the controller
        else:
            self.output["running"]= True # the goal has not been reached, evaluate the trajectory

        # get path data at the parameter s
        (x, y) = splev(self.s, self._tck)
        (x_, y_) = splev(self.s, self._tck, der=1)
        (x__,y__) = splev(self.s, self._tck,der=2)
        
        # calculate base vectors of the moving coordinate frame
        s0 = np.array(
            [x_ / np.sqrt(x_**2 + y_**2), y_ / np.sqrt(x_**2 + y_**2)]
        )
        z0 = np.array(
            [-y_ / np.sqrt(x_**2 + y_**2), x_ / np.sqrt(x_**2 + y_**2)]
        )

        # calculate path curvature
        c=abs(x__*y_-x_*y__)/((x_**2+y_**2)**(3/2))

        # get speed reference
        self.s_ref += splev(self.s, self._speed_tck)*control_step

        self.output["ref_pos"]=np.array([x,y])
        self.output["s0"] = s0
        self.output["z0"] = z0
        self.output["c"] = c

        self.output["s"] = self.s
        self.output["s_ref"]=self.s_ref # might be more effective if output["s"] & self.s are combined
        self.output["v_ref"] = splev(self.s, self._speed_tck)

        return self.output
    


    @staticmethod
    def _clamp(value: Union[float,int], bound: Union[int,float,list,tuple,np.ndarray]) -> float:
        """Helper function that clamps the given value with the specified bounds

        Args:
            value (float | int): The value to clamp
            bound (list | tuple | np.ndarray): If int | float the function constrains the value into [-bound,bound]
                                               If tuple| list | np.ndarray the value is constained into the range of [bound[0],bound[1]]

        Returns:
            float: The clamped value
        """
        if isinstance(bound, int) or isinstance(bound, float):
            if value < -bound:
                return float(-bound)
            elif value > bound:
                return float(bound)
            return float(value)
        elif isinstance(bound, tuple) or isinstance(bound,list) or isinstance(bound, np.ndarray):
            if value < bound[0]:
                return float(bound[0])
            elif value > bound[1]:
                return float(bound[1])
            return float(value)
    

    @staticmethod
    def _normalize(angle: float) -> float:
        """Normalizes the given angle into the [-pi/2, pi/2] range

        Args:
            angle (float): Input angle

        Returns:
            float: Normalized angle
        """
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi

        return angle
    
    def plot_trajectory(self) -> None:
        """ Plots, the defined path of the trajectory in the X-Y plane. Nota, that this function interrupts the main thread and the simulator!
        """

        if self._tck is None: # check if data has already been provided
            raise ValueError("No Spline trajectory is specified!")
        
        # evaluate the path between the bounds and plot
        (x,y)=splev(np.linspace(self.s_bounds[0], self.s_bounds[1], 100), self._tck)
        plt.plot(x,y)
        plt.show()