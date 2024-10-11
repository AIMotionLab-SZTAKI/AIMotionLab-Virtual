import numpy as np
from typing import Union

from aiml_virtual.controller import controller

class CarLPVController(controller.Controller):
    def __init__(self, mass: float, inertia: Union[tuple, np.ndarray, list], long_gains: Union[np.ndarray, None] = None,
                 lat_gains: Union[np.ndarray, None] = None, disturbed: bool = False, **kwargs):
        """Trajectory tracking LPV feedback controller, based on the decoupled longitudinal and lateral dynamics

        Args:
            mass (float): Vehicle mass
            inertia (float | np.ndarray | list): Vehicle inertia
            long_gains (np.ndarray | None, optional): Polynomial coefficients of the longitudinal controller gains. Defaults to None.
            lat_gains (np.ndarray | None, optional): Polynomila coefficients of the lateral controller gains. Defaults to None.
            disturbed (bool, optional): Apply some kind of disturbance to test adaptive trajectory planner of hooked drone. Defaults to False.

            Additionally drivetrain parameters (C_m1, C_m2, C_m3) can be updated through keyword arguments.
        """
        super().__init__()
        # get mass/intertia & additional vehicle params
        self.m = mass
        self.I_z = inertia[2]  # only Z-axis inertia is needed
        self.disturbed = disturbed

        # set additional vehicle params or default to predefined values
        try:
            self.C_m1 = kwargs["C_m1"]
        except KeyError:
            self.C_m1 = 52.4282
        try:
            self.C_m2 = kwargs["C_m2"]
        except KeyError:
            self.C_m2 = 5.2465
        try:
            self.C_m3 = kwargs["C_m3"]
        except KeyError:
            self.C_m3 = 1.119465

        try:
            self.dt = kwargs["control_step"]
        except KeyError:
            self.dt = 1.0 / 40  # standard frequency for which the controllers are designed

        if long_gains is not None:  # gains are specified
            self.k_long1 = np.poly1d(long_gains[0, :])
            self.k_long2 = np.poly1d(long_gains[1, :])

        else:  # no gains specified use the default
            self.k_long1 = np.poly1d([0.0010, -0.0132, 0.4243])
            self.k_long2 = np.poly1d([-0.0044, 0.0563, 0.0959])

        if lat_gains is not None:
            self.k_lat1 = np.poly1d(lat_gains[0, :])
            self.k_lat2 = np.poly1d(lat_gains[1, :])
            self.k_lat3 = np.poly1d(lat_gains[2, :])
        else:
            self.k_lat1 = np.poly1d([0.00127, -0.00864, 0.0192, 0.0159])
            self.k_lat2 = np.poly1d([0.172, -1.17, 2.59, 5.14])
            self.k_lat3 = np.poly1d([0.00423, 0.0948, 0.463, 0.00936])

        self.C_f = 41.7372
        self.C_r = 29.4662
        self.l_f = 0.163
        self.l_r = 0.168

        # define the initial value of the lateral controller integrator
        self.q = 0  # lateral controller integrator

    def compute_control(self, state: dict, setpoint: dict, time, **kwargs) -> tuple[float, float]:
        """Method for calculating the control input, based on the current state and setpoints

        Args:
            state (dict): Dict containing the state variables
            setpoint (dict): Setpoint determined by the trajectory object
            time (float): Current simuator time

        Returns:
            np.array: Computed control inputs [d, delta]
        """

        # check if the the trajectory exectuion is still needed
        if not setpoint["running"]:
            return 0, 0

        # retrieve setpoint & state data
        s0 = setpoint["s0"]
        z0 = setpoint["z0"]
        ref_pos = setpoint["ref_pos"]
        c = setpoint["c"]
        s = setpoint["s"]
        s_ref = setpoint["s_ref"]
        v_ref = setpoint["v_ref"]

        pos = np.array([state["pos_x"], state["pos_y"]])
        phi = state["head_angle"]
        v_xi = state["long_vel"]
        v_eta = state["lat_vel"]

        beta = np.arctan2(v_eta, abs(v_xi))  # abs() needed for reversing

        theta_p = np.arctan2(s0[1], s0[0])

        # lateral error
        z1 = np.dot(pos - ref_pos, z0)

        # heading error
        theta_e = self._normalize(phi - theta_p)

        # longitudinal model parameter
        p = abs(np.cos(theta_e + beta) / np.cos(beta) / (1 - c * z1))

        # invert z1 for lateral dynamics:
        e = -z1
        self.q += e
        self.q = self._clamp(self.q, 0.1)

        # estimate error derivative
        try:
            self.edot = 0.5 * (
                        (e - self.ep) / self.dt - self.edot) + self.edot  # calculate \dot{e} by finite difference
            self.ep = e  # 0.5 coeff if used for smoothing
        except AttributeError:  # if no previous error value exist assume 0 & store the current value
            self.edot = 0
            self.ep = e

        # compute control inputss

        delta = -theta_e + self.k_lat1(v_xi) * self.q + self.k_lat2(v_xi) * e + self.k_lat3(v_xi) * self.edot \
                - self.m / self.C_f * ((self.l_r * self.C_r - self.l_f * self.C_f) / self.m - 1) * c

        if self.disturbed:
            C_m2 = 2
            C_m3 = 1
        else:
            C_m2 = self.C_m2
            C_m3 = self.C_m3

        d = (C_m2 * v_ref / p + C_m3 * np.sign(v_ref)) / self.C_m1 - self.k_long1(p) * (s - s_ref) - self.k_long2(p) * (
                    v_xi - v_ref)

        # clamp control inputs into the feasible range
        d = self._clamp(d, (0, 0.25))  # currently only forward motion, TODO: reversing control
        delta = self._clamp(delta, (-.5, .5))

        return d, delta

    @staticmethod
    def _clamp(value: Union[float, int], bound: Union[int, float, list, tuple, np.ndarray]) -> float:
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
        elif isinstance(bound, tuple) or isinstance(bound, list) or isinstance(bound, np.ndarray):
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
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi

        return angle

