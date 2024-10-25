"""
This module implements an LPV feedback controller for the Car, originally written by
Prof. Dr. PhD Mr. Floch Kristóf I believe.

.. note::
       The classes in this module, as well as the car trajectories were carried over from
       the previous aiml_virtual with as minimal changes as possible. I was not
       comfortable enough with the code and the working of the
       car to refactor it with confidence. Because of this, the docstrings and
       comments in this module (and the car_controller.py) are rather sparse
       and the code is in dire need of refactoring.

.. todo::
    Refactor this module.
"""

import numpy as np
from typing import Union

from aiml_virtual.controller import controller
from aiml_virtual.simulated_object.dynamic_object.controlled_object import car
from aiml_virtual.utils import utils_general

normalize = utils_general.normalize
clamp = utils_general.clamp

class CarLPVController(controller.Controller):
    def __init__(self, mass: np.ndarray, inertia: Union[tuple, np.ndarray, list],
                 long_gains: np.ndarray = np.array([[0.0010, -0.0132, 0.4243], [-0.0044, 0.0563, 0.0959]]),
                 lat_gains: np.ndarray = np.array([[0.00127, -0.00864, 0.0192, 0.0159], [0.172, -1.17, 2.59, 5.14], [0.00423, 0.0948, 0.463, 0.00936]]),
                 disturbed: bool = False, **kwargs):
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
        self.m: np.ndarray = mass  #: Mass of the vehicle.
        self.I_z: float = inertia[2]  #: Z-axis inertia of the vehicle.
        self.disturbed: bool = disturbed  # Whether disturbances are applied.

        # set additional vehicle params or default to predefined values
        self.C_m1: float = car.Car.C_M1 if "C_m1" not in kwargs else kwargs["C_m1"]  #: lumped drivetrain parameters 1
        self.C_m2: float = car.Car.C_M2 if "C_m2" not in kwargs else kwargs["C_m2"]  #: lumped drivetrain parameters 2
        self.C_m3: float = car.Car.C_M3 if "C_m3" not in kwargs else kwargs["C_m3"]  #: lumped drivetrain parameters 3

        self.dt: float = 1/car.Car.CTRL_FREQ if "control_step" not in kwargs else kwargs["control_step"]  #: controller tipestep

        self.k_long1: np.poly1d = np.poly1d(long_gains[0, :])  #: longitude gains 1
        self.k_long2: np.poly1d = np.poly1d(long_gains[1, :])  #: longitude gains 2

        self.k_lat1: np.poly1d = np.poly1d(lat_gains[0, :])  #: latitude gains 1
        self.k_lat2: np.poly1d = np.poly1d(lat_gains[1, :])  #: latitude gains 2
        self.k_lat3: np.poly1d = np.poly1d(lat_gains[2, :])  #: latitude gains 3

        self.C_f = car.Car.C_F  #: Cornering stiffness of front tire
        self.C_r = car.Car.C_R  #: Cornering stiffness of the rear tire
        self.l_f = car.Car.L_F  #: Distance of the front axis from the center of mass
        self.l_r = car.Car.L_R  #: Distance of the rear axis from the center of mass

        # define the initial value of the lateral controller integrator
        self.q = 0  # lateral controller integrator

    def compute_control(self, state: dict, setpoint: dict, time, **kwargs) -> tuple[float, float]:
        """
        Overrides (implements) superclass' compture_control, ensuring CarLPVCOntroller is a complete class.
        Other than the required positional arguments, additional keyword arguments may be supplies.

        Args:
            state (dict): Dict containing the state variables.
            setpoint (dict): Setpoint determined by the trajectory object.
            time (float): Current simuator time.

        Returns:
            tuple[float, float]: Computed control inputs [d, delta]
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
        theta_e = normalize(phi - theta_p)

        # longitudinal model parameter
        p = abs(np.cos(theta_e + beta) / np.cos(beta) / (1 - c * z1))

        # invert z1 for lateral dynamics:
        e = -z1
        self.q += e
        self.q = clamp(self.q, 0.1)

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
        d = clamp(d, (0, 0.25))  # currently only forward motion, TODO: reversing control
        delta = clamp(delta, (-.5, .5))

        return d, delta



