"""
This module contains classes relating to the general drone model.
"""

import mujoco
from typing import Optional
import numpy as np
from abc import abstractmethod

from aiml_virtual.simulated_object.dynamic_object.controlled_object import controlled_object


class Propeller:
    """
    Class that encapsulates most of the data relating to a Propeller in mujoco.
    """
    DIR_NEGATIVE: float = -1.0  #: **classvar** | Clockwise (left-hand) spin direction.
    DIR_POSITIVE: float = 1.0  #: **classvar** | Counter-clockwise (right-hand) spin direction.

    def __init__(self, direction: float, spin_speed: float = 0.08):
        self.ctrl: np.ndarray = np.zeros(1)  #: The allocated control input (target thrust, N) for the motor.
        self.qpos: np.ndarray = np.zeros(1)  #: Propeller joint coordinate (generalised position).
        self.angle: float = 0  #: Actual physical propeller angle.
        self.direction: float = direction  #: The direction of spin.
        self.spin_speed = spin_speed  #: The speed of the spin in the animation.

    @property
    def dir_float(self) -> float:
        """
        Property to grab the spin direction as a float for calculations.
        """
        return self.direction

    @property
    def dir_str(self) -> str:
        """
        Property to grab the spin direction as a string for XML creation.
        """
        return "" if self.direction > 0 else "-"

    @property
    def dir_mesh(self) -> str:
        """
        Property to grab the spin direction in terms of cw/ccw for XML creation.
        """
        return "ccw" if self.direction > 0 else "cw"

    def spin(self) -> None:
        """
        Spins the propellers in the mujoco display (but has no physical effect).

        .. note::
            Here used to be a code segment that spun the propeller proportionally to the actuator output, but personally
            I think it looked a bit weird.

            .. code-block:: python

                self.angle += self.dir_float * self.ctrl[0] * self.spin_speed
                self.qpos[0] = self.angle

            I think it looks more natural if the propellers simply spin at an even rate, when looked at via the naked
            eye we can't tell whether the spin rate is proportional to the thrust anyway.
        """
        self.angle += self.spin_speed
        self.qpos[0] = self.angle

class Drone(controlled_object.ControlledObject):
    """
    Class encapsulating behaviour common to all drones (which are ControlledObjects).
    """

    def __init__(self):
        super().__init__()
        self.sensors: dict[str, np.ndarray] = {}  #: Dictionary of sensor data.
        self.propellers: list[Propeller] = [Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE),
                                            Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE)] #: List of propellers; first and third are ccw, second and fourth are cw
        self.state: dict[str, np.ndarray] = self.sensors  #: Convenience alias for sensors.
        self.ctrl_output: np.ndarray = np.zeros(4)  #: Output of the controllers, usually force-torque(x3).

    @property
    @abstractmethod
    def input_matrix(self) -> np.ndarray:
        """
        Property to grab the input matrix for use in input allocation: it shows the connection between the control
        outputs (thrust-toruqeX-torqueY-torqueZ) and the individual motor thrusts.
        """
        pass

    @property
    def prop_offset(self) -> list[np.ndarray]:
        """Property to grab the position of the propellers relative to the drone's center."""
        return [self.model.joint(f"{self.name}_prop{i}").pos for i in range(4)]

    @property
    def prop_vel(self) -> np.ndarray:
        """
        Property to estimate propeller angular velocity.
        """
        raise NotImplementedError

    def spin_propellers(self) -> None:
        """
        Updates the display of the propellers, to make it look like they are spinning.
        """
        if self.sensors["pos"][2] > 0.1:  # only start spinning if the drone has taken flight
            for propeller in self.propellers:
                propeller.spin()

    def update(self) -> None:
        """
        Overrides SimulatedObject.update. Updates the position of the propellers to make it look like they are
        spinning, and runs the controller.
        """
        # : check this as compared to the original when cleaning up
        self.spin_propellers()  # update how the propellers look
        # if we don't have a trajectory, we don't have a reference for the controller: skip
        if self.trajectory is not None and self.controller is not None:
            setpoint = self.trajectory.evaluate(self.data.time)
            self.ctrl_output = self.controller.compute_control(state=self.state, setpoint=setpoint)
            motor_thrusts = self.input_matrix @ self.ctrl_output
            for propeller, thrust in zip(self.propellers, motor_thrusts):
                propeller.ctrl[0] = thrust

    def bind_to_data(self, data: mujoco.MjData) -> None:
        """
        Overrides SimulatedObstacle.bind_to_data. Saves the references of the sensors and propellers from the data.
        Also sets a default controller if one has not been set yet.

        Args:
             data (mujoco.MjData): The data of the simulation (as opposed to the *model*).
        """
        if self.model is None:
            raise RuntimeError
        if self.controller is None:
            self.set_default_controller()
        self.data = data
        self.sensors["ang_vel"] = self.data.sensor(self.name + "_gyro").data
        self.sensors["vel"] = self.data.sensor(self.name + "_velocimeter").data
        self.sensors["acc"] = self.data.sensor(self.name + "_accelerometer").data
        self.sensors["pos"] = self.data.sensor(self.name + "_posimeter").data
        self.sensors["quat"] = self.data.sensor(self.name + "_orimeter").data
        self.sensors["ang_acc"] = self.data.sensor(self.name + "_ang_accelerometer").data
        for i, propeller in enumerate(self.propellers):
            prop_joint = self.data.joint(f"{self.name}_prop{i}")
            propeller.qpos = prop_joint.qpos
            propeller.angle = propeller.qpos[0]
            propeller.ctrl = self.data.actuator(f"{self.name}_actr{i}").ctrl





