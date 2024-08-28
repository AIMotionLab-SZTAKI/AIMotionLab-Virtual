import xml.etree.ElementTree as ET
import mujoco
from typing import Optional, Any, Union
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from aiml_virtual.simulated_object.moving_object import moving_object
from aiml_virtual.controller import controller, drone_geom_controller
from aiml_virtual.trajectory import trajectory, dummy_drone_trajectory


class Propeller:
    DIR_NEGATIVE = -1.0
    DIR_POSITIVE = 1.0

    def __init__(self, direction: float):
        self.ctrl: np.ndarray = np.zeros(1)  # the allocated thrust for the motor
        self.qpos: np.ndarray = np.zeros(1)  # propeller joint coordinate (generalised position)
        self.angle: float = 0  # actual physical propeller angle
        self.actr_force: np.ndarray = np.zeros(1)  # the actual thrust generated
        self.direction: float = direction

    @property
    def dir_float(self) -> float:
        return self.direction

    @property
    def dir_str(self) -> str:
        return "" if self.direction > 0 else "-"

    @property
    def dir_mesh(self) -> str:
        return "ccw" if self.direction > 0 else "cw"


class Drone(moving_object.MovingObject):
    def __init__(self):  # in addition to typing, also check if all of these variables are necessary
        # Todo: idea: make these variables into Properties?
        super().__init__()
        self.sensors: dict[str, np.array] = {}  # storage for sensor data
        self.propellers: list[Propeller] = [Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE),
                                            Propeller(Propeller.DIR_POSITIVE), Propeller(Propeller.DIR_NEGATIVE)]
        self.state: dict[str, np.ndarray] = self.sensors  # just an alias for sensors actually
        self.ctrl_input: np.ndarray = np.zeros(4)  # the output of the controllers, before input allocation

    @property
    @abstractmethod
    def input_matrix(self) -> np.ndarray:
        pass

    @property
    def mass(self) -> Union[None, float, np.array]:
        # property to make sure that mass gets updated in case the model is updated. I haven't decided whether
        # to leave it as a numpy array of length 1 (which is what model.body().mass returns), or extract the float
        if self.model:
            return self.model.body(self.name).mass
        else:
            return None

    @property
    def inertia(self) -> Union[None, np.ndarray]:
        # property to make sure that inertia gets updated whenever the model is updated
        if self.model:
            return self.model.body(self.name).inertia
        else:
            return None

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        # returning None opts out of XML parsing
        return None

    @abstractmethod
    def set_default_controller(self) -> None:
        pass

    def spin_propellers(self) -> None:
        if self.sensors["pos"][2] > 0.015:
            for propeller in self.propellers:
                propeller.angle += propeller.dir_float * propeller.ctrl[0] * 100
                propeller.qpos[0] = propeller.angle

    # todo: types
    def update(self, mj_step_count: int, step: float) -> None:
        # todo: check this as compared to the original when cleaning up
        self.spin_propellers()
        if self.trajectory:
            setpoint = self.trajectory.evaluate(self.data.time)
            self.ctrl_input = self.controller.compute_control(state=self.state, setpoint=setpoint)
            motor_thrusts = self.input_matrix @ self.ctrl_input
            for propeller, thrust in zip(self.propellers, motor_thrusts):
                propeller.ctrl[0] = thrust

    def bind_to_data(self, data: mujoco.MjData) -> None:
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
            propeller.actr_force = self.data.actuator(f"{self.name}_actr{i}").force





