from typing import Optional
import mujoco
import xml.etree.ElementTree as ET
import numpy as np
import math

from aiml_virtual.simulated_object.dynamic_object.controlled_object import controlled_object
from aiml_virtual.controller import car_lpv_controller
from aiml_virtual.utils import utils_general

class Wheel:
    RADIUS: str = ".052388"
    WIDTH: str = ".022225"
    SIZE: str = RADIUS + " " + WIDTH
    ARMATURE: str = "0.05"
    ARMATURE_STEER: str = "0.001"
    FRIC_STEER: str = "0.2"
    DAMP_STEER: str = "0.2"
    DAMPING: str = "0.00001"
    FRICTIONLOSS: str = "0.01"
    STEER_RANGE: str = "-0.6 0.6"
    FRICTION: str = "2.5 2.5 .009 .0001 .0001"
    KP: str = "15"
    MASS: str = "0.1"
    FRONT_LEFT: str = "_wheelfr"
    FRONT_RIGHT: str = "_wheelfl"
    REAR_LEFT: str = "_wheelrl"
    REAR_RIGHT: str = "_wheelrr"
    def __init__(self, name: str, pos: str):
        self.name: str = name
        self.pos: str = pos
        self.ctrl: np.ndarray = np.zeros(1)
        self.actr_force: np.ndarray = np.zeros(1)
        self.ctrl_steer: Optional[np.ndarray] = np.zeros(1) if "_wheelfr" in name or "_wheelfl" in name else None
        self.actr_force_steer: Optional[np.ndarray] = np.zeros(1) if "_wheelfr" in name or "_wheelfl" in name else None

class Car(controlled_object.ControlledObject):
    DIAGINERTIA: str = ".05 .05 .08"
    MASS: str = "3.0"
    WHEEL_X: float = 0.16113
    WHEEL_Y: float = 0.122385
    STEER_Y: float = 0.10016
    MAX_TORQUE: float = 2.0
    C_M1: float = 52.4282  #: lumped drivetrain parameters 1
    C_M2: float = 5.2465  #: lumped drivetrain parameters 2
    C_M3: float = 1.119465  #: lumped drivetrain parameters 3
    C_F: float = 41.7372  #: Cornering stiffness of front tire
    C_R: float = 29.4662  #: Cornering stiffness of the rear tire
    L_F: float = 0.163  #: Distance of the front axis from the center of mass
    L_R: float = 0.168  #: Distance of the rear axis from the center of mass
    CTRL_FREQ: float = 40.0

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["Car"]

    def __init__(self):
        super().__init__()
        self.update_frequency = Car.CTRL_FREQ
        self.wb: float = 2*Car.WHEEL_Y
        self.tw: float = 2*Car.WHEEL_X
        self.sensors: dict[str, np.array] = {}  #: Dictionary of sensor data.
        self.wheels: dict[str, Wheel] = {
            "front_left": Wheel(self.name+Wheel.FRONT_LEFT, pos=f"{Car.WHEEL_X} {Car.WHEEL_Y} 0"),
            "rear_left": Wheel(self.name+Wheel.REAR_LEFT, pos=f"-{Car.WHEEL_X} {Car.WHEEL_Y} 0"),
            "front_right": Wheel(self.name+Wheel.FRONT_RIGHT, pos=f"{Car.WHEEL_X} -{Car.WHEEL_Y} 0"),
            "rear_right": Wheel(self.name+Wheel.REAR_RIGHT, pos=f"-{Car.WHEEL_X} -{Car.WHEEL_Y} 0")
        }

    @property
    def state(self) -> dict[str, float]:
        roll, pitch, yaw = utils_general.euler_from_quaternion(*self.sensors["quat"])
        state = {"pos_x": self.sensors["pos"][0],
                 "pos_y": self.sensors["pos"][1],
                 "head_angle": yaw,
                 "long_vel": self.sensors["vel"][0],
                 "lat_vel": self.sensors["vel"][1],
                 "yaw_rate": self.sensors["ang_vel"][2]}
        return state

    def set_steer_angles(self, delta_in: float):
        num = self.wb * math.tan(delta_in)
        delta_left = math.atan(num / (self.wb + (0.5 * self.tw * math.tan(delta_in))))
        delta_right = math.atan(num / (self.wb - (0.5 * self.tw * math.tan(delta_in))))
        self.wheels["front_left"].ctrl_steer[0] = delta_left
        self.wheels["front_right"].ctrl_steer[0] = delta_right

    def set_torques(self, d: float):
        v = self.sensors["vel"][0]
        torque = Car.C_M1*d - Car.C_M2*v - Car.C_M3*np.sign(v)
        for wheel in self.wheels.values():
            wheel.ctrl[0] = utils_general.clamp(torque, Car.MAX_TORQUE)

    def update(self) -> None:
        if self.trajectory is not None and self.controller is not None:
            setpoint = self.trajectory.evaluate(self.data.time, self.state)
            d, delta_in = self.controller.compute_control(self.state, setpoint, self.data.time)
            self.set_steer_angles(delta_in)
            self.set_torques(d)

    def set_default_controller(self) -> None:
        self.controller = car_lpv_controller.CarLPVController(self.mass, self.inertia)

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        # for now without trailer and hook
        site_name = f"{self.name}_cog"
        car = ET.Element("body", name=self.name, pos=pos, quat=quat)  # top level body with free joint
        ret = {"worldbody" : [car],
               "actuator": [],
               "sensor": [],
               "contact": []}
        ET.SubElement(car, "inertial", pos="0 0 0", diaginertia=Car.DIAGINERTIA, mass=Car.MASS)
        ET.SubElement(car, "joint", name=self.name, type="free")
        ET.SubElement(car, "site", name=site_name, pos="0 0 0")
        ######################### ADDING ALL THE GEOMS THAT MAKE UP THE MAIN BODY OF THE CAR ###########################
        ET.SubElement(car, "geom", name=self.name + "_chassis_b", type="box", size=".10113 .1016 .02", pos="-.06 0 0", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_chassis_f", type="box", size=".06 .07 .02", pos=".10113 0 0", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_front", type="box", size=".052388 .02 .02", pos=".2135 0 0", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_back", type="box", size=".062488 .02 .02", pos="-.2236 0 0", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_front_bumper", type="box", size=".005 .09 .02", pos=".265888 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_back_bumper", type="box", size=".005 .08 .02", pos="-.2861 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_number", type="cylinder", size=".01984 .03", pos=".12 0 .05", rgba="0.1 0.1 0.1 1.0")
        ET.SubElement(car, "geom", name=self.name + "_camera", type="box", size=".012 .06 0.02", pos=".18 0 .08")
        ET.SubElement(car, "geom", name=self.name + "_camera_holder", type="box", size=".012 .008 .02", pos=".18 0 .04")
        ET.SubElement(car, "geom", name=self.name + "_circuits", type="box", size=".08 .06 .03", pos="-.05 0 .05", rgba=color)
        ET.SubElement(car, "geom", name=self.name + "_antennal", type="box", size=".007 .004 .06", pos="-.16 -.01 .105", euler="0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=self.name + "_antennar", type="box", size=".007 .004 .06", pos="-.16 .01 .105", euler="-0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=self.name + "_antenna_holder", type="box", size=".008 .008 .02", pos="-.16 0 .04", rgba=".1 .1 .1 1.0")
        ################################################ WHEELS ########################################################
        for wheel in self.wheels.values():
            # note to self: the order of elements in a mjcf file matters!
            wheelbody = ET.SubElement(car, "body", name=wheel.name)
            if wheel.ctrl_steer is not None and wheel.actr_force_steer is not None:
                pos = wheel.pos.split(" ")
                y = np.sign(float(pos[1]))*Car.STEER_Y
                ET.SubElement(wheelbody, "joint", name=wheel.name+"_steer", type="hinge",
                              pos = f"{pos[0]} {y} {pos[2]}",limited="true",
                              frictionloss=Wheel.FRIC_STEER, damping=Wheel.DAMP_STEER, armature=Wheel.ARMATURE_STEER,
                              range=Wheel.STEER_RANGE, axis="0 0 1")
                ret["actuator"].append(ET.Element("position", forcelimited="true", forcerange="-5 5",
                                                  name=wheel.name+"_actr_steer", joint=wheel.name+"_steer", kp=Wheel.KP))
            ET.SubElement(wheelbody, "joint", name=wheel.name, type="hinge", pos=wheel.pos, axis="0 1 0",
                          frictionloss=Wheel.FRICTIONLOSS, damping=Wheel.DAMPING, armature=Wheel.ARMATURE,
                          limited="false")  # wheel rotational joint
            ET.SubElement(wheelbody, "geom", name=wheel.name, type="cylinder", size=Wheel.SIZE, pos=wheel.pos,
                          mass=Wheel.MASS, material="material_check", euler="1.571 0 0")
            ret["contact"].append(ET.Element("pair", geom1=wheel.name, geom2="ground", condim="6", friction=Wheel.FRICTION))
            ret["actuator"].append(ET.Element("motor", name=wheel.name+"_actr", joint=wheel.name))

        ################################################ SENSORS #######################################################
        ret["sensor"].append(ET.Element("gyro", site=site_name, name=self.name + "_gyro"))
        ret["sensor"].append(ET.Element("velocimeter", site=site_name, name=self.name + "_velocimeter"))
        ret["sensor"].append(ET.Element("framepos", objtype="site", objname=site_name, name=self.name + "_posimeter"))
        ret["sensor"].append(ET.Element("framequat", objtype="site", objname=site_name, name=self.name + "_orimeter"))
        return ret

    def bind_to_data(self, data: mujoco.MjData) -> None:
        if self.model is None:
            raise RuntimeError
        if self.controller is None:
            self.set_default_controller()
            self.data = data
            self.sensors["ang_vel"] = self.data.sensor(self.name + "_gyro").data
            self.sensors["vel"] = self.data.sensor(self.name + "_velocimeter").data
            self.sensors["pos"] = self.data.sensor(self.name + "_posimeter").data
            self.sensors["quat"] = self.data.sensor(self.name + "_orimeter").data
        for wheel in self.wheels.values():
            wheel.ctrl = self.data.actuator(wheel.name+"_actr").ctrl
            wheel.actr_force = self.data.actuator(wheel.name+"_actr").force
            if wheel.ctrl_steer is not None and wheel.actr_force_steer is not None:
                wheel.ctrl_steer = self.data.actuator(wheel.name + "_actr_steer").ctrl
                wheel.actr_force_steer = self.data.actuator(wheel.name + "_actr_steer").force