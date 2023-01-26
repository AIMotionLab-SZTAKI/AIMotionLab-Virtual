import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.active_simulation import ActiveSimulator
import os

class Wheel:

    def __init__(self, model, data, name_in_xml):

        self.name_in_xml = name_in_xml
        self.model = model
        self.data = data

        self.joint = self.data.joint(self.name_in_xml)

        try:
            self.actr = self.data.actuator(self.name_in_xml + "_actr")
            self.ctrl = self.actr.ctrl
        except:
            print("no actuator for this wheel")

class FrontWheel(Wheel):

    def __init__(self, model, data, name_in_xml):
        super().__init__(model, data, name_in_xml)
        
        self.actr_steer = self.data.actuator(self.name_in_xml + "_actr_steer")
        self.ctrl_steer = self.actr_steer.ctrl



class Car:

    def __init__(self, model, data, name_in_xml):

        self.name_in_xml = name_in_xml
        self.model = model
        self.data = data

        self.joint = self.data.joint(self.name_in_xml)
        self.qpos = self.joint.qpos

        self.wheelfl = FrontWheel(model, data, name_in_xml + "_wheelfl")
        self.wheelfr = FrontWheel(model, data, name_in_xml + "_wheelfr")
        self.wheelrl = Wheel(model, data, name_in_xml + "_wheelrl")
        self.wheelrr = Wheel(model, data, name_in_xml + "_wheelrr")
    
    def get_qpos(self):
        return self.qpos



xml_path = os.path.join("..", "xml_models")
simulator = ActiveSimulator(os.path.join(xml_path, "test_friction.xml"), None, 0.01, 0.02, False)

simulator.cam.azimuth = 90
simulator.camOnBoard.elevation = 10

car = Car(simulator.model, simulator.data, "car0")

simulator.drones += [car]

#wheel.joint.qvel[0] = 1

#print(wheelfl.actr)
car.wheelrl.ctrl[0] = 15
car.wheelrr.ctrl[0] = 15
car.wheelfl.ctrl_steer[0] = .5
car.wheelfr.ctrl_steer[0] = .5

i = 0
j = 0
sign = 1
while not simulator.glfw_window_should_close():

    simulator.update(i)

    if j > 500 and j < 600:
        car.wheelfl.ctrl_steer[0] -= .01 * sign
        car.wheelfr.ctrl_steer[0] -= .01 * sign

        if j == 599:
            j = 0
            sign *= -1

    #wheelfl.joint.qvel[0] = 1
    #wheelfr.joint.qvel[0] = 1

    j += 1
    i += 1

simulator.close()