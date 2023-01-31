import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.moving_object import MovingObject, MovingMocapObject
import os
from util import mujoco_helper


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

class CarMocap(MovingMocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        
        self.data = data
        self.mocapid = mocapid
        self.name_in_motive = name_in_motive
    
    def get_name_in_xml(self):
        return self.name_in_xml
        
    def update(self, pos, quat):
        
        euler = mujoco_helper.euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])

        euler[0] = 0
        euler[1] = 0

        quat = mujoco_helper.quaternion_from_euler(euler[0], euler[1], euler[2])

        pos = np.copy(pos)
        pos[2] -= 0.12

        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat

    def spin_propellers(self, control_step, speed):
        pass

    @staticmethod
    def parse_mocap_cars(data, model, body_names):
        return []

class Car(MovingObject):

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

        self.j = 0
        self.sign = 1

        self.up_pressed = False
        self.down_pressed = False
        self.left_pressed = False
        self.right_pressed = False

        self.cacc = 1
    
    def get_qpos(self):
        return self.qpos

    def update(self, i):
        #if self.j > 500 and self.j < 600:
        #    self.wheelfl.ctrl_steer[0] -= .01 * self.sign
        #    self.wheelfr.ctrl_steer[0] -= .01 * self.sign
#
        #    if self.j == 599:
        #        self.j = 0
        #        self.sign *= -1
#
#
        #self.j += 1
        self.control_by_keyboard()
    

    def control_by_keyboard(self):
        if self.up_pressed:
            if self.wheelrl.ctrl[0] < 100:
                self.wheelrl.ctrl[0] += self.cacc
                self.wheelrr.ctrl[0] += self.cacc
                self.wheelfl.ctrl[0] += self.cacc
                self.wheelfr.ctrl[0] += self.cacc

        else:
            if self.wheelrl.ctrl[0] > 0:
                self.wheelrl.ctrl[0] -= self.cacc
                self.wheelrr.ctrl[0] -= self.cacc
                self.wheelfl.ctrl[0] -= self.cacc
                self.wheelfr.ctrl[0] -= self.cacc

        if self.down_pressed:
            if self.wheelrl.ctrl[0] > -20:
                self.wheelrl.ctrl[0] -= self.cacc
                self.wheelrr.ctrl[0] -= self.cacc
                self.wheelfl.ctrl[0] -= self.cacc
                self.wheelfr.ctrl[0] -= self.cacc

        else:
            if self.wheelrl.ctrl[0] < 0:
                self.wheelrl.ctrl[0] += self.cacc
                self.wheelrr.ctrl[0] += self.cacc
                self.wheelfl.ctrl[0] += self.cacc
                self.wheelfr.ctrl[0] += self.cacc

        if self.right_pressed:
            if self.wheelfl.ctrl_steer > -0.5:
                self.wheelfl.ctrl_steer -= 0.01
                self.wheelfr.ctrl_steer -= 0.01
        
        else:
            if self.wheelfl.ctrl_steer < 0:
                self.wheelfl.ctrl_steer += 0.01
                self.wheelfr.ctrl_steer += 0.01

        if self.left_pressed:
            if self.wheelfl.ctrl_steer < 0.5:
                self.wheelfl.ctrl_steer += 0.01
                self.wheelfr.ctrl_steer += 0.01
        
        else:
            if self.wheelfl.ctrl_steer > 0:
                self.wheelfl.ctrl_steer -= 0.01
                self.wheelfr.ctrl_steer -= 0.01
        
    @staticmethod
    def parse_cars(data, joint_names):
        return []