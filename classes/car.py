import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.moving_object import MovingObject, MovingMocapObject
import os
from util import mujoco_helper


class Wheel:

    def __init__(self, data, name_in_xml):

        self.name_in_xml = name_in_xml
        self.data = data

        self.joint = self.data.joint(self.name_in_xml)

        try:
            self.actr = self.data.actuator(self.name_in_xml + "_actr")
            self.ctrl = self.actr.ctrl
        except:
            print("no actuator for this wheel")

class FrontWheel(Wheel):

    def __init__(self, data, name_in_xml):
        super().__init__(data, name_in_xml)
        
        self.actr_steer = self.data.actuator(self.name_in_xml + "_actr_steer")
        self.ctrl_steer = self.actr_steer.ctrl
        self.joint_steer = self.data.joint(self.name_in_xml + "_steer")

class Car(MovingObject):

    def __init__(self, data, name_in_xml):

        super().__init__()

        self.WB = .32226
        self.TW = .20032

        self.name_in_xml = name_in_xml
        self.data = data

        self.joint = self.data.joint(self.name_in_xml)
        self.qpos = self.joint.qpos

        self.wheelfl = FrontWheel(data, name_in_xml + "_wheelfl")
        self.wheelfr = FrontWheel(data, name_in_xml + "_wheelfr")
        self.wheelrl = Wheel(data, name_in_xml + "_wheelrl")
        self.wheelrr = Wheel(data, name_in_xml + "_wheelrr")

        self.j = 0
        self.sign = 1

        self.up_pressed = False
        self.down_pressed = False
        self.left_pressed = False
        self.right_pressed = False

        self.cacc = .005
        self.max_vel = 0.1

        self.qvel = self.joint.qvel

        self.sensor_data = self.data.sensor(self.name_in_xml + "_gyro").data

        self.steer_angle = 0
        self.max_steer = 0.4
    
    def get_qpos(self):
        return self.qpos
    
    def set_controller(self, controller):
        self.controller = controller

    def update(self, i):

        self.control_by_keyboard()
        #print(self.qpos)
    
    def calc_frontwheel_angles(self, delta_in):
        
        num = self.WB * math.tan(delta_in)

        delta_left = math.atan(num / (self.WB + (0.5 * self.TW * math.tan(delta_in))))

        delta_right = math.atan(num / (self.WB - (0.5 * self.TW * math.tan(delta_in))))

        return delta_left, delta_right
        #return delta_in, delta_in
    
    def print_info(self):
        print("Virtual")
        print("name in xml:      " + self.name_in_xml)

    def control_by_keyboard(self):
        if self.up_pressed:
            if self.wheelrl.ctrl[0] < self.max_vel:
                self.wheelrl.ctrl[0] += self.cacc
                self.wheelrr.ctrl[0] += self.cacc
                self.wheelfl.ctrl[0] += self.cacc
                self.wheelfr.ctrl[0] += self.cacc

        else:
            if self.wheelrl.ctrl[0] > 0:
                if self.wheelrl.ctrl[0] < self.cacc:
                    self.wheelrl.ctrl[0] = 0
                    self.wheelrr.ctrl[0] = 0
                    self.wheelfl.ctrl[0] = 0
                    self.wheelfr.ctrl[0] = 0
                self.wheelrl.ctrl[0] -= self.cacc
                self.wheelrr.ctrl[0] -= self.cacc
                self.wheelfl.ctrl[0] -= self.cacc
                self.wheelfr.ctrl[0] -= self.cacc

                #print(self.wheelrl.ctrl)

        if self.down_pressed:
            if self.wheelrl.ctrl[0] > -self.max_vel:
                self.wheelrl.ctrl[0] -= self.cacc
                self.wheelrr.ctrl[0] -= self.cacc
                self.wheelfl.ctrl[0] -= self.cacc
                self.wheelfr.ctrl[0] -= self.cacc

        else:
            if self.wheelrl.ctrl[0] < 0:
                if self.wheelrl.ctrl[0] > -self.cacc:
                    self.wheelrl.ctrl[0] = 0
                    self.wheelrr.ctrl[0] = 0
                    self.wheelfl.ctrl[0] = 0
                    self.wheelfr.ctrl[0] = 0
                self.wheelrl.ctrl[0] += self.cacc
                self.wheelrr.ctrl[0] += self.cacc
                self.wheelfl.ctrl[0] += self.cacc
                self.wheelfr.ctrl[0] += self.cacc

        if self.right_pressed:
            #self.wheelfl.ctrl_steer[0] = -0.5
            #self.wheelfr.ctrl_steer[0] = -0.5
            if self.steer_angle > -self.max_steer:
                self.steer_angle -= 0.01
                delta_left, delta_right = self.calc_frontwheel_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right
        
        else:
            if self.steer_angle < 0:
                self.steer_angle += 0.01
                delta_left, delta_right = self.calc_frontwheel_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right

        if self.left_pressed:
            #self.wheelfl.ctrl_steer[0] = 0.5
            #self.wheelfr.ctrl_steer[0] = 0.5
            if self.steer_angle < self.max_steer:
                self.steer_angle += 0.01
                delta_left, delta_right = self.calc_frontwheel_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right
        
        else:
        #    self.wheelfl.ctrl_steer[0] = 0
        #    self.wheelfr.ctrl_steer[0] = 0
            if self.steer_angle > 0:
                self.steer_angle -= 0.01
                delta_left, delta_right = self.calc_frontwheel_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right
        
        #if (not self.left_pressed) and (not self.right_pressed):
            #self.wheelfl.ctrl_steer[0] = 0
            #self.wheelfr.ctrl_steer[0] = 0
            #pass
            #print("no steer")
            #self.wheelfl.ctrl_steer[0] = 0
            #self.wheelfr.ctrl_steer[0] = 0

        
    @staticmethod
    def parse(data, model):
        cars = []
        ivc = 0

        joint_names = mujoco_helper.get_joint_name_list(model)

        for name in joint_names:

            name_cut = name[: len(name) - 2]

            if name.startswith("virtfleet1tenth") and not name.endswith("steer") and not name_cut.endswith("wheel"):

                car = Car(data, name)

                cars += [car]
                ivc += 1


        return cars


class CarMocap(MovingMocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(name_in_xml, name_in_motive)
        
        self.data = data
        self.mocapid = mocapid
    
    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])
    
    def get_name_in_xml(self):
        return self.name_in_xml
        
    def update(self, pos, quat):
        
        #euler = mujoco_helper.euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])

        #euler[0] = 0
        #euler[1] = 0

        #quat = mujoco_helper.quaternion_from_euler(euler[0], euler[1], euler[2])

        pos = np.copy(pos)
        pos[2] -= 0.12

        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat

    @staticmethod
    def parse(data, model):
        realcars = []
        irc = 1

        body_names = mujoco_helper.get_body_name_list(model)

        for name in body_names:
            name_cut = name[:len(name) - 2]
            if name.startswith("realfleet1tenth") and not name_cut.endswith("wheel"):
                
                mocapid = model.body(name).mocapid[0]
                c = CarMocap(model, data, mocapid, name, "RC_car_0" + str(irc))
                
                realcars += [c]
                irc += 1
        
        return realcars
