import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.active_simulation import ActiveSimulator
from classes.moving_object import MovingObject
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

class CarMocap():

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        
        self.data = data
        self.mocapid = mocapid
        self.name_in_motive = name_in_motive
        
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

#xml_path = os.path.join("..", "xml_models")
#simulator = ActiveSimulator(os.path.join(xml_path, "test_friction.xml"), None, 0.01, 0.02, False)
#car = Car(simulator.model, simulator.data, "car0")
#
#def up_press():
#    car.up_pressed = True
#def up_release():
#    car.up_pressed = False
#def down_press():
#    car.down_pressed = True
#def down_release():
#    car.down_pressed = False
#def left_press():
#    car.left_pressed = True
#def left_release():
#    car.left_pressed = False
#def right_press():
#    car.right_pressed = True
#def right_release():
#    car.right_pressed = False
#
#simulator.set_key_up_callback(up_press)
#simulator.set_key_up_release_callback(up_release)
#simulator.set_key_down_callback(down_press)
#simulator.set_key_down_release_callback(down_release)
#simulator.set_key_left_callback(left_press)
#simulator.set_key_left_release_callback(left_release)
#simulator.set_key_right_callback(right_press)
#simulator.set_key_right_release_callback(right_release)
#
#
#simulator.cam.azimuth = 90
#simulator.onBoard_elev_offset = 20
#
#
#simulator.drones += [car]
#simulator.cars += [car]
#
##wheel.joint.qvel[0] = 1
#
##print(wheelfl.actr)
##car.wheelrl.ctrl[0] = 15
##car.wheelrr.ctrl[0] = 15
##car.wheelfl.ctrl[0] = 15
##car.wheelfr.ctrl[0] = 15
##car.wheelfl.ctrl_steer[0] = .5
##car.wheelfr.ctrl_steer[0] = .5
#
#i = 0
#sign = 1
#while not simulator.glfw_window_should_close():
#
#    simulator.update(i)
#
#    i += 1
#
#simulator.close()