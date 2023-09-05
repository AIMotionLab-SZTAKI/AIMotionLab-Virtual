import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.moving_object import MovingObject, MocapObject
import os
from util import mujoco_helper

class F1T_PROP(Enum):
    WHEEL_RADIUS = ".052388"
    WHEEL_WIDTH = ".022225"
    WHEEL_SIZE = WHEEL_RADIUS + " " + WHEEL_WIDTH

class Wheel:

    def __init__(self, model, data, name_in_xml):

        self.name_in_xml = name_in_xml
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
        self.joint_steer = self.data.joint(self.name_in_xml + "_steer")

class Car(MovingObject):

    def __init__(self, model, data, name_in_xml):

        super().__init__(model, name_in_xml)

        self.data = data

        self.joint = self.data.joint(self.name_in_xml)
        self.qpos = self.joint.qpos

        self.wheelfl = FrontWheel(model, data, name_in_xml + "_wheelfl")
        self.wheelfr = FrontWheel(model, data, name_in_xml + "_wheelfr")
        self.wheelrl = Wheel(model, data, name_in_xml + "_wheelrl")
        self.wheelrr = Wheel(model, data, name_in_xml + "_wheelrr")

        self.up_pressed = False
        self.down_pressed = False
        self.left_pressed = False
        self.right_pressed = False

        self.cacc = .005
        self.max_vel = 0.1

        self.qvel = self.joint.qvel

        self.sensor_gyro = self.data.sensor(self.name_in_xml + "_gyro").data
        self.sensor_velocimeter = self.data.sensor(self.name_in_xml + "_velocimeter").data
        self.sensor_posimeter = self.data.sensor(self.name_in_xml + "_posimeter").data
        self.sensor_orimeter = self.data.sensor(self.name_in_xml + "_orimeter").data

        roll, pitch, yaw = mujoco_helper.euler_from_quaternion(*self.sensor_orimeter)

        #self.mass = model.body(self.name_in_xml).mass
        self.state = {
            "pos_x" : self.sensor_posimeter[0],
            "pos_y" : self.sensor_posimeter[1],
            "head_angle" : yaw,
            "long_vel" : self.sensor_velocimeter[0],
            "lat_vel" : self.sensor_velocimeter[1],
            "yaw_rate" : self.sensor_gyro[2]
        }

        self.steer_angle = 0
        self.max_steer = 0.5
    
    def set_ackermann_parameters(self, WB, TW):
        self.WB = WB
        self.TW = TW
    
    def get_qpos(self):
        return self.qpos
    

    def update(self, i, control_step):
        
        # implement this in subclass
        return
    
    def set_ctrl(self, ctrl):
        self.d = ctrl[0]
        self.delta = ctrl[1]

        self.set_torque(self.calc_torque())

        self.set_steer_angle()



    def get_state(self):
        # x, y
        # heading angle phi (with respect to x axis)
        # longitudinal velocity
        # lateral velocity
        # yaw rate
        
        roll, pitch, yaw = mujoco_helper.euler_from_quaternion(*self.sensor_orimeter)

        #self.mass = model.body(self.name_in_xml).mass
        self.state["pos_x"] = self.sensor_posimeter[0]
        self.state["pos_y"] = self.sensor_posimeter[1]
        self.state["head_angle"] = yaw
        self.state["long_vel"] = self.sensor_velocimeter[0]
        self.state["lat_vel"] = self.sensor_velocimeter[1]
        self.state["yaw_rate"] = self.sensor_gyro[2]
        return self.state
    


    def calc_torque(self):
        
        d = self.d

        v = self.sensor_velocimeter
        #self.v_long = math.sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]))

        self.v_long = v[0]

        return (self.C_m1 * d) - (self.C_m2 * self.v_long) - (self.C_m3 * np.sign(self.v_long))
        #self.torque = self.clamp(self.torque, -.2, .2)

        #self.torque = self.d

    
    def set_torque(self, torque):
        self.wheelrl.ctrl[0] = torque
        self.wheelrr.ctrl[0] = torque
        self.wheelfl.ctrl[0] = torque
        self.wheelfr.ctrl[0] = torque

    
    def calc_ackerman_angles(self, delta_in):
        
        num = self.WB * math.tan(delta_in)

        delta_left = math.atan(num / (self.WB + (0.5 * self.TW * math.tan(delta_in))))

        delta_right = math.atan(num / (self.WB - (0.5 * self.TW * math.tan(delta_in))))

        return delta_left, delta_right
        #return delta_in, delta_in
    
    def set_steer_angle(self):
        
        delta_left, delta_right = self.calc_ackerman_angles(self.delta)
        self.wheelfl.ctrl_steer[0] = delta_left
        self.wheelfr.ctrl_steer[0] = delta_right


    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

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
                delta_left, delta_right = self.calc_ackerman_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right
        
        else:
            if self.steer_angle < 0:
                self.steer_angle += 0.01
                delta_left, delta_right = self.calc_ackerman_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right

        if self.left_pressed:
            #self.wheelfl.ctrl_steer[0] = 0.5
            #self.wheelfr.ctrl_steer[0] = 0.5
            if self.steer_angle < self.max_steer:
                self.steer_angle += 0.01
                delta_left, delta_right = self.calc_ackerman_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right
        
        else:
        #    self.wheelfl.ctrl_steer[0] = 0
        #    self.wheelfr.ctrl_steer[0] = 0
            if self.steer_angle > 0:
                self.steer_angle -= 0.01
                delta_left, delta_right = self.calc_ackerman_angles(self.steer_angle)
                self.wheelfl.ctrl_steer[0] = delta_left
                self.wheelfr.ctrl_steer[0] = delta_right
        
        #if (not self.left_pressed) and (not self.right_pressed):
            #self.wheelfl.ctrl_steer[0] = 0
            #self.wheelfr.ctrl_steer[0] = 0
            #pass
            #print("no steer")
            #self.wheelfl.ctrl_steer[0] = 0
            #self.wheelfr.ctrl_steer[0] = 0

class Fleet1Tenth(Car):

    def __init__(self, model, data, name_in_xml):
        super().__init__(model, data, name_in_xml)

        self.set_ackermann_parameters(.32226, .20032)

        self.torque = 0.0
        self.d = 0.0
        self.v_long = 0.0

        self.C_m1 = 65
        self.C_m2 = 3.3
        self.C_m3 = 1.05
    

    def update(self, i, control_step):
        
        #self.calc_torque()

        #self.set_torque()

        if self.trajectory is not None:
            state = self.get_state()

            setpoint = self.trajectory.evaluate(state, i, self.data.time, control_step)

            self.update_controller_type(state, setpoint, self.data.time, i)
        
            if self.controller is not None:
                ctrl = self.controller.compute_control(state, setpoint, self.data.time)
            
            if ctrl is not None:
                self.set_ctrl(ctrl)
    
    


class CarMocap(MocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(model, data, mocapid, name_in_xml, name_in_motive)
        
    
    def get_name_in_xml(self):
        return self.name_in_xml
        
    def update(self, pos, quat):
        
        euler = mujoco_helper.euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])

        euler[0] = 0
        euler[1] = 0

        quat = mujoco_helper.quaternion_from_euler(euler[0], euler[1], euler[2])

        pos = np.copy(pos)
        pos[2] = 0.052388

        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat
