import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.active_simulation import ActiveSimulator
from classes.moving_object import MovingObject
import os
from util import mujoco_helper
from classes.car import Car, CarMocap

xml_path = os.path.join("..", "xml_models")
simulator = ActiveSimulator(os.path.join(xml_path, "test_friction.xml"), None, 0.01, 0.02, False)
car = Car(simulator.model, simulator.data, "car0")

def up_press():
    car.up_pressed = True
def up_release():
    car.up_pressed = False
def down_press():
    car.down_pressed = True
def down_release():
    car.down_pressed = False
def left_press():
    car.left_pressed = True
def left_release():
    car.left_pressed = False
def right_press():
    car.right_pressed = True
def right_release():
    car.right_pressed = False

simulator.set_key_up_callback(up_press)
simulator.set_key_up_release_callback(up_release)
simulator.set_key_down_callback(down_press)
simulator.set_key_down_release_callback(down_release)
simulator.set_key_left_callback(left_press)
simulator.set_key_left_release_callback(left_release)
simulator.set_key_right_callback(right_press)
simulator.set_key_right_release_callback(right_release)


simulator.cam.azimuth = 90
simulator.onBoard_elev_offset = 20


simulator.drones += [car]
simulator.cars += [car]

#wheel.joint.qvel[0] = 1

#print(wheelfl.actr)
#car.wheelrl.ctrl[0] = 15
#car.wheelrr.ctrl[0] = 15
#car.wheelfl.ctrl[0] = 15
#car.wheelfr.ctrl[0] = 15
#car.wheelfl.ctrl_steer[0] = .5
#car.wheelfr.ctrl_steer[0] = .5

i = 0
sign = 1
while not simulator.glfw_window_should_close():

    simulator.update(i)

    i += 1

simulator.close()