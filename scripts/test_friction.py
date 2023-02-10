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
from util.xml_generator import SceneXmlGenerator
import matplotlib.pyplot as plt

RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"
# init simulator

xml_path = os.path.join("..", "xml_models")
xml_base_filename = "scene.xml"
save_filename = "built_scene.xml"

scene = SceneXmlGenerator(os.path.join(xml_path, xml_base_filename))
scene.add_car("0 0 0", "1 0 0 0", RED_COLOR, True)
scene.save_xml(os.path.join(xml_path, save_filename))

simulator = ActiveSimulator(os.path.join(xml_path, save_filename), None, 0.01, 0.02, False)

simulator.cam.elevation = -90
simulator.cam.distance = 4

#car = Car(simulator.model, simulator.data, "virtfleet1tenth_0")
car = simulator.virtcars[0]

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

#car.up_pressed = True
#car.left_pressed = True

i = 0
sign = 1

pos_arr = []

while not simulator.glfw_window_should_close():

    simulator.update(i)

    if i > 100:
        if (i % 10) == 0:
            pos_arr += [np.copy(car.get_qpos()[:2])]

    i += 1


simulator.close()

pos_arr = np.array(pos_arr)
print(pos_arr)
plt.plot(pos_arr[:,0], pos_arr[:,1])
plt.show()