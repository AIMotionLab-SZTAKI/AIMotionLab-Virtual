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
from classes.drone import Drone, DroneMocap
from util.xml_generator import SceneXmlGenerator
import matplotlib.pyplot as plt
import time

RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"
# init simulator

xml_path = os.path.join("..", "xml_models")
xml_base_filename = "scene.xml"
save_filename = "built_scene.xml"

scene = SceneXmlGenerator(os.path.join(xml_path, xml_base_filename))

quat = mujoco_helper.quaternion_from_euler(0, 0, math.pi)
quat = str(quat[0]) + " " + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])
scene.add_car("4 0 0.05", quat, RED_COLOR, True)
scene.save_xml(os.path.join(xml_path, save_filename))

virt_parsers = [Drone.parse, Car.parse]
mocap_parsers = [DroneMocap.parse, CarMocap.parse]

simulator = ActiveSimulator(os.path.join(xml_path, save_filename), None, 0.01, 0.02, virt_parsers, mocap_parsers, False)

#simulator.cam.elevation = -90
#simulator.cam.distance = 4

car = None

#car = Car(simulator.model, simulator.data, "virtfleet1tenth_0")
for i in range(len(simulator.all_virt_vehicles)):
    if isinstance(simulator.all_virt_vehicles[i], Car):
        car = simulator.all_virt_vehicles[i]
        break

if car is None:
    print("no car found")
    exit()

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

simulator.change_cam()

#car.up_pressed = True
#car.left_pressed = True



d = 0.04
sample_t = 1.0 / 40.0

for j in range(6):
    i = 0

    pos_arr = []
    vel_arr = []

    car.qpos[0] = 4
    
    d += 0.01
    car.d = d
    prev_sample_time = -1

    simulator.start_time = time.time()
    while not simulator.glfw_window_should_close():

        t = time.time() - simulator.start_time
        simulator.update(i)


        if t >= 6:
            car.d = 0.0
        
        if t >= 8:
            break

        time_since_prev_sample = t - prev_sample_time
        
        if time_since_prev_sample >= sample_t:

            #print(car.sensor_velocimeter)
            vel_arr += [(t, car.sensor_velocimeter[0])]
            prev_sample_time = t

        i += 1


    real_data = np.loadtxt("../velocities.csv", delimiter=',', dtype=float)

    vel_arr = np.array(vel_arr)
    #print(vel_arr)
    plt.subplot(2, 3, j + 1)
    plt.plot(vel_arr[:, 0], vel_arr[:, 1])
    plt.plot(real_data[:, 0], real_data[:, j + 1])
    plt.title("d = {:.2f}".format(d))
    plt.xlabel("time (s)")
    plt.ylabel("longitudinal vel (m/s)")

simulator.close()
plt.show()