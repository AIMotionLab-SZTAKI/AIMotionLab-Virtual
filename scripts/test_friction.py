import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.active_simulation import ActiveSimulator
from classes.object_parser import parseMovingObjects, parseMocapObjects
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

#for i in range(100):
#    t1 = time.time()
#    time.sleep(0.01)
#    t2 = time.time()
#
#    print(t2 - t1)

abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene_base.xml"
save_filename = "built_scene.xml"

scene = SceneXmlGenerator(xmlBaseFileName)
quat = mujoco_helper.quaternion_from_euler(0, 0, math.pi)
quat = str(quat[0]) + " " + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])
car_name = scene.add_car("4 0 0.05", quat, RED_COLOR, True)
scene.save_xml(os.path.join(xml_path, save_filename))

virt_parsers = [parseMovingObjects]
mocap_parsers = [parseMocapObjects]

simulator = ActiveSimulator(os.path.join(xml_path, save_filename), None, 0.005, 0.02, virt_parsers, mocap_parsers, False)


#simulator.cam.elevation = -90
#simulator.cam.distance = 4

car = simulator.get_MovingObject_by_name_in_xml(car_name)


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

#simulator.change_cam()

#car.up_pressed = True
#car.left_pressed = True

SAMPLE_T = 1.0 / 20.0

def simulate_with_graphix(vel_arr, pos_arr):
    #simulator.start_time = time.time()
    prev_sample_time = -1

    while not simulator.glfw_window_should_close():

        #print(simulator.start_time)

        simulator.update()
        
        #t = time.time() - simulator.start_time
        t = simulator.i * simulator.control_step

        #print(t)

        if t >= 6:
            car.d = 0.0
        
        if t >= 8:
            break

        time_since_prev_sample = t - prev_sample_time


        #print(time_since_prev_sample)
        
        if time_since_prev_sample >= SAMPLE_T:

            #print(car.sensor_velocimeter)
            vel_arr += [(t, car.sensor_velocimeter[0])]
            prev_sample_time = t


def simulate_without_graphix(vel_arr, pos_arr):
    #simulator.start_time = time.time()
    prev_sample_time = -1

    while not simulator.glfw_window_should_close():

        t = simulator.i * simulator.sim_step
        simulator.update_()


        if t >= 6:
            car.d = 0.0
        
        if t >= 8:
            break

        time_since_prev_sample = t - prev_sample_time
        
        if time_since_prev_sample >= (SAMPLE_T - 0.0001):
            #print(time_since_prev_sample)

            #print(car.sensor_velocimeter)
            vel_arr += [(t, car.sensor_velocimeter[0])]
            prev_sample_time = t




def straight_line_vel_profile():
    d = 0.04
    d_increment = 0.01
    car_init_pos_x = 4

    for j in range(6):

        pos_arr = []
        vel_arr = []

        car.qpos[0] = car_init_pos_x
        
        d += d_increment
        car.d = d


        simulate_with_graphix(vel_arr, pos_arr)
        #simulate_without_graphix(vel_arr, pos_arr)


        real_data = np.loadtxt(os.path.join(abs_path, "..", "f1tenth_measurements", "velocities.csv"), delimiter=',', dtype=float)

        vel_arr = np.array(vel_arr)
        #print(vel_arr)
        plt.subplot(2, 3, j + 1)
        plt.plot(vel_arr[:, 0], vel_arr[:, 1])
        plt.plot(real_data[:, 0], real_data[:, j + 1])
        plt.title("d = {:.2f}".format(d))
        plt.xlabel("time (s)")
        plt.ylabel("longitudinal vel (m/s)")
        plt.legend(["simulated", "real"])

    simulator.close()
    plt.show()


def simulate_circular(vel_arr, pos_arr, d, delta, with_graphics=False):
    #simulator.start_time = time.time()
    prev_sample_time = -1

    steer_angles = []

    while not simulator.glfw_window_should_close():

        if with_graphics:
            simulator.update()
            t = time.time() - simulator.start_time
            time_since_prev_sample = t - prev_sample_time
        else:
            simulator.update_()
            t = simulator.i * simulator.sim_step
            time_since_prev_sample = t - prev_sample_time


        if t > 0.2:
            car.set_steer_angle(delta)
            car.d = d
        
        #if t > 3:

        if t >= 8.875:
            break

        
        if time_since_prev_sample >= SAMPLE_T:

            #print(car.sensor_velocimeter)
            vel_arr += [(t, car.sensor_velocimeter[0])]
            pos_arr += [(t, car.sensor_posimeter[0], car.sensor_posimeter[1])]

            steer_angles += [(t, car.wheelfl.joint_steer.qpos[0], car.wheelfr.joint_steer.qpos[0])]

            prev_sample_time = t
    
    
    simulator.close()
    steer_angles = np.array(steer_angles)
    plt.plot(steer_angles[:, 0], steer_angles[:, 1])
    plt.plot(steer_angles[:, 0], steer_angles[:, 2])
    plt.show()
    

def get_filename(d, delta):
    if delta < 0:
        filename = "t_deln" + str(int(abs(delta) * 10))
    else:
        filename = "t_del" + str(int(delta * 10))
    
    d_conv = d * 1000
    if d_conv < 100:
        ending = "_d0" + str(int(d_conv)) + "_state.csv"
    
    else:
        ending = "_d" + str(int(d_conv)) + "_state.csv"
    
    return filename + ending


def circular_():
    d = 0.15
    delta = -0.5
    d_increment = 0.01
    sample_t = 1.0 / 40.0

    folder = os.path.join(abs_path, "..", "f1tenth_measurements", "rekrmozgsf1tenth")

    filename = get_filename(d, delta)

    #filename = os.path.join(folder, "t_del3_d075_state.csv")
    filename = os.path.join(folder, filename)

    real_data = np.loadtxt(filename, delimiter=",", dtype=float)


    #quat = mujoco_helper.quaternion_from_euler(0, 0, math.radians(2))
    #print(real_data[0, 3])
    quat = mujoco_helper.quaternion_from_euler(0, 0, real_data[0, 3])
    car_init_pos_x = real_data[0, 1]
    car_init_pos_y = real_data[0, 2]

    #car.d = d
    #car.set_steer_angle(delta)
    car.qpos[0] = car_init_pos_x
    car.qpos[1] = car_init_pos_y

    car.qpos[3] = quat[0]
    car.qpos[4] = quat[1]
    car.qpos[5] = quat[2]
    car.qpos[6] = quat[3]

    vel_arr = []
    pos_arr = []

    simulate_circular(vel_arr, pos_arr, d, delta, True)

    vel_arr = np.array(vel_arr)
    pos_arr = np.array(pos_arr)
    #print(vel_arr)
    #plt.subplot(2, 3, j + 1)
    #print(pos_arr)
    plt.plot(pos_arr[:, 1], pos_arr[:, 2])
    plt.plot(real_data[:, 1], real_data[:, 2])
    plt.title("d = {:.3f} ; delta = {:.2f}".format(d, delta))
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend(["simulated", "real"])

    plt.show()

def simulate_random_curve(data, vel_arr, pos_arr, with_graphics=False):
    
    #simulator.start_time = time.time()
    prev_sample_time = -1
    prev_t = 0

    d = 0
    delta = 0

    steer_angles = []

    j = 0

    delej = 0.02

    j_start = int(delej / simulator.control_step)

    while not simulator.glfw_window_should_close():


        if with_graphics:
            simulator.update()
            #t = time.time() - simulator.start_time
            t = simulator.i * simulator.control_step
            #print(t - prev_t)
            #prev_t = t
        else:
            simulator.update_()
            t = simulator.i * simulator.sim_step
        
        time_since_prev_sample = t - prev_sample_time
        #print(t)

        
        if time_since_prev_sample >= (SAMPLE_T - 0.0001):
            #print(time_since_prev_sample)

            if j >= j_start:
                d = data[0, j - j_start]
                delta = data[1, j - j_start]
            
            car.delta = delta
            car.set_steer_angle()
            car.d = d + 0.005
            j += 1
            #print(car.sensor_velocimeter)
            vel_arr += [(t, car.sensor_velocimeter[0])]
            pos_arr += [(t, car.sensor_posimeter[0], car.sensor_posimeter[1])]

            steer_angles += [(t, car.wheelfl.joint_steer.qpos[0], car.wheelfr.joint_steer.qpos[0])]

            prev_sample_time = t
            
            if j == data.shape[1]:
                break
    
    
    simulator.close()
    #steer_angles = np.array(steer_angles)
    #plt.plot(steer_angles[:, 0], steer_angles[:, 1])
    #plt.plot(steer_angles[:, 0], steer_angles[:, 2])
    #plt.show()



def random_curve():

    global SAMPLE_T
    
    vel_arr = []
    pos_arr = []

    #filename = os.path.join("..", "valid_sin1.csv")
    filename = os.path.join(abs_path, "..", "f1tenth_measurements", "motion_2023-03-03-12-22-06.csv")

    data = np.loadtxt(filename, delimiter=",", dtype=float)
    SAMPLE_T = (data[20, 0] - data[0, 0]) / 20
    print("Sampling time: " + str(SAMPLE_T))
    quat = mujoco_helper.quaternion_from_euler(0, 0, data[0, 3])
    car_init_pos_x = data[0, 1]
    car_init_pos_y = data[0, 2]

    #car.d = d
    #car.set_steer_angle(delta)
    car.qpos[0] = car_init_pos_x
    car.qpos[1] = car_init_pos_y

    car.qpos[3] = quat[0]
    car.qpos[4] = quat[1]
    car.qpos[5] = quat[2]
    car.qpos[6] = quat[3]

    control_inp = np.vstack((data[:, 8], data[:, 10]))
    print(control_inp.shape)

    simulate_random_curve(control_inp, vel_arr, pos_arr, True)
    
    vel_arr = np.array(vel_arr)
    pos_arr = np.array(pos_arr)
    #print(vel_arr)
    #plt.subplot(2, 3, j + 1)
    #print(pos_arr)
    plt.subplot(2, 2, 1)
    plt.plot(pos_arr[:, 1], pos_arr[:, 2])
    plt.plot(data[:, 1], data[:, 2])
    #plt.title("d = {:.3f} ; delta = {:.2f}".format(d, delta))
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend(["simulated", "real"])

    #plt.show()
    
    data[:, 0] = data[:, 0] - data[0, 0]

    plt.subplot(2, 2, 2)
    plt.plot(vel_arr[:, 0], vel_arr[:, 1])
    plt.plot(data[:, 0], data[:, 5])
    plt.xlabel("time")
    plt.ylabel("Longitudinal velocity")

    plt.subplot(2, 2, 3)


    plt.plot(pos_arr[:, 0], pos_arr[:, 1])
    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel("time")
    plt.ylabel("X position")

    plt.subplot(2, 2, 4)

    plt.plot(pos_arr[:, 0], pos_arr[:, 2])
    plt.plot(data[:, 0], data[:, 2])
    plt.xlabel("time")
    plt.ylabel("Y position")

    plt.show()




#straight_line_vel_profile()
#circular_()
random_curve()