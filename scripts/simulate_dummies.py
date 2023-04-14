from lib2to3.pgen2 import driver
import os

from classes.active_simulation import ActiveSimulator

from util import xml_generator

from classes.drone import Drone, DroneMocap, HookMocap
from classes.car import Car, CarMocap
from classes.payload import PayloadMocap, PAYLOAD_TYPES, Payload

from classes.controller_base import DummyDroneController, DummyCarController
from classes.trajectory_base import DummyDroneTrajectory, DummyCarTrajectory
from util import mujoco_helper

import numpy as np
import matplotlib.pyplot as plt
import math


RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"
BLACK_COLOR = ".1 .1 .1 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene.xml"
save_filename = "built_scene.xml"

# create xml with a drone and a car
scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)
drone0_name = scene.add_drone("-5 0 1", "1 0 0 0", RED_COLOR, True, "bumblebee", True, 1)
dronemocap0_name = scene.add_drone("1 1 1", "1 0 0 0", BLUE_COLOR, False, "bumblebee", True, 1)
car0_name = scene.add_car("-5 1 0.6", "1 0 0 0", RED_COLOR, True)
mocap_load0_name = scene.add_load("-1 0 0", ".1 .1 .1", None, "1 0 0 0", BLACK_COLOR, PAYLOAD_TYPES.Teardrop.value, True)
load0_name = scene.add_load("1 0 15", ".1 .1 .5", ".1", "1 1 0 1", BLACK_COLOR, PAYLOAD_TYPES.Box.value)

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers
virt_parsers = [Drone.parse, Car.parse, Payload.parse]
mocap_parsers = [DroneMocap.parse, CarMocap.parse, PayloadMocap.parse, HookMocap.parse]


control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

# initializing simulator
simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False)

# grabbing the drone and the car
drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
car0 = simulator.get_MovingObject_by_name_in_xml(car0_name)
mocap_load0 = simulator.get_MovingMocapObject_by_name_in_xml(mocap_load0_name)
load0 = simulator.get_MovingObject_by_name_in_xml(load0_name)

# creating trajectory and controller for drone0
drone0_trajectory = DummyDroneTrajectory()
drone0_controller0 = DummyDroneController(drone0.mass, drone0.inertia, simulator.gravity)
drone0_controller1 = DummyDroneController(drone0.mass, drone0.inertia, simulator.gravity)

drone0_controllers = [drone0_controller0]

# creating trajectory and controller for car0
car0_trajectory = DummyCarTrajectory()
car0_controller0 = DummyCarController(car0.mass, car0.inertia, simulator.gravity)
car0_controller1 = DummyCarController(car0.mass, car0.inertia, simulator.gravity)

car0_controllers = [car0_controller0]

# setting update_controller_type method, trajectory and controller for drone0
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)

# setting update_controller_type method, trajectory and controller for car0
car0.set_trajectory(car0_trajectory)
car0.set_controllers(car0_controllers)


# start simulation
i = 0
sensor_data = []
q_data = []
# print(drone0.qacc)
#drone0.qvel[1] = 0.0
#drone0.hook_qvel_y[0] = 1
#drone0.hook_qvel_x[0] = -0.1
drone0.qfrc_applied[5] += .01
while not simulator.glfw_window_should_close():
    simulator.update(i)
    if i % 10 == 0:
        #print(drone0.qfrc_applied)
        print(load0.sensor_posimeter)

    i += 1

simulator.close()

#plt.plot(sensor_data)
#plt.plot(q_data, '--')
#plt.legend(["sensor_joint_ang0", "sensor_joint_ang1", "q_joint_ang0", "q_joint_ang1"])
#plt.show()