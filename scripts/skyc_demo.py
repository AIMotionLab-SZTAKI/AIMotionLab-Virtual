"""
This script reads 3 crazyflie trajectories from an unzipped skyc file
using the RemoteDroneTrajectory class and creates a simulation
that goes through the them with 3 crazyflies.
"""

from aiml_virtual.trajectory import RemoteDroneTrajectory
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.controller import GeomControl

import matplotlib.pyplot as plt

import os
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))


demo_path0 = os.path.join(abs_path, "..", "Demo", "drones", "drone_0")
traj0 = RemoteDroneTrajectory(directory=demo_path0)
demo_path1 = os.path.join(abs_path, "..", "Demo", "drones", "drone_1")
traj1 = RemoteDroneTrajectory(directory=demo_path1)
demo_path2 = os.path.join(abs_path, "..", "Demo", "drones", "drone_2")
traj2 = RemoteDroneTrajectory(directory=demo_path2)

# get initial positions from trajectories
pos, vel = traj0.evaluate_trajectory(0)
drone0_init_pos = np.array(pos)
pos, vel = traj1.evaluate_trajectory(0)
drone1_init_pos = np.array(pos)
pos, vel = traj2.evaluate_trajectory(0)
drone2_init_pos = np.array(pos)

#n = 50000
#ts = np.linspace(1, 50, n)
#vels = np.zeros((n, 3))
#
#i = 0
#for t in ts:
#    pos, vel = traj2.evaluate_trajectory(t)
#    vels[i] = vel
#    i += 1
#
#plt.plot(ts, vels[:, 0], label="Vx")
#plt.plot(ts, vels[:, 1], label="Vy")
#plt.plot(ts, vels[:, 2], label="Vz")
#plt.legend()
#plt.show()

xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base.xml"
save_filename = "built_scene.xml"

scene = SceneXmlGenerator(xml_base_file_name)

drone0_name = scene.add_drone(np.array2string(drone0_init_pos)[1:-1], "1 0 0 0", "0.9 0.1 0.1 1.0", DRONE_TYPES.CRAZYFLIE)
drone1_name = scene.add_drone(np.array2string(drone1_init_pos)[1:-1], "1 0 0 0", "0.1 0.1 0.9 1.0", DRONE_TYPES.CRAZYFLIE)
drone2_name = scene.add_drone(np.array2string(drone2_init_pos)[1:-1], "1 0 0 0", "0.1 0.9 0.1 1.0", DRONE_TYPES.CRAZYFLIE)

scene.save_xml(os.path.join(xml_path, save_filename))


xml_filename = os.path.join(xml_path, save_filename)
simulator = ActiveSimulator(xml_filename, None, 0.01, 0.02, [parseMovingObjects], None)

drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
drone1 = simulator.get_MovingObject_by_name_in_xml(drone1_name)
drone2 = simulator.get_MovingObject_by_name_in_xml(drone2_name)

controller0 = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)
controller1 = GeomControl(drone1.mass, drone1.inertia, simulator.gravity)
controller2 = GeomControl(drone2.mass, drone2.inertia, simulator.gravity)

drone0.set_controllers([controller0])
drone0.set_trajectory(traj0)

drone1.set_controllers([controller1])
drone1.set_trajectory(traj1)

drone2.set_controllers([controller2])
drone2.set_trajectory(traj2)

list_ctrl = []
drone_to_sample = drone0

while not simulator.glfw_window_should_close():

    simulator.update()

    list_ctrl += [np.array((drone_to_sample.ctrl0[0], drone_to_sample.ctrl1[0], drone_to_sample.ctrl2[0], drone_to_sample.ctrl3[0]))]

    #if simulator.i > 1500:
    #    break

simulator.close()


#list_ctrl = np.array(list_ctrl)
#
#plt.plot(list_ctrl[:, 0])
#plt.plot(list_ctrl[:, 1])
#plt.plot(list_ctrl[:, 2])
#plt.plot(list_ctrl[:, 3])
#
#plt.show()