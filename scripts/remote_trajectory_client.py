from aiml_virtual.trajectory import TrajectoryDistributor, RemoteDroneTrajectory
from aiml_virtual.controller import GeomControl
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object import parseMovingObjects
import os
import numpy as np
import time

RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"
GREEN = "0.2 0.85 0.2 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene_base.xml"
save_filename = "built_scene.xml"


scene = SceneXmlGenerator(xmlBaseFileName)

drone0_initpos = np.array((1.25, 0.0, 0.0))
drone1_initpos = np.array((1.25, -0.5, 0.0))
drone2_initpos = np.array((0.0, 1.0, 0.0))
drone3_initpos = np.array((1.0, 0.0, 0.0))
drone4_initpos = np.array((1.0, 1.0, 0.0))

drone0_name = scene.add_drone(np.array2string(drone0_initpos)[1:-1], "1 0 0 0", BLUE, DRONE_TYPES.CRAZYFLIE)
drone1_name = scene.add_drone(np.array2string(drone1_initpos)[1:-1], "1 0 0 0", GREEN, DRONE_TYPES.CRAZYFLIE)
drone2_name = scene.add_drone(np.array2string(drone2_initpos)[1:-1], "1 0 0 0", RED, DRONE_TYPES.CRAZYFLIE)
drone3_name = scene.add_drone(np.array2string(drone3_initpos)[1:-1], "1 0 0 0", BLUE, DRONE_TYPES.CRAZYFLIE)
drone4_name = scene.add_drone(np.array2string(drone4_initpos)[1:-1], "1 0 0 0", BLUE, DRONE_TYPES.CRAZYFLIE)
#drone2_name = scene.add_drone("1 0 1", "1 0 0 0", BLUE, DRONE_TYPES.CRAZYFLIE)

scene.save_xml(os.path.join(xml_path, save_filename))

virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)


simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers,
                            connect_to_optitrack=False)

simulator.video_save_folder = os.path.join(abs_path, "..")

simulator.cam.distance = 3.5
simulator.cam.azimuth = 0

drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
drone1 = simulator.get_MovingObject_by_name_in_xml(drone1_name)
drone2 = simulator.get_MovingObject_by_name_in_xml(drone2_name)
drone3 = simulator.get_MovingObject_by_name_in_xml(drone3_name)
drone4 = simulator.get_MovingObject_by_name_in_xml(drone4_name)
#drone2 = simulator.get_MovingObject_by_name_in_xml(drone2_name)

trajectory0 = RemoteDroneTrajectory(can_execute=False, init_pos=drone0_initpos)
trajectory1 = RemoteDroneTrajectory(can_execute=False, init_pos=drone1_initpos)
trajectory2 = RemoteDroneTrajectory(can_execute=False, init_pos=drone2_initpos)
trajectory3 = RemoteDroneTrajectory(can_execute=False, init_pos=drone3_initpos)
trajectory4 = RemoteDroneTrajectory(can_execute=False, init_pos=drone4_initpos)
#trajectory2 = RemoteDroneTrajectory(can_execute=False)

drone0.set_trajectory(trajectory0)
drone1.set_trajectory(trajectory1)
drone2.set_trajectory(trajectory2)
drone3.set_trajectory(trajectory3)
drone4.set_trajectory(trajectory4)

controller0 = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)
controller1 = GeomControl(drone1.mass, drone1.inertia, simulator.gravity)
controller2 = GeomControl(drone2.mass, drone2.inertia, simulator.gravity)
controller3 = GeomControl(drone3.mass, drone3.inertia, simulator.gravity)
controller4 = GeomControl(drone4.mass, drone4.inertia, simulator.gravity)

drone0.set_controllers([controller0])
drone1.set_controllers([controller1])
drone2.set_controllers([controller2])
drone3.set_controllers([controller3])
drone4.set_controllers([controller4])

td = TrajectoryDistributor(simulator.get_all_MovingObjects(), os.path.join(abs_path, "..", "SKYC_files"), False)
if td.connect("127.0.0.1", 7002):
#if td.connect("192.168.2.77", 7002):
    td.start_background_thread()

time0 = time.time()
while not simulator.glfw_window_should_close():

    simulator.update()

time_past = time.time() - time0

print("wall time: " + str(time_past))
print("simulator time: " + str(simulator.data.time))
print("i * control_step: " + str(simulator.i * simulator.control_step))
    
simulator.close()