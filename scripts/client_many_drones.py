import math
from aiml_virtual.trajectory import TrajectoryDistributor, RemoteDroneTrajectory
from aiml_virtual.controller import GeomControl
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object import parseMovingObjects
import os
import numpy as np
if os.name == 'nt':
    import win_precise_time as time
else:
    import time

RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"

NUM_ROWS_CF = 5
NUM_COLS_CF = 6
DIST_ROWS = 0.3
DIST_COLS = 0.3


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene_without_walls.xml"
save_filename = "built_scene.xml"


scene = SceneXmlGenerator(xmlBaseFileName)

drone_names = []

for i in range(NUM_ROWS_CF):
    c_r = str(float(i) / NUM_ROWS_CF)
    for j in range(NUM_COLS_CF):
        c_b = str(float(j) / NUM_COLS_CF)
        drone_initpos = np.array((DIST_COLS * j - (((NUM_COLS_CF - 1) * DIST_COLS) / 2.0), DIST_ROWS * i - (((NUM_ROWS_CF - 1) * DIST_ROWS) / 2.0), 0.1))
        color = c_r + " 0.1 " + c_b + " 1.0" 
        drone_names += [scene.add_drone(np.array2string(drone_initpos)[1:-1], "1 0 0 0", color, DRONE_TYPES.BUMBLEBEE)]


scene.save_xml(os.path.join(xml_path, save_filename))

virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)


simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers,
                            connect_to_optitrack=False, window_size=[1200, 800])

simulator.cam.distance = 2.5
simulator.cam.azimuth = 0

drones = []
x = -1.0
hstep = 1.0 / (NUM_COLS_CF * NUM_ROWS_CF)
h = 0.0
i = 1.0
j = 1.0
for name in drone_names:
    #print(i)
    #h = (math.sin(i / 2.0) * math.cos(j / 2.0) + 1.1) / 2.0
    #i += 1.0
    #if i > NUM_COLS_CF:
    #    j += 1.0
    #    i = 0.0
    drone = simulator.get_MovingObject_by_name_in_xml(name)
    init_pos = drone.get_qpos()[:3].copy()
    trajectory = RemoteDroneTrajectory(can_execute=False, init_pos=init_pos) # + np.array((0.0, 0.0, h)))
    controller = GeomControl(drone.mass, drone.inertia, simulator.gravity)
    drone.set_trajectory(trajectory)
    drone.set_controllers([controller])
    x += 0.2


td = TrajectoryDistributor(simulator.get_all_MovingObjects(), os.path.join(abs_path, "..", "SKYC_files"), False)
if td.connect("127.0.0.1", 7002):
#if td.connect("192.168.2.77", 7002):
    td.start_background_thread()

d0 = simulator.get_MovingObject_by_name_in_xml(drone_names[0])
while not simulator.glfw_window_should_close():

    #print(d0.trajectory.evaluate_trajectory(0.0))
    simulator.update()

time_past = time.time() - simulator.start_time
print("wall time: " + str(time_past))
print("simulator time: " + str(simulator.data.time))
print("i * control_step: " + str(simulator.i * simulator.control_step))
simulator.close()