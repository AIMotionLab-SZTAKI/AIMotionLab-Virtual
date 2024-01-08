import os
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.controller import LqrControl
from aiml_virtual.object.drone import BUMBLEBEE_PROP, DRONE_TYPES
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.trajectory.drone_keyboard_trajectory import DroneKeyboardTraj
import numpy as np
from aiml_virtual.util.mujoco_helper import Radar
import math


BLUE = "0.2 0.6 0.85 1.0"
TRANSPARENT_BLUE = "0.2 0.2 0.85 0.1"
BLACK = "0.1 0.1 0.1 1.0"


rod_length = float(BUMBLEBEE_PROP.ROD_LENGTH.value)

hover_height = 1010
radar_pos = np.array((-2200, 2200, hover_height))

# ------- 1. -------
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base_terrain.xml"
save_filename = "built_scene.xml"

radar0 = Radar(np.array((-2400, 2400, 1000)), 800., 1.5, 150, 90, height_scale=.5, tilt=-0.1)
#radar1 = Radar(np.array((20, 38, 21)), 20., 2.5, 50, 60)
#radar2 = Radar(np.array((-36, -38, 10)), 15., 2.5, 50, 60)

hover_height = radar0.pos[2] + 100.
#radars = [radar0, radar1, radar2]
radars = [radar0]

drone0_initpos = np.array((radar0.pos[0] - (2 * radar0.a) - 1.0, radar0.pos[1], hover_height))


scene = SceneXmlGenerator(xml_base_file_name)
drone0_name = scene.add_drone(np.array2string(drone0_initpos)[1:-1], "1 0 0 0", BLUE, DRONE_TYPES.BUMBLEBEE)

for radar in radars:
    scene.add_radar_field(np.array2string(radar.pos)[1:-1], radar.color, radar.a, radar.exp, radar.rres, radar.res, radar.height_scale, radar.tilt, sampling="curv")

scene.save_xml(os.path.join(xml_path, save_filename))


virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False, window_size=[1920, 1088])
simulator.cam.lookat = drone0_initpos
simulator.cam.distance = 2
simulator.cam.azimuth = 0
simulator.scroll_distance_step = 20
simulator.right_button_move_scale = .1
simulator.camOnBoard.distance = 2
simulator.onBoard_elev_offset = 15

trajectory = DroneKeyboardTraj(0, drone0_initpos)

trajectory.set_key_callbacks(simulator)

d0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
d0.set_controllers([LqrControl(d0.mass, d0.inertia, simulator.gravity)])
d0.set_trajectory(trajectory)

simulator.set_key_t_callback(d0.toggle_sphere_alpha)


while not simulator.glfw_window_should_close():
    simulator.update()
    d0_pos = d0.get_state()["pos"]
    #print()
    #print(d0.xquat)
    #print(d0.state["quat"])

    d0.scale_sphere(simulator)
    
    busted = False
    for radar in radars:
        if radar.sees_drone(d0):
            busted = True
            break
    
    if busted:
        simulator.append_title(" BUSTED")
        d0.set_sphere_color(np.array((.8, .2, .2)))
    else:
        simulator.reset_title()
        d0.reset_sphere_color()

simulator.close()

