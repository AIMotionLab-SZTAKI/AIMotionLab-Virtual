import os
from aiml_virtual.util import mujoco_helper

from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.controller import LqrControl, GeomControl
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.trajectory.drone_keyboard_trajectory import DroneKeyboardTraj
from aiml_virtual.trajectory import RemoteDroneTrajectory
import numpy as np
import math
from aiml_virtual.object import Radar
from aiml_virtual.util.mujoco_helper import radars_see_point
from aiml_virtual.scenario import RadarScenario


abs_path = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(abs_path, "..", "radar_scenario.config")

RadarScenario.radar_stl_resolution = 50
RadarScenario.radar_stl_rot_resolution = 60
radar_scenario = RadarScenario.parse_config_file(filename)

BLUE = "0.2 0.6 0.85 1.0"

xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base_terrain.xml"

xml_filename, drone_names = radar_scenario.generate_xml(xml_base_file_name, xml_path)

virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, xml_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers,
                            connect_to_optitrack=False, window_size=[1280, 720])
simulator.cam.lookat = np.array((0.0, 0.0, 800.0))
simulator.cam.distance = 10000
simulator.cam.elevation = -90
simulator.scroll_distance_step = 50
simulator.right_button_move_scale = 10
simulator.camOnBoard.distance = 4
simulator.onBoard_elev_offset = 15


for radar in radar_scenario.radar_list:
    radar.parse(simulator.model, simulator.data)


d0 = simulator.get_MovingObject_by_name_in_xml(drone_names[0])


trajectory = DroneKeyboardTraj(0, d0.get_qpos()[:3])
trajectory.speed = 30
trajectory.set_key_callbacks(simulator)

#trajectory = RemoteDroneTrajectory(directory=os.path.join(abs_path, "..", ""))


d0.set_controllers([GeomControl(d0.mass, d0.inertia, simulator.gravity)])
d0.set_trajectory(trajectory)


simulator.set_key_t_callback(d0.toggle_sphere_alpha)
simulator.set_key_b_callback(d0.toggle_safety_sphere_alpha)

radars_overlap_checklist = radar_scenario.radar_list
d0_safe_sphere_radius = radar_scenario.drone_param_list[0].safe_sphere_radius

while not simulator.should_close():
    simulator.update()

    d0_pos = d0.get_state()["pos"]
    d0_vel = d0.get_state()["vel"]
    d0.scale_sphere(simulator)

    d0_target_pos = d0.trajectory.get_target_pos()
    dist_pos_target_pos = mujoco_helper.distance(d0_pos, d0_target_pos)

    if dist_pos_target_pos > d0_safe_sphere_radius:
        d0.set_safety_sphere_color(np.array((.8, .2, .2)))
    else:
        d0.reset_safety_sphere_color()

    d0.set_safety_sphere_pos(d0_target_pos)

    #radar_on_board.set_qpos(d0_pos)

    #if simulator.i % 20 == 0:
    #    print(d0_vel)

    if radars_see_point(radars_overlap_checklist, d0_pos):
        simulator.append_title(" BUSTED")
        d0.set_sphere_color(np.array((.8, .2, .2)))
    else:
        simulator.reset_title()
        d0.reset_sphere_color()

simulator.close()