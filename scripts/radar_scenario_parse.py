from curses import def_prog_mode
import os
from aiml_virtual.util import mujoco_helper

from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.controller import LqrControl, GeomControl
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.trajectory.drone_keyboard_trajectory import DroneKeyboardTraj
import numpy as np
import math
from aiml_virtual.object import Radar
from aiml_virtual.util.mujoco_helper import radars_see_point


def parentheses_contents(string: str):
    stack = []
    for i, c in enumerate(string):
        if c == '[':
            stack.append(i)
        elif c == ']' and stack:
            start = stack.pop()
            yield (len(stack), string[start + 1: i])

class DroneParams:

    def __init__(self, size: np.array, position_idx: int, safe_sphere_radius: float) -> None:
        self.size = size
        self.position_idx = position_idx
        self.safe_sphere_radius = safe_sphere_radius


class RadarScenario:

    def __init__(self, sim_volume_size=np.array((0.0, 0.0, 0.0)), mountain_height=0.0, height_map_name="",
                 target_point_list=np.array(((0., 0., 0.), (0., 0., 0.))),
                 drone_param_list=[DroneParams(np.array((0.1, 0.1, 0.1)), 0, 0.0)],
                 radar_list=[Radar(np.array((0.0, 0.0, 0.0)), 1.0, 1.0, 50, 60)]) -> None:
        
        self.sim_volume_size = sim_volume_size
        self.mountain_height = mountain_height
        self.height_map_name = height_map_name
        self.target_point_list = target_point_list
        self.drone_param_list = drone_param_list
        self.radar_list = radar_list

    @staticmethod
    def parse_config_file(full_filename: str) -> "RadarScenario":

        file = open(full_filename,'r')

        line = file.readline().split('#')[0].strip()

        split_line = line.split(' ')

        volume_x = float(split_line[1])
        volume_y = float(split_line[3])
        volume_z = float(split_line[5])

        volume_size = np.array((volume_x, volume_y, volume_z))

        line = file.readline().split('#')[0].strip()

        split_line = line.split(' ')

        mountain_height = float(split_line[1])

        line = file.readline().split('#')[0].strip()
        
        height_map_filename = line

        line = file.readline().split('#')[0].strip()[1:-2]

        split_line = line.split("] ")

        target_point_list = []

        for s in split_line:
            point = np.fromstring(s[1:], sep=' ')
            target_point_list += [point]
        

        line = file.readline().split('#')[0].strip()

        drone_param_list = []
        
        for l in list(parentheses_contents(line)):
            if l[0] == 1:
                split_dp = l[1][1:-1].split('] [')

                size = np.fromstring(split_dp[0], sep=' ')
                position_idx = int(split_dp[1])
                safe_sphere_radius = float(split_dp[2])

                drone_param_list += [DroneParams(size, position_idx, safe_sphere_radius)]


        line = file.readline().split('#')[0].strip()

        radar_list = []

        for l in list(parentheses_contents(line)):

            if l[0] == 1:
                split_r = l[1][1:-1].split('] [')

                pos = np.fromstring(split_r[0], sep=' ')
                a = float(split_r[1])
                exp = float(split_r[2])
                height_scale = float(split_r[3])
                tilt = float(split_r[4])    

                radar_list += [Radar(pos, a, exp, 50, 60, height_scale, tilt, display_lobe=True)]


        return RadarScenario(volume_size, mountain_height, height_map_filename,
                             target_point_list, drone_param_list, radar_list)



abs_path = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(abs_path, "..", "radar_scenario.config")

radar_scenario = RadarScenario.parse_config_file(filename)

BLUE = "0.2 0.6 0.85 1.0"

abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base_terrain.xml"
save_filename = "built_scene.xml"


scene = SceneXmlGenerator(xml_base_file_name)

terrain_hfield_filename = os.path.join("heightmaps", radar_scenario.height_map_name)
terrain_size = radar_scenario.sim_volume_size / 2.
terrain_size[2] = radar_scenario.mountain_height

scene.add_terrain(terrain_hfield_filename, size=np.array2string(terrain_size)[1:-1])

drone_names = []
for dp in radar_scenario.drone_param_list:

    pos = np.array2string(radar_scenario.target_point_list[dp.position_idx])[1:-1]
    size = dp.size
    safe_sphere_radius = str(dp.safe_sphere_radius)

    drone_names += [scene.add_drone(pos, "1 0 0 0", BLUE, type=DRONE_TYPES.BUMBLEBEE, safety_sphere_size=safe_sphere_radius)]


for radar in radar_scenario.radar_list:

    name = scene.add_radar_field(np.array2string(radar.pos)[1:-1], radar.color, radar.a, radar.exp, radar.rres,
                                 radar.res, radar.height_scale, radar.tilt, sampling="curv", display_lobe=True)

    radar.set_name(name)


scene.save_xml(os.path.join(xml_path, save_filename))


virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers,
                            connect_to_optitrack=False, window_size=[1280, 720])
simulator.cam.lookat = np.array((0.0, 0.0, 800.0))
simulator.cam.distance = 10000
simulator.cam.elevation = -90
simulator.scroll_distance_step = 20
simulator.right_button_move_scale = .01
simulator.camOnBoard.distance = 4
simulator.onBoard_elev_offset = 15


for radar in radar_scenario.radar_list:
    radar.parse(simulator.model, simulator.data)


d0 = simulator.get_MovingObject_by_name_in_xml(drone_names[0])

trajectory = DroneKeyboardTraj(0, d0.get_qpos()[:3])
trajectory.speed = 30

trajectory.set_key_callbacks(simulator)

d0.set_controllers([GeomControl(d0.mass, d0.inertia, simulator.gravity)])
d0.set_trajectory(trajectory)


simulator.set_key_t_callback(d0.toggle_sphere_alpha)

radars_overlap_checklist = radar_scenario.radar_list
d0_safe_sphere_radius = radar_scenario.drone_param_list[0].safe_sphere_radius

while not simulator.should_close():
    simulator.update()

    d0_pos = d0.get_state()["pos"]
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
    #    print(d0_pos)

    if radars_see_point(radars_overlap_checklist, d0_pos):
        simulator.append_title(" BUSTED")
        d0.set_sphere_color(np.array((.8, .2, .2)))
    else:
        simulator.reset_title()
        d0.reset_sphere_color()

simulator.close()