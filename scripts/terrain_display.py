import os

from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.controller import LqrControl, GeomControl
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.trajectory.drone_keyboard_trajectory import DroneKeyboardTraj
import numpy as np
import math
from aiml_virtual.object import Radar

from aiml_virtual.util.mujoco_helper import radars_see_point, create_3D_bool_array

if os.name == 'nt':
    import win_precise_time as time
else:
    import time


BLUE = "0.2 0.6 0.85 1.0"
TRANSPARENT_BLUE = "0.2 0.2 0.85 0.1"
BLACK = "0.1 0.1 0.1 1.0"
GREEN = "0.2 0.6 0.2 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base_terrain_tiny.xml"
save_filename = "built_scene.xml"

radar0 = Radar(np.array((-45, 45, 25)), 20., 1.5, 50, 60, height_scale=0.5, tilt=-0.15, display_lobe=True)
radar1 = Radar(np.array((20, 38, 20)), 20., 2.5, 50, 60, height_scale=0.5, tilt=0.05, display_lobe=True)
radar2 = Radar(np.array((-36, -38, 9)), 15., 2.5, 50, 60, height_scale=0.5, tilt=-0.05, display_lobe=True)
#radar0 = Radar(np.array((0, 20, 20)), 10., 1.5, 50, 60, height_scale=0.5, tilt=-0.2, display_lobe=True)
#radar1 = Radar(np.array((20, 38, 20)), 20., 2.5, 50, 60, height_scale=0.5, tilt=0.0, display_lobe=True)
#radar2 = Radar(np.array((-36, -20, 9)), 15., 2.5, 50, 60, height_scale=0.5, tilt=0.0, display_lobe=True)
radar_on_board = Radar(np.array((0, 0, 0)), 3., 2.5, 50, 60, height_scale=1.0, tilt=0.0, display_lobe=True)

hover_height = radar0.pos[2]

radars = [radar0, radar1, radar2]
#radars = [radar0, radar1, radar2, radar_on_board]
#radars = [radar0]

drone0_initpos = np.array((radar0.pos[0] - (2 * radar0.a) - 1., radar0.pos[1], hover_height))
drone0_initpos = np.array((radar0.pos[0] - (2 * radar0.a) - 1., radar0.pos[1], 26))


scene = SceneXmlGenerator(xml_base_file_name)
scene.ground_geom_name = "terrain0"
drone0_name = scene.add_drone(np.array2string(drone0_initpos)[1:-1], "1 0 0 0", BLUE, DRONE_TYPES.BUMBLEBEE)

for radar in radars:
    radar_name = scene.add_radar_field(np.array2string(radar.pos)[1:-1], radar.color, radar.a, radar.exp, radar.rres, radar.res,
                                       radar.height_scale, radar.tilt, sampling="curv", display_lobe=radar.display_lobe)
    radar.set_name(radar_name)

scene.save_xml(os.path.join(xml_path, save_filename))


virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False, window_size=[1280, 720])
simulator.cam.lookat = np.array((0.0, 0.0, 0.0))
simulator.cam.distance = 50
simulator.cam.elevation = -90
simulator.scroll_distance_step = 2
simulator.right_button_move_scale = .01
simulator.camOnBoard.distance = 4
simulator.onBoard_elev_offset = 15

for radar in radars:
    radar.parse(simulator.model, simulator.data)

# get the height field
terrain_hfield = simulator.model.hfield("terrain0")

#create_2D_slice(slice_height, terrain_hfield, None)

bool_space_save_folder = os.path.join(abs_path, "..", "3D_bool_space")
t1 = time.time()
slices = create_3D_bool_array(terrain_hfield, None, bool_space_save_folder, save_images=True)
dt = time.time() - t1
print("Time passed: ", dt)

#print(slices)

trajectory = DroneKeyboardTraj(0, drone0_initpos)

trajectory.set_key_callbacks(simulator)

d0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
d0.set_controllers([GeomControl(d0.mass, d0.inertia, simulator.gravity)])
d0.set_trajectory(trajectory)

simulator.set_key_t_callback(d0.toggle_sphere_alpha)

radars_overlap_checklist = [radar0, radar1, radar2]

while not simulator.should_close():
    simulator.update()
    d0_pos = d0.get_state()["pos"]
    d0.scale_sphere(simulator)

    #radar_on_board.set_qpos(d0_pos)

    if simulator.i % 20 == 0:
        print(d0_pos)

    if radars_see_point(radars_overlap_checklist, d0_pos):
        simulator.append_title(" BUSTED")
        d0.set_sphere_color(np.array((.8, .2, .2)))
    else:
        simulator.reset_title()
        d0.reset_sphere_color()

simulator.close()

