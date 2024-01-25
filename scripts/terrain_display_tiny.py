import os

from sympy import true
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.controller import LqrControl, GeomControl
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.trajectory.drone_keyboard_trajectory import DroneKeyboardTraj
import numpy as np
import math
from aiml_virtual.object import Radar
from PIL import Image


def radars_see_point(radars, point):

    if radars is None:
        return False

    for radar in radars:
        if radar.sees_point(point):
            return True

    return False

def create_2D_slice(slice_height, terrain_hfield, radars=None):
    slice2D = np.empty(terrain_hfield.data.shape)
    dimensions = terrain_hfield.size[:3]
    x_offset = dimensions[0]
    y_offset = dimensions[1]
    
    for row in range(slice2D.shape[0]):
        
        y = row * (2 * dimensions[0] / slice2D.shape[0]) - y_offset

        for col in range(slice2D.shape[1]):
            
            height_value = dimensions[2] * terrain_hfield.data[row, col]

            x = col * (2 * dimensions[1] / slice2D.shape[1]) - x_offset

            #print("####")
            #print(row, col)
            #print(x, y, slice_height)

            if height_value >= slice_height or radars_see_point(radars, np.array((x, y, slice_height))):
                slice2D[row, col] = 1.
            else:
                slice2D[row, col] = 0.


    im = Image.fromarray(np.flip(slice2D, 0) * 255)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("slice_h" + str(slice_height) + ".png")

BLUE = "0.2 0.6 0.85 1.0"
TRANSPARENT_BLUE = "0.2 0.2 0.85 0.1"
BLACK = "0.1 0.1 0.1 1.0"
GREEN = "0.2 0.6 0.2 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base_terrain_tiny.xml"
save_filename = "built_scene.xml"

radar0 = Radar(np.array((-45, 45, 25)), 20., 1.5, 50, 60, height_scale=1., tilt=0.0)
radar1 = Radar(np.array((20, 38, 20)), 20., 2.5, 50, 60, height_scale=1., tilt=0.0)
radar2 = Radar(np.array((-36, -38, 9)), 15., 2.5, 50, 60, height_scale=1., tilt=0.0)

hover_height = radar0.pos[2]
radars = [radar0, radar1, radar2]
#radars = [radar0]

drone0_initpos = np.array((radar0.pos[0] - (2 * radar0.a) - 1., radar0.pos[1], hover_height))


scene = SceneXmlGenerator(xml_base_file_name)
scene.ground_geom_name = "terrain0"
drone0_name = scene.add_drone(np.array2string(drone0_initpos)[1:-1], "1 0 0 0", BLUE, DRONE_TYPES.BUMBLEBEE)
car0_name = scene.add_car("0 0 10", "1 0 0 0", GREEN, True, False)

for radar in radars:
    scene.add_radar_field(np.array2string(radar.pos)[1:-1], radar.color, radar.a, radar.exp, radar.rres, radar.res, radar.height_scale, radar.tilt, sampling="curv")

scene.save_xml(os.path.join(xml_path, save_filename))


virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False, window_size=[1280, 720])
simulator.cam.lookat = drone0_initpos
simulator.cam.distance = 2
simulator.cam.azimuth = -90
simulator.scroll_distance_step = 2
simulator.right_button_move_scale = .01
simulator.camOnBoard.distance = 4
simulator.onBoard_elev_offset = 15

# get the height field
terrain_hfield = simulator.model.hfield("terrain0")
slice_height = 4 # in meters

create_2D_slice(slice_height, terrain_hfield, None)


trajectory = DroneKeyboardTraj(0, drone0_initpos)

trajectory.set_key_callbacks(simulator)

d0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
d0.set_controllers([GeomControl(d0.mass, d0.inertia, simulator.gravity)])
d0.set_trajectory(trajectory)

simulator.set_key_t_callback(d0.toggle_sphere_alpha)


while not simulator.glfw_window_should_close():
    simulator.update()
    d0_pos = d0.get_state()["pos"]
    d0.scale_sphere(simulator)

    #if simulator.i % 20 == 0:
    #    print(d0_pos)
    
    if radars_see_point(radars, d0_pos):
        simulator.append_title(" BUSTED")
        d0.set_sphere_color(np.array((.8, .2, .2)))
    else:
        simulator.reset_title()
        d0.reset_sphere_color()

simulator.close()

