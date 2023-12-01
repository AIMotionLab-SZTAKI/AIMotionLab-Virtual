import os
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.controller import GeomControl
from aiml_virtual.object.drone import BUMBLEBEE_PROP, DRONE_TYPES
from aiml_virtual.airflow import AirflowSampler
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.trajectory.trajectory_base import TrajectoryBase
import numpy as np
from aiml_virtual.object.payload import Payload, PAYLOAD_TYPES
from aiml_virtual.util import plot_payload_and_airflow_volume
from aiml_virtual.util import mujoco_helper


BLUE = "0.2 0.6 0.85 1.0"
TRANSPARENT_BLUE = "0.2 0.2 0.85 0.1"
BLACK = "0.1 0.1 0.1 1.0"

class DummyKeyboardTraj(TrajectoryBase):

    def __init__(self, load_mass, target_pos):
        super().__init__()
        self.load_mass = load_mass
        self.target_pos = np.copy(target_pos)
        self.target_rpy = np.array((0.0, 0.0, 0.0))

        self.speed = 0.04
        self.rot_speed = 0.004

        self.up_pressed = False
        self.down_pressed = False
        self.left_pressed = False
        self.right_pressed = False
        
        self.a_pressed = False
        self.s_pressed = False
        self.d_pressed = False
        self.w_pressed = False
    
    
    def evaluate(self, state, i, time, control_step):

        if self.up_pressed:
            self.move_forward(state)
        
        if self.down_pressed:
            self.move_backward(state)

        if self.left_pressed:
            self.move_left(state)

        if self.right_pressed:
            self.move_right(state)
        
        if self.a_pressed:
            self.rot_left(state)
        
        if self.d_pressed:
            self.rot_right(state)
        
        if self.w_pressed:
            self.move_up(state)
            
        if self.s_pressed:
            self.move_down(state)

        self.output["load_mass"] = self.load_mass
        self.output["target_pos"] = self.target_pos
        self.output["target_rpy"] = self.target_rpy
        self.output["target_vel"] = np.array((0.0, 0.0, 0.0))
        self.output["target_pos_load"] = np.array((0.0, 0.0, 0.0))
        self.output["target_eul"] = np.zeros(3)
        self.output["target_pole_eul"] = np.zeros(2)
        self.output["target_ang_vel"] = np.zeros(3)
        self.output["target_pole_ang_vel"] = np.zeros(2)
        return self.output

    def up_press(self):
        self.up_pressed = True
    
    def up_release(self):
        self.up_pressed = False
    
    def down_press(self):
        self.down_pressed = True
    
    def down_release(self):
        self.down_pressed = False
    
    def left_press(self):
        self.left_pressed = True
    
    def left_release(self):
        self.left_pressed = False
    
    def right_press(self):
        self.right_pressed = True
    
    def right_release(self):
        self.right_pressed = False
    
    def a_press(self):
        self.a_pressed = True
    
    def a_release(self):
        self.a_pressed = False
    
    def s_press(self):
        self.s_pressed = True
    
    def s_release(self):
        self.s_pressed = False
    
    def d_press(self):
        self.d_pressed = True
    
    def d_release(self):
        self.d_pressed = False
    
    def w_press(self):
        self.w_pressed = True
    
    def w_release(self):
        self.w_pressed = False
    
    def move_forward(self, state):
        diff = mujoco_helper.qv_mult(state["quat"], np.array((self.speed, 0.0, 0.0)))
        diff[2] = 0.0
        self.target_pos += diff
        
    def move_backward(self, state):
        diff = mujoco_helper.qv_mult(state["quat"], np.array((self.speed, 0.0, 0.0)))
        diff[2] = 0.0
        self.target_pos -= diff
    
    def move_left(self, state):
        diff = mujoco_helper.qv_mult(state["quat"], np.array((0.0, self.speed, 0.0)))
        diff[2] = 0.0
        self.target_pos += diff
    
    def move_right(self, state):
        diff = mujoco_helper.qv_mult(state["quat"], np.array((0.0, self.speed, 0.0)))
        diff[2] = 0.0
        self.target_pos -= diff
    
    def move_up(self, state):

        self.target_pos[2] += self.speed
    
    def move_down(self, state):

        self.target_pos[2] -= self.speed

    def rot_left(self, state):

        self.target_rpy[2] += self.rot_speed
        print(self.target_rpy)
    
    def rot_right(self, state):

        self.target_rpy[2] -= self.rot_speed
        print(self.target_rpy)


rod_length = float(BUMBLEBEE_PROP.ROD_LENGTH.value)

hover_height = 250

# ------- 1. -------
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base_terrain.xml"
save_filename = "built_scene.xml"


scene = SceneXmlGenerator(xml_base_file_name)
drone0_name = scene.add_drone("0 0 " + str(hover_height + 15), "1 0 0 0", BLUE, DRONE_TYPES.BUMBLEBEE)
scene.add_radar_field("-2000 2000 1000", ".5 .2 .2 0.5")
#scene.add_radar_field("-1.6 10.35 6.5", "0.1 0.8 0.1 1.0")
#scene.add_radar_field("2.11 -7.65 5.40")

scene.save_xml(os.path.join(xml_path, save_filename))


virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False)
simulator.cam.lookat = np.array((0., 0., hover_height))
simulator.cam.distance = 2
simulator.cam.azimuth = 0
simulator.scroll_distance_step = 10
simulator.right_button_move_scale = 1
simulator.camOnBoard.distance = 2
simulator.onBoard_elev_offset = 15

trajectory = DummyKeyboardTraj(0, np.array((0., 0., hover_height)))

simulator.set_key_up_callback(trajectory.up_press)
simulator.set_key_up_release_callback(trajectory.up_release)
simulator.set_key_down_callback(trajectory.down_press)
simulator.set_key_down_release_callback(trajectory.down_release)
simulator.set_key_left_callback(trajectory.left_press)
simulator.set_key_left_release_callback(trajectory.left_release)
simulator.set_key_right_callback(trajectory.right_press)
simulator.set_key_right_release_callback(trajectory.right_release)
simulator.set_key_a_callback(trajectory.a_press)
simulator.set_key_a_release_callback(trajectory.a_release)
simulator.set_key_s_callback(trajectory.s_press)
simulator.set_key_s_release_callback(trajectory.s_release)
simulator.set_key_d_callback(trajectory.d_press)
simulator.set_key_d_release_callback(trajectory.d_release)
simulator.set_key_w_callback(trajectory.w_press)
simulator.set_key_w_release_callback(trajectory.w_release)

d0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
d0.set_controllers([GeomControl(d0.mass, d0.inertia, simulator.gravity)])
d0.set_trajectory(trajectory)

ctrl3_max = 0

def is_greater_than(new_value, current_max):

    if new_value > current_max:
        return new_value
    
    return current_max

# ------- 7. -------
while not simulator.glfw_window_should_close():
    simulator.update()
    #if simulator.i % 2 == 0:
    #    simulator.cam.azimuth -= 0.5

simulator.close()

