import os
from aiml_virtual.xml_generator import xml_generator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.controller import GeomControl
from aiml_virtual.object.drone import BUMBLEBEE_PROP, DRONE_TYPES
from aiml_virtual.airflow import AirflowSampler
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.trajectory.trajectory_base import TrajectoryBase
import numpy as np
from aiml_virtual.object.payload import Payload, PAYLOAD_TYPES
from aiml_virtual.util import plot_payload_and_airflow_volume


BLUE = "0.2 0.6 0.85 1.0"
TRANSPARENT_BLUE = "0.2 0.2 0.85 0.1"
BLACK = "0.1 0.1 0.1 1.0"

class DummyHoverTraj(TrajectoryBase):

    def __init__(self, load_mass, target_pos):
        super().__init__()
        self.load_mass = load_mass
        self.target_pos = target_pos
    
    
    def evaluate(self, state, i, time, control_step):

        self.output["load_mass"] = self.load_mass
        self.output["target_pos"] = self.target_pos
        self.output["target_rpy"] = np.array((0.0, 0.0, 0.0))
        self.output["target_vel"] = np.array((0.0, 0.0, 0.0))
        self.output["target_pos_load"] = np.array((0.0, 0.0, 0.0))
        self.output["target_eul"] = np.zeros(3)
        self.output["target_pole_eul"] = np.zeros(2)
        self.output["target_ang_vel"] = np.zeros(3)
        self.output["target_pole_ang_vel"] = np.zeros(2)
        return self.output

rod_length = float(BUMBLEBEE_PROP.ROD_LENGTH.value)

# ------- 1. -------
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base_terrain.xml"
save_filename = "built_scene.xml"

drone0_init_pos = np.array([0.0, -1.0, 3.0, 0])  # initial drone position and yaw angle
load0_mass = 0.1
load0_size = np.array([.07, .07, .03])
load0_initpos = np.array([drone0_init_pos[0], drone0_init_pos[1], drone0_init_pos[2] - (2 * load0_size[2]) - (rod_length + .2) ])

scene = xml_generator.SceneXmlGenerator(xml_base_file_name)
drone0_name = scene.add_drone("0 0 15", "1 0 0 0", BLUE, DRONE_TYPES.BUMBLEBEE)

#drone0_name = scene.add_drone(np.array2string(drone0_init_pos[0:3])[1:-1], "1 0 0 0", TRANSPARENT_BLUE, DRONE_TYPES.BUMBLEBEE_HOOKED, 1)
#drone_mocap0_name = scene.add_mocap_drone("1 1 1", "1 0 0 0", BLACK, DRONE_TYPES.BUMBLEBEE_HOOKED, 2)
#payload0_name = scene.add_payload(np.array2string(load0_initpos)[1:-1], np.array2string(load0_size)[1:-1], str(load0_mass), "1 0 0 0", BLUE, PAYLOAD_TYPES.Box)
#
scene.save_xml(os.path.join(xml_path, save_filename))


virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False)
simulator.cam.lookat = np.array((0., 0., 8.))
simulator.cam.distance = 3
#simulator.scroll_distance_step = 0.05

d0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
d0.set_controllers([GeomControl(d0.mass, d0.inertia, simulator.gravity)])
d0.set_trajectory(DummyHoverTraj(0, np.array((0., 0., 8.))))

ctrl3_max = 0

def is_greater_than(new_value, current_max):

    if new_value > current_max:
        return new_value
    
    return current_max

# ------- 7. -------
while not simulator.glfw_window_should_close():
    simulator.update()

simulator.close()

