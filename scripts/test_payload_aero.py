import os
import numpy as np
from classes import payload
from util import xml_generator
from classes.drone import Drone
from classes.payload import Payload
from classes.active_simulation import ActiveSimulator
#from classes.drone_classes.hooked_drone_trajectory import HookedDroneTrajectory
from classes.drone_classes.drone_geom_control import GeomControl
from classes.trajectory_base import DummyDroneTrajectory
from random import seed, random

class DummyHoverTraj(DummyDroneTrajectory):

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
        return self.output


RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"

abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene.xml"
save_filename = "built_scene.xml"

# Set scenario parameters
drone_init_pos = np.array([0.0, 0.0, 1.0, 0])  # initial drone position and yaw angle
load_mass = 0.2

# create xml with a drone and a car
scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)
drone0_name = scene.add_drone(np.array2string(drone_init_pos[0:3])[1:-2], "1 0 0 0", RED_COLOR, True, "bumblebee",
                                True, 1)
payload0_name = scene.add_load("0.0 0.0 0.23", ".1 .1 .1", str(load_mass), "1 0 0 0", BLUE_COLOR)


scene.save_xml(os.path.join(xml_path, save_filename))

virt_parsers = [Drone.parse, Payload.parse]

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)


simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers=None,
                            connect_to_optitrack=False)


drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)

drone0_trajectory = DummyHoverTraj(load_mass, drone_init_pos[0:3])
drone0_controller = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)


drone0_controllers = [drone0_controller]
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)

payload0 = simulator.get_MovingObject_by_name_in_xml(payload0_name)
i = 0


while not simulator.glfw_window_should_close():
    simulator.update(i)
    if i % 10 == 0:
        payload0.qfrc_applied[0] = (random() - .5) / 5.0
        payload0.qfrc_applied[1] = (random() - .5) / 5.0
        payload0.qfrc_applied[2] = (random() - .5) / 5.0
        payload0.qfrc_applied[3] = (random() - .5) / 20.0
        payload0.qfrc_applied[4] = (random() - .5) / 20.0
        payload0.qfrc_applied[5] = (random() - .5) / 20.0
    #print(payload0.get_minirectangle_data_at(0, 0))
    i += 1

simulator.close()