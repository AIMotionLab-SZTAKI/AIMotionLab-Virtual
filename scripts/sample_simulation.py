import os
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.object.drone import Drone, DRONE_TYPES
from aiml_virtual.object.payload import Payload
from aiml_virtual.controller import GeomControl
from aiml_virtual.trajectory import TrajectoryBase
import numpy as np
import matplotlib.pyplot as plt
from aiml_virtual.airflow import AirflowSampler
from aiml_virtual.wind_flow.wind_sampler import WindSampler
from aiml_virtual.object import parseMovingObjects
import glob

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


RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"

abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene_base.xml"
save_filename = "built_scene.xml"

# Set scenario parameters
drone_init_pos = np.array([0, 0, 1.5, 0])  # initial drone position and yaw angle

# create xml with a drone and a car
scene = SceneXmlGenerator(xmlBaseFileName)
drone0_name = scene.add_drone(np.array2string(drone_init_pos[0:3] + np.array([0, 0, 0.4]))[1:-2], "1 0 0 0", RED_COLOR, DRONE_TYPES.CRAZYFLIE)

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers
virt_parsers = [parseMovingObjects]

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

# initializing simulator
simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers=None,
                            connect_to_optitrack=False)

# grabbing the drone and the car
drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)

drone0_trajectory = DummyHoverTraj(0.0, drone_init_pos[0:3])

wind_velocity_folder = os.path.join(abs_path, "..", "airflow_data", "wind_data")
wind_velocity_filenames = glob.glob(os.path.join(wind_velocity_folder, '*'))
wind_sampler = WindSampler(wind_velocity_filenames)
drone0.add_wind_sampler(wind_sampler)


# drone0_controller = LtvLqrLoadControl(drone0.mass, drone0.inertia, simulator.gravity)
drone0_controller = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)

drone0_controllers = [drone0_controller]

# setting update_controller_type method, trajectory and controller for drone0
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)

# start simulation

while not simulator.glfw_window_should_close():
    simulator.update()


simulator.close()