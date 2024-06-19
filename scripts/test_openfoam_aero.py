import os
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import xml_generator
from aiml_virtual.object.drone import Drone, DRONE_TYPES
from aiml_virtual.object.payload import Payload
from aiml_virtual.trajectory.trajectory_base import TrajectoryBase
from aiml_virtual.controller import LqrLoadControl, GeomControl
import numpy as np
import matplotlib.pyplot as plt
from aiml_virtual.airflow import AirflowSampler
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.util import plot_payload_and_airflow_volume

# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


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

rod_length = xml_generator.ROD_LENGTH

abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene_base.xml"
save_filename = "built_scene.xml"

# Set scenario parameters
drone0_init_pos = np.array([0.0, 0.0, 2, 0])
load0_mass = 0.25
load0_size = np.array([.07, .07, .04])
#load0_mass = 0.14
#load0_size = np.array([16.2, 11.8, 8.6]) / 100
load0_initpos = np.array([drone0_init_pos[0], drone0_init_pos[1], drone0_init_pos[2] - (2 * load0_size[2]) - .595])


# create the model as xml
scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)

drone0_name = scene.add_drone(np.array2string(drone0_init_pos[0:3])[1:-1], "1 0 0 0", BLUE_COLOR, DRONE_TYPES.BUMBLEBEE_HOOKED, 2)

payload0_name = scene.add_payload(np.array2string(load0_initpos)[1:-1], np.array2string(load0_size)[1:-1], str(load0_mass), "1 0 0 0", RED_COLOR)

scene.save_xml(os.path.join(xml_path, save_filename))
virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers,
                            connect_to_optitrack=False)

drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
payload0 = simulator.get_MovingObject_by_name_in_xml(payload0_name)

drone0_trajectory = DummyHoverTraj(payload0.mass, drone0_init_pos[0:3])
drone0_controller = LqrLoadControl(drone0.mass, drone0.inertia, simulator.gravity)
# drone0_controller = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)


drone0_controllers = [drone0_controller]
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)

#pressure_data_filename = os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_pressure_shifted.txt")
#velocity_data_filename = os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_velocity_shifted.txt")

#airflow_sampl0 = AirflowSampler(pressure_data_filename, drone0, velocity_data_filename)

pressure_data_filename = os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "openfoam_pressure.txt")
velocity_data_filename = os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "openfoam_velocity.txt")
airflow_sampl0 = AirflowSampler(pressure_data_filename, drone0, velocity_data_filename)

payload0.create_surface_mesh(0.00001)
payload0.add_airflow_sampler(airflow_sampl0)

simulator.pause()
while not simulator.glfw_window_should_close():
    simulator.update()

simulator.close()

plot_payload_and_airflow_volume(payload0, airflow_sampl0, "black")
