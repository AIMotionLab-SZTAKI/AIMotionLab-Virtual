import os
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.object.drone import Drone, DRONE_TYPES
from aiml_virtual.object.payload import Payload
from aiml_virtual.controller import GeomControl
from aiml_virtual.trajectory import TrajectoryBase
import numpy as np
import matplotlib.pyplot as plt
from aiml_virtual.wind_flow.wind_sampler import WindSampler
from aiml_virtual.object import parseMovingObjects


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
    
class TestTraj(TrajectoryBase):

    def __init__(self, target_z, traj_freq=1, target_yaw=1.5):
        super().__init__()
        self.target_z = target_z
        self.traj_freq = traj_freq
        self.target_yaw = target_yaw
    
    
    def evaluate(self, state, i, time, control_step):

        self.output["load_mass"] = 0.0
        self.output["target_pos"] = np.array([0.8 * np.sin(self.traj_freq * time), 0.8 * np.sin(2 * self.traj_freq * (time - np.pi / 2)), self.target_z])
        self.output["target_rpy"] = np.array([0, 0, self.target_yaw])
        self.output["target_vel"] = np.array([0.8 * self.traj_freq * np.cos(self.traj_freq * time), 0.8 * 2 * self.traj_freq * np.cos(2 * self.traj_freq * (time - np.pi / 2)), 0])
        self.output["target_pos_load"] = None
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

drone0_z = 1.35
drone1_z = 1.35

# Set scenario parameters
drone0_init_pos = np.array([0, 0, drone0_z, 0])  # initial drone position and yaw angle
drone1_init_pos = np.array([0, 0, drone1_z, 0])  # initial drone position and yaw angle

# create xml with a drone and a car
scene = SceneXmlGenerator(xmlBaseFileName)
drone0_name = scene.add_drone(np.array2string(drone0_init_pos[0:3])[1:-2], "1 0 0 0", RED_COLOR, DRONE_TYPES.CRAZYFLIE)
drone1_name = scene.add_drone(np.array2string(drone1_init_pos[0:3])[1:-2], "1 0 0 0", BLUE_COLOR, DRONE_TYPES.CRAZYFLIE)

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
drone1 = simulator.get_MovingObject_by_name_in_xml(drone1_name)

#drone0_trajectory = DummyHoverTraj(0.0, drone_init_pos[0:3])
drone0_trajectory = TestTraj(drone0_z)
drone1_trajectory = TestTraj(drone1_z)

# drone0_controller = LtvLqrLoadControl(drone0.mass, drone0.inertia, simulator.gravity)
drone0_controller = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)
drone1_controller = GeomControl(drone1.mass, drone1.inertia, simulator.gravity)

drone0_controllers = [drone0_controller]
drone1_controllers = [drone1_controller]

# setting update_controller_type method, trajectory and controller for drone0
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)
drone1.set_trajectory(drone1_trajectory)
drone1.set_controllers(drone1_controllers)

wind_velocity_filename = os.path.join(abs_path, "..", "airflow_data", "wind_data", "wind_1ms_0deg.csv")
wind_sampler = WindSampler(wind_velocity_filename)
drone0.add_wind_sampler(wind_sampler)

# start simulation

time = []
drone0_ex = []
drone0_ey = []
drone1_ex = []
drone1_ey = []

t = 0
t_end = 10

while t < t_end and not simulator.glfw_window_should_close():
    simulator.update()
    t = simulator.i * simulator.control_step
    target = drone0_trajectory.get_target_pos()
    drone0_pos = drone0.get_state()["pos"]
    drone1_pos = drone1.get_state()["pos"]
    time += [t]
    drone0_ex += [target[0] - drone0_pos[0]]
    drone0_ey += [target[1] - drone0_pos[1]]
    drone1_ex += [target[0] - drone1_pos[0]]
    drone1_ey += [target[1] - drone1_pos[1]]


simulator.close()

fig, ax = plt.subplots(2, 1, sharex=True, layout="tight")
ax[0].plot(time, drone0_ex)
ax[0].plot(time, drone1_ex)
ax[0].set_ylabel(r"$x$ error [m]")
ax[0].legend(["With wind", "Without wind"])

ax[1].plot(time, drone0_ey)
ax[1].plot(time, drone1_ey)
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel(r"$y$ error [m]")


plt.show()