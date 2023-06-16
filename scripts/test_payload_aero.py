import os
import math
import numpy as np
import matplotlib.pylab as plt
from classes import payload
from classes import pressure_sampler
from util import xml_generator
from classes.drone import Drone
from classes.payload import PAYLOAD_TYPES, Payload, PayloadMocap
from classes.active_simulation import ActiveSimulator
#from classes.drone_classes.hooked_drone_trajectory import HookedDroneTrajectory
from classes.drone_classes.drone_geom_control import GeomControl
from classes.drone_classes.hooked_drone_lq_control import LqrLoadControl
from classes.trajectory_base import DummyDroneTrajectory
from random import seed, random
from classes.pressure_sampler import PressureSampler

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
        self.output["target_eul"] = np.zeros(3)
        self.output["target_pole_eul"] = np.zeros(2)
        self.output["target_ang_vel"] = np.zeros(3)
        self.output["target_pole_ang_vel"] = np.zeros(2)
        return self.output


RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene.xml"
save_filename = "built_scene.xml"

# Set scenario parameters
drone_init_pos = np.array([0.0, 0.0, 2.4, 0])  # initial drone position and yaw angle
load_mass = 0.020
load_size = np.array([.05, .05, .025])
load_initpos = np.array([drone_init_pos[0], drone_init_pos[1], drone_init_pos[2] - (2 * load_size[2]) - .57 ])

# create xml with a drone and a car
scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)
drone0_name = scene.add_drone(np.array2string(drone_init_pos[0:3])[1:-1], "1 0 0 0", RED_COLOR, True, "bumblebee",
                                True, 1)
#payload0_name = scene.add_load("0.0 0.0 0.83", ".8 .8 .3", str(load_mass), "1 0 0 0", BLUE_COLOR)
payload0_name = scene.add_load(np.array2string(load_initpos)[1:-1], np.array2string(load_size)[1:-1], str(load_mass), "1 0 0 0", BLUE_COLOR)

#tpmocap_name = scene.add_load("-2 -2 1", ".03 .03 .15", None, ".924 .383 0 0", RED_COLOR, PAYLOAD_TYPES.Box.value, True)


scene.save_xml(os.path.join(xml_path, save_filename))

virt_parsers = [Drone.parse, Payload.parse]
mocap_parsers = [PayloadMocap.parse]

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)


simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers,
                            connect_to_optitrack=False)


drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
payload0 = simulator.get_MovingObject_by_name_in_xml(payload0_name)
#tpmocap = simulator.get_MovingMocapObject_by_name_in_xml(tpmocap_name)

drone0_trajectory = DummyHoverTraj(payload0.mass, drone_init_pos[0:3])
drone0_controller = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)
# drone0_controller = LqrLoadControl(drone0.mass, drone0.inertia, simulator.gravity)


drone0_controllers = [drone0_controller]
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)


pressure_sampl = PressureSampler(os.path.join(abs_path, "..", "combined_data.txt"), drone0)
payload0.set_top_subdivision(10, 10)

i = 0
lsize=500
log_ff=[]

payload0.set_force_torque(np.array([0, 0, 0]), np.array([-0.02, 0, 0]))
while not simulator.glfw_window_should_close():
    simulator.update(i)
    
    force, torque = pressure_sampl.generate_forces(payload0)

    if i < lsize:
        total_force=math.sqrt(force[0]**2+force[1]**2+force[2]**2)
        log_ff.append(torque)
    
    payload0.set_force_torque(force, torque)
    i += 1


plt.plot(log_ff)
plt.show()

simulator.close()