import os
import math
import numpy as np
#import matplotlib.pylab as plt
from util import xml_generator
from classes.active_simulation import ActiveSimulator
#from classes.drone_classes.hooked_drone_trajectory import HookedDroneTrajectory
from classes.drone_classes.drone_geom_control import GeomControl
from classes.drone_classes.hooked_drone_lq_control import LqrLoadControl
from classes.trajectory_base import DummyDroneTrajectory
from random import seed, random
from classes.airflow_sampler import AirflowSampler
from classes.object_parser import parseMovingObjects


# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


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
drone0_init_pos = np.array([0.0, -1.0, 1.4, 0])  # initial drone position and yaw angle
load0_mass = 0.10
load0_size = np.array([.07, .07, .07])
load0_initpos = np.array([drone0_init_pos[0], drone0_init_pos[1], drone0_init_pos[2] - (2 * load0_size[2]) - .57 ])

# create xml with two drones
scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)
drone0_name = scene.add_drone(np.array2string(drone0_init_pos[0:3])[1:-1], "1 0 0 0", RED_COLOR, True, "bumblebee",
                                True, 1)

payload0_name = scene.add_load(np.array2string(load0_initpos)[1:-1], np.array2string(load0_size)[1:-1], str(load0_mass), "1 0 0 0", BLUE_COLOR)

# Set scenario parameters
drone1_init_pos = np.array([0.0, 1.0, 1.4, 0])  # initial drone position and yaw angle
load1_mass = 0.10
load1_size = np.array([.07, .07, .07])
load1_initpos = np.array([drone1_init_pos[0], drone1_init_pos[1], drone1_init_pos[2] - (2 * load1_size[2]) - .57 ])

drone1_name = scene.add_drone(np.array2string(drone1_init_pos[0:3])[1:-1], "1 0 0 0", RED_COLOR, True, "bumblebee",
                                True, 1)

payload1_name = scene.add_load(np.array2string(load1_initpos)[1:-1], np.array2string(load1_size)[1:-1], str(load1_mass), "1 0 0 0", BLUE_COLOR)

scene.save_xml(os.path.join(xml_path, save_filename))

virt_parsers = [parseMovingObjects]
mocap_parsers = None

control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)


simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers,
                            connect_to_optitrack=False)


drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
payload0 = simulator.get_MovingObject_by_name_in_xml(payload0_name)
drone1 = simulator.get_MovingObject_by_name_in_xml(drone1_name)
payload1 = simulator.get_MovingObject_by_name_in_xml(payload1_name)
#tpmocap = simulator.get_MovingMocapObject_by_name_in_xml(tpmocap_name)

drone0_trajectory = DummyHoverTraj(payload0.mass, drone0_init_pos[0:3])
drone0_controller = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)
drone1_trajectory = DummyHoverTraj(payload1.mass, drone1_init_pos[0:3])
drone1_controller = GeomControl(drone1.mass, drone1.inertia, simulator.gravity)
# drone0_controller = LqrLoadControl(drone0.mass, drone0.inertia, simulator.gravity)


drone0_controllers = [drone0_controller]
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)

drone1_controllers = [drone1_controller]
drone1.set_trajectory(drone1_trajectory)
drone1.set_controllers(drone1_controllers)

pressure_data_filename = os.path.join(abs_path, "..", "airflow_luts", "flow_pressure_shifted.txt")
velocity_data_filename = os.path.join(abs_path, "..", "airflow_luts", "flow_velocity_shifted.txt")

airflow_sampl0 = AirflowSampler(pressure_data_filename, drone0, velocity_data_filename)
payload0.set_top_subdivision(30, 30)
payload0.set_side_subdivision(30, 30, 30)

airflow_sampl1 = AirflowSampler(pressure_data_filename, drone1)
payload1.set_top_subdivision(30, 30)
payload1.set_side_subdivision(30, 30, 30)

i = 0
lsize=500
log_ff=[]


#payload0.set_force_torque(np.array([0, 0, 0]), np.array([-0.04, .04, 0]))
while not simulator.glfw_window_should_close():
    simulator.update(i)

    #print(str(drone1.ctrl0) + " " + str(drone1.ctrl1) + " " + str(drone1.ctrl2) + " " + str(drone1.ctrl3))
    
    force0, torque0 = airflow_sampl0.generate_forces_opt(payload0)
    force1, torque1 = airflow_sampl1.generate_forces_opt(payload1)

    #if i < lsize:
    #    total_force=math.sqrt(force[0]**2+force[1]**2+force[2]**2)
    #    log_ff.append(torque)

    #if i % 20 == 0:
    #    force_opt, torque_opt = airflow_sampl.generate_forces(payload0)
    #    
    #    print("F: " + str(force) + "   " + str(force_opt))
    #    print("M: " + str(torque) + "   " + str(torque_opt))

    
    payload0.set_force_torque(force0 / 2., torque0 / 2.)
    payload1.set_force_torque(force1 / 2., torque1 / 2.)
    i += 1

simulator.close()


p, pos, n, a = payload0.get_top_minirectangle_data()

p[:, 2] += airflow_sampl0.shift_payload_up_meter

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(p[:, 0], p[:, 1], p[:, 2])

p_n, p_p, pown_n, pown_p, n_n, n_p, area_xz = payload0.get_side_xz_minirectangle_data()
p_n[:, 2] += airflow_sampl0.shift_payload_up_meter
p_p[:, 2] += airflow_sampl0.shift_payload_up_meter
ax.scatter(p_n[:, 0], p_n[:, 1], p_n[:, 2])
ax.scatter(p_p[:, 0], p_p[:, 1], p_p[:, 2])

p_n, p_p, pown_n, pown_p, n_n, n_p, area_xz = payload0.get_side_yz_minirectangle_data()
p_n[:, 2] += airflow_sampl0.shift_payload_up_meter
p_p[:, 2] += airflow_sampl0.shift_payload_up_meter
ax.scatter(p_n[:, 0], p_n[:, 1], p_n[:, 2])
ax.scatter(p_p[:, 0], p_p[:, 1], p_p[:, 2])


faces = []
faces.append(np.zeros([5,3]))
faces.append(np.zeros([5,3]))
faces.append(np.zeros([5,3]))
faces.append(np.zeros([5,3]))
faces.append(np.zeros([5,3]))
faces.append(np.zeros([5,3]))


vs = airflow_sampl0.get_transformed_vertices()


# Bottom face
faces[0][0, :] = np.array(vs[0])
faces[0][1, :] = np.array(vs[2])
faces[0][2, :] = np.array(vs[3])
faces[0][3, :] = np.array(vs[1])
faces[0][4, :] = np.array(vs[0])

# Top face
faces[1][0, :] = np.array(vs[4])
faces[1][1, :] = np.array(vs[6])
faces[1][2, :] = np.array(vs[7])
faces[1][3, :] = np.array(vs[5])
faces[1][4, :] = np.array(vs[4])


ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.25))
ax.axis("equal")

plt.show()

#pressure_sampl.generate_forces_opt(payload0)

#plt.plot(log_ff)
#plt.show()
