import os
import time
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.trajectory import CarTrajectory, HookedDroneNLTrajectory
from aiml_virtual.controller import CarLPVController, LtvLqrLoadControl
from aiml_virtual.util import mujoco_helper, carHeading2quaternion
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object.payload import PAYLOAD_TYPES, TeardropPayload, BoxPayload
from aiml_virtual.object.car import Fleet1Tenth
import numpy as np
import matplotlib.pyplot as plt
from aiml_virtual.object import parseMovingObjects
from scipy.spatial.transform import Rotation
from aiml_virtual.trajectory.car_path_point_generator import paperclip, dented_paperclip
from aiml_virtual.trajectory.trailer_predictor import TrailerPredictor



RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"
BLACK = "0.1 0.1 0.1 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_filename = "scene_base.xml"
save_filename = "built_scene.xml"

car_trajectory = CarTrajectory()
# define path points and build the path

# Scenario #1
# path_points = paperclip()
# drone_init_pos = np.array([-1, -1.13, 1.1, 0])  # initial drone position and yaw angle
# load_target_pos = np.array([-1, -1.13, 0.19])

# Scenario #2
path_points = dented_paperclip()
drone_init_pos = np.array([1, 1.13, 1.1, 0])  # initial drone position and yaw angle
load_target_pos = np.array([1, 1.13, 0.19])
car_trajectory.build_from_points_smooth_const_speed(path_points=path_points, path_smoothing=1e-4, path_degree=5,
                                                    virtual_speed=0.6)

car_pos = np.array([car_trajectory.pos_tck[1][0][0], car_trajectory.pos_tck[1][1][0], 0.052])
heading_smoothing_index = 5
car_heading = np.arctan2(car_trajectory.pos_tck[1][1][heading_smoothing_index] - car_trajectory.pos_tck[1][1][0],
                            car_trajectory.pos_tck[1][0][heading_smoothing_index] - car_trajectory.pos_tck[1][0][0])
car_quat = carHeading2quaternion(car_heading)
car_rot = np.array([[np.cos(car_heading), -np.sin(car_heading), 0],
                    [np.sin(car_heading), np.cos(car_heading), 0],
                    [0, 0, 1]])
payload_offset = np.array([-0.5, 0, 0.125])
payload_pos = car_pos + car_rot @ payload_offset
payload_quat = car_quat


# create xml with car, trailer, payload, and bumblebee
scene = SceneXmlGenerator(xml_base_filename)
car_name = scene.add_car(pos=np.array2string(car_pos)[1:-1], quat=car_quat, color=RED,
                            is_virtual=True, has_rod=False, has_trailer=True)
trailer_name = car_name + "_trailer"
load_mass = 0.1
payload_name = scene.add_payload(pos=np.array2string(payload_pos)[1:-1], size="0.05 0.05 0.05", mass=str(load_mass),
                                    quat=payload_quat, color=BLACK, type=PAYLOAD_TYPES.Box)

bb_name = scene.add_drone(np.array2string(drone_init_pos[0:3])[1:-2], "1 0 0 0", RED, DRONE_TYPES.BUMBLEBEE_HOOKED, 2)

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers 
virt_parsers = [parseMovingObjects]

control_step, graphics_step = 0.025, 0.05
xml_filename = os.path.join(xml_path, save_filename)

# recording interval for automatic video capture
#rec_interval=[1,25]
rec_interval = None # no video capture

simulator = ActiveSimulator(xml_filename, rec_interval, control_step, graphics_step)

simulator.onBoard_elev_offset = 20

# grabbing the drone and the car
car = simulator.get_MovingObject_by_name_in_xml(car_name)

car_controller = CarLPVController(car.mass, car.inertia)

car_controllers = [car_controller]

# setting trajectory and controller for car
car.set_trajectory(car_trajectory)
car.set_controllers(car_controllers)

bb = simulator.get_MovingObject_by_name_in_xml(bb_name)

bb_trajectory = HookedDroneNLTrajectory(plot_trajs=False)
bb_trajectory.set_control_step(control_step)
bb_trajectory.set_rod_length(bb.rod_length)
bb_controller = LtvLqrLoadControl(bb.mass, bb.inertia, simulator.gravity)

bb.set_controllers([bb_controller])
bb.set_trajectory(bb_trajectory)

# Simulate car + trailer to get payload trajectory
predictor = TrailerPredictor(car.trajectory)
init_state = np.hstack((car_pos, np.fromstring(car_quat, sep=" "),
                                     np.zeros(6), 0, 0, 0, 0,
                                     np.nan * payload_pos, np.nan * np.fromstring(payload_quat, sep=" ")))
car_state = {}
car_state["pos_x"] = car_pos[0]
car_state["pos_y"] = car_pos[1]
car_state["head_angle"] = car_heading
car_state["long_vel"] = 0
car_state["lat_vel"] = 0
car_state["yaw_rate"] = 0
payload_predicted_points = predictor.simulate(init_state, car_state, 0, 15)

def load_init_pos(t, t0):
    t_interp = predictor.simulator.control_step * np.arange(payload_predicted_points.shape[0])
    return [np.interp(t-t0, t_interp, dim) for dim in payload_predicted_points[:, :3].T]

def load_init_vel(t, t0):
    t_interp = predictor.simulator.control_step * np.arange(payload_predicted_points.shape[0])
    return [np.interp(t-t0, t_interp, dim) for dim in payload_predicted_points[:, 3:6].T]

'''plt.figure()
t_plot = np.linspace(0, 10, 100)
car_pos_arr = np.asarray([load_init_pos(t_, 0) for t_ in t_plot])
plt.plot(t_plot, car_pos_arr)
plt.figure()
plt.plot(car_pos_arr[:, 0], car_pos_arr[:, 1])
plt.show()'''

def load_init_yaw(t, t0):
    t_interp = predictor.simulator.control_step * np.arange(payload_predicted_points.shape[0])
    payload_yaw = Rotation.from_quat(payload_predicted_points[:, [7, 8, 9, 6]]).as_euler('xyz')[:, 2]
    for i in range(1, payload_yaw.shape[0]):
        if payload_yaw[i] < payload_yaw[i-1] - np.pi:
            payload_yaw[i:] += 2*np.pi
        elif payload_yaw[i] > payload_yaw[i-1] + np.pi:
            payload_yaw[i:] -= 2*np.pi
    return np.interp(t-t0, t_interp, payload_yaw)

'''plt.figure()
t_plot = np.linspace(0, 10, 100)
car_pos_arr = np.asarray([load_init_yaw(t_, 0) for t_ in t_plot])
plt.plot(t_plot, car_pos_arr)
plt.show()'''

# Plan trajectory
bb_trajectory.construct(drone_init_pos[0:3]-np.array([0, 0, 0.4]), drone_init_pos[3], [load_init_pos, load_init_vel], 
                        load_init_yaw, load_target_pos, 0, load_mass, grasp_speed=1.3)  # TODO: now grasp_speed is dummmy

# Compute control gains
bb_controller.setup_hook_up(bb_trajectory, hook_mass=0.001, payload_mass=load_mass)


while not simulator.glfw_window_should_close():
    simulator.update()
    #if simulator.i == int(5 / control_step):
    #    simulator.pause()

simulator.close()
