import os

from aiml_virtual.simulator import ActiveSimulator

from aiml_virtual.xml_generator import SceneXmlGenerator

from aiml_virtual.trajectory import CarTrajectory, CarTrajectorySpatial, HookedDronePolyTrajectory
from aiml_virtual.controller import CarLPVController
from aiml_virtual.controller import LtvLqrLoadControl
from aiml_virtual.util import mujoco_helper, carHeading2quaternion
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object.payload import PAYLOAD_TYPES
import numpy as np
import matplotlib.pyplot as plt
import scipy

from aiml_virtual.object import parseMovingObjects


RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"
BLACK = "0.1 0.1 0.1 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_filename = "scene_base.xml"
save_filename = "built_scene.xml"


# create xml with a car
scene = SceneXmlGenerator(xml_base_filename)
car0_name = scene.add_car(pos="0 0 0.052", quat=carHeading2quaternion(0.64424), color=RED, is_virtual=True, has_rod=False, has_trailer=True)
load_mass = 0.1
payload0_name = scene.add_payload("-.4 -.3 .18", "0.05 0.05 0.05", str(load_mass), carHeading2quaternion(0.64424), BLACK, PAYLOAD_TYPES.Teardrop)

drone_init_pos = np.array([0.76, -1.13, 1, 0])  # initial drone position and yaw angle
load_init_pos = np.array([0.05309104, -0.08713943, 0.15795468]) + np.array([0, 0, 0.15])
load_init_yaw = 2.451028352314646
load_target_pos = np.array([0.76, 1.13, 0.19])
grasp_speed = 1.5
bb0_name = scene.add_drone(np.array2string(drone_init_pos[0:3])[1:-2], "1 0 0 0", RED, DRONE_TYPES.BUMBLEBEE_HOOKED, 2)
 

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers 
virt_parsers = [parseMovingObjects]



control_step, graphics_step = 0.025, 0.05
xml_filename = os.path.join(xml_path, save_filename)

# recording interval for automatic video capture
#rec_interval=[0,22]
rec_interval = None # no video capture

simulator = ActiveSimulator(xml_filename, rec_interval, control_step, graphics_step, window_size=[3840, 2160])
simulator.cam.azimuth = 45
simulator.cam.elevation = -25
simulator.cam.lookat, simulator.cam.distance = [-.5, -.5, .5], 4
simulator.onBoard_elev_offset = 20

# grabbing the drone and the car
car0 = simulator.get_MovingObject_by_name_in_xml(car0_name)

car0_trajectory=CarTrajectorySpatial()

# define path points and build the path
path_points = np.array(
    [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 2],
        [4, 1],
        [4.5, 0],
        [4, -1],
        [3, -2],
        [2, -2],
        [1, -1],
        [0, 0],
        [-1, 1],
        [-2, 2],
        [-3, 2],
        [-4, 1],
        [-4.5, 0],
        [-4, -2.1],
        [-3, -2.3],
        [-2, -2],
        [-1, -1],
        [0, 0],
    ]
)

path_points /= 1.5

car0_trajectory.build_from_points_const_speed(path_points=path_points, path_smoothing=0.01, path_degree=4, const_speed=1., start_delay=1.0)
#car0_trajectory.plot_trajectory()

car0_controller = CarLPVController(car0.mass, car0.inertia)

car0_controllers = [car0_controller]


# setting trajectory and controller for car0
car0.set_trajectory(car0_trajectory)
car0.set_controllers(car0_controllers)

bb0 = simulator.get_MovingObject_by_name_in_xml(bb0_name)

bb0_trajectory = HookedDronePolyTrajectory()
bb0_trajectory.set_control_step(control_step)
bb0_trajectory.set_rod_length(bb0.rod_length)
bb0_controller = LtvLqrLoadControl(bb0.mass, bb0.inertia, simulator.gravity)

bb0.set_controllers([bb0_controller])
bb0.set_trajectory(bb0_trajectory)

# Plan trajectory
bb0_trajectory.construct(drone_init_pos[0:3], drone_init_pos[3], load_init_pos, load_init_yaw,
                         load_target_pos, 0, load_mass, grasp_speed)

# Compute control gains
bb0_controller.setup_hook_up(bb0_trajectory, hook_mass=0.01, payload_mass=load_mass)


while not simulator.glfw_window_should_close():
    simulator.update()
    if abs(simulator.data.time - 10) < 0.01:
        print(f"Payload position: {repr(simulator.all_moving_objects[1].sensor_posimeter)}, yaw: "
              f"{scipy.spatial.transform.Rotation.from_quat(np.roll(simulator.all_moving_objects[1].sensor_orimeter, -1)).as_euler('xyz')[2]},"
              f"and speed: {np.linalg.norm(simulator.all_moving_objects[1].data.qvel[0:3])} at time: {simulator.data.time}")
simulator.close()