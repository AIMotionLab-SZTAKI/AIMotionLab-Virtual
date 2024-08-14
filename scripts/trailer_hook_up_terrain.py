import os
import time
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.trajectory import CarTrajectory, HookedDroneNLTrajectory, HookedDroneNLAdaptiveTrajectory
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
import threading as thr
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import splev


def replan_parallel(bb_traj_temp: HookedDroneNLAdaptiveTrajectory, 
                    bb_ctrl_temp: LtvLqrLoadControl, init_state, init_time, 
                    duration, num_ter, num_traj_to_compute):
    load_init_pos_new, load_init_vel_new, load_init_yaw_new, _ = predictor.simulate(None, init_time, duration, 
                                                                                    mujoco_state=init_state, num_ter=num_ter)        
    bb_traj_temp.replan(num_traj_to_compute, load_init_pos_new, load_init_vel_new, load_init_yaw_new)
    bb_traj_temp.switch(num_traj_to_compute)
    #bb_ctrl_temp.setup_hook_up(bb_traj_temp, hook_mass=0.001, payload_mass=load_mass)
    return bb_traj_temp, bb_ctrl_temp


def plot_trajectory(x, y, z, vel, fig: plt.Figure, ax: plt.Axes, add_cbar=False):
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, vel.max())
    lc = Line3DCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(vel)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    if add_cbar:
        cbar = fig.colorbar(line, pad=0.15)
        cbar.set_label("velocity (m/s)")


def plot_car_path(car_traj, block=True):
    if car_traj.pos_tck is None or car_traj.evol_tck is None: # check if data has already been provided
        raise ValueError("No Spline trajectory is specified!")
    
    # evaluate the path between the bounds and plot
    t_eval=np.linspace(0, car_traj.t_end, 100)
    
    s=splev(t_eval, car_traj.evol_tck)
    (x,y)=splev(s, car_traj.pos_tck)
    (vx,vy)=splev(s, car_traj.pos_tck, der=1)
    
    fig, axs = plt.subplots()

    axs.plot(y,x)
    axs.invert_xaxis()
    axs.set_xlabel("y [m]")
    axs.set_ylabel("x [m]")
    axs.grid(True)
    #axs.set_title("UAV reference path")
    axs.axis("equal")

    plt.tight_layout()
    plt.show(block=block)


def plot_car_path_to_axs(car_traj, fig, axs):
    if car_traj.pos_tck is None or car_traj.evol_tck is None: # check if data has already been provided
        raise ValueError("No Spline trajectory is specified!")
    
    # evaluate the path between the bounds and plot
    t_eval=np.linspace(0, car_traj.t_end, 100)
    
    s=splev(t_eval, car_traj.evol_tck)
    (x,y)=splev(s, car_traj.pos_tck)
    (vx,vy)=splev(s, car_traj.pos_tck, der=1)
    
    axs.plot(x,y,zs=0*x)


RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"
BLACK = "0.1 0.1 0.1 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_filename = "scene_base_indoor_hfield.xml"
save_filename = "built_scene.xml"

car_trajectory = CarTrajectory()
# define path points and build the path

# Scenario #1
path_points = np.roll(paperclip(), shift=13, axis=0)
drone_init_pos = np.array([-1, 1, 1.6, 0])  # initial drone position and yaw angle
load_target_pos = np.array([1, 1.13, 0.6])

# Scenario #2
# path_points = dented_paperclip()
# drone_init_pos = np.array([1, 1.13, 1.1, 0])  # initial drone position and yaw angle
# load_target_pos = np.array([1, 1.13, 0.19])
car_trajectory.build_from_points_const_speed(path_points=path_points, path_smoothing=1e-4, path_degree=5,
                                            const_speed=0.5)

car_pos = np.array([car_trajectory.pos_tck[1][0][0], car_trajectory.pos_tck[1][1][0], 0.17])
heading_smoothing_index = 5
car_heading = np.arctan2(car_trajectory.pos_tck[1][1][heading_smoothing_index] - car_trajectory.pos_tck[1][1][0],
                            car_trajectory.pos_tck[1][0][heading_smoothing_index] - car_trajectory.pos_tck[1][0][0])
car_quat = carHeading2quaternion(car_heading)
car_rot = np.array([[np.cos(car_heading), -np.sin(car_heading), 0],
                    [np.sin(car_heading), np.cos(car_heading), 0],
                    [0, 0, 1]])
payload_offset = np.array([-0.5, 0, 0.082])
payload_pos = car_pos + car_rot @ payload_offset
payload_quat = carHeading2quaternion(car_heading+np.deg2rad(0))


# create xml with car, trailer, payload, and bumblebee
scene = SceneXmlGenerator(xml_base_filename)
car_name = scene.add_car(pos=np.array2string(car_pos)[1:-1], quat=car_quat, color=RED,
                            is_virtual=True, has_rod=False, has_trailer=True)
trailer_name = car_name + "_trailer"
load_mass = 0.1
payload_name = scene.add_payload(pos=np.array2string(payload_pos)[1:-1], size="0.05 0.05 0.05", mass=str(load_mass),
                                    quat=payload_quat, color=BLACK, type=PAYLOAD_TYPES.Teardrop)

bb_name = scene.add_drone(np.array2string(drone_init_pos[0:3])[1:-2], "1 0 0 0", RED, DRONE_TYPES.BUMBLEBEE_HOOKED, 2)

terrain0 = scene.add_moving_terrain("0 0 0")
terrain1 = scene.add_moving_terrain("0 0 0")
terrain2 = scene.add_moving_terrain("0 0 0")
terrain3 = scene.add_moving_terrain("0 0 0")
terrain4 = scene.add_moving_terrain("0 0 0")
terrain5 = scene.add_moving_terrain("0 0 0")

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers 
virt_parsers = [parseMovingObjects]

control_step, graphics_step = 0.025, 0.05
xml_filename = os.path.join(xml_path, save_filename)

replanning_timesteps = [1.1, 3.1]
bb_trajectory = HookedDroneNLAdaptiveTrajectory(replanning_timesteps=replanning_timesteps, plot_trajs=False)
bb_trajectory.set_control_step(control_step)
bb_controller = LtvLqrLoadControl([0.605], [1.5e-3, 1.5e-3, 2.6e-3], [0, 0, -9.81])

# Simulate car + trailer to get payload trajectory
predictor = TrailerPredictor(car_trajectory, with_graphics=False, payload_type=PAYLOAD_TYPES.Teardrop,
                            with_terrain=True)
predictor.trailer_top_plate_height = car_pos[2] + 0.067 
init_state = np.hstack((car_pos, np.fromstring(car_quat, sep=" "),
                                    np.zeros(6), 0, 0, 0, 0,
                                    payload_pos, np.fromstring(payload_quat, sep=" "), np.zeros(3)))
load_init_pos, load_init_vel, load_init_yaw, load_yaw_rel = predictor.simulate(init_state, 0, 15, num_ter=0)

# Plan trajectory
bb_trajectory.construct(drone_init_pos[0:3]-np.array([0, 0, 0.4]), drone_init_pos[3], [load_init_pos, load_init_vel],
                        [load_init_yaw, load_yaw_rel], load_target_pos, 0, load_mass, grasp_speed=1.5)

# Compute control gains
bb_controller.setup_hook_up(bb_trajectory, hook_mass=0.02, payload_mass=load_mass)

# recording interval for automatic video capture
#rec_interval=[0,12]
rec_interval = None # no video capture

simulator = ActiveSimulator(xml_filename, rec_interval, control_step, graphics_step, window_size=[3840, 2160],
                            with_graphics=True)
#simulator.cam.azimuth = -45
#simulator.cam.elevation = -25
#simulator.cam.lookat, simulator.cam.distance = [0, 0, .5], 4
#simulator.onBoard_elev_offset = 20

# grabbing the drone and the car
car = simulator.get_MovingObject_by_name_in_xml(car_name)

car_controller = CarLPVController(car.mass, car.inertia, disturbed=False)

car_controllers = [car_controller]

# setting trajectory and controller for car
car.set_trajectory(car_trajectory)
car.set_controllers(car_controllers)

bb = simulator.get_MovingObject_by_name_in_xml(bb_name)

bb.set_controllers([bb_controller])
bb.set_trajectory(bb_trajectory)

payload = simulator.get_MovingObject_by_name_in_xml(payload_name)

num_traj_to_compute = 1

scenario_duration = bb_trajectory.segment_times[-1]
print(scenario_duration)

activate_timestep = 1e3  # just to initialize
replanning_steps = [int(elem / control_step) for elem in replanning_timesteps]

bb_trajectories = [bb_trajectory]
payload_pos = []

while not simulator.should_close(scenario_duration):
    simulator.update()
    # Update terrain
    num_ter = 0
    if simulator.time > 1:
        num_ter = 1
    if simulator.time > 3:
        num_ter = 2
    if simulator.time > 5:
        num_ter = 3
    if simulator.time > 7:
        num_ter = 4
    if simulator.time > 9:
        num_ter = 5
    for i in range(6):
        if i == num_ter:
            simulator.data.mocap_pos[i][-1] = 0
        else:
            simulator.data.mocap_pos[i][-1] = -10
    #if simulator.i == int(5 / control_step):
    #    simulator.pause()
    # Replan trajectory
    if num_traj_to_compute <= len(replanning_timesteps) and simulator.i == replanning_steps[num_traj_to_compute-1]:
        #simulator.pause()
        # This part should be executed in parallel
        # get qpos and qvel of car + trailer
        # car_to_rod = car.data.joint("car_to_rod")  # ball joint
        # rod_yaw = Rotation.from_quat(np.roll(car_to_rod.qpos, -1)).as_euler('xyz')[2]
        # rod_yaw_rate = car_to_rod.qvel[2]
        # front_to_rear = car.data.joint("front_to_rear")  # hinge joint
        # car_trailer_state = np.hstack((car.joint.qpos, car.joint.qvel, rod_yaw, rod_yaw_rate, front_to_rear.qpos, front_to_rear.qvel, 
        #                                payload.sensor_posimeter, payload.sensor_orimeter, payload.sensor_velocimeter))
        car_trailer_state = copy.deepcopy(np.hstack((simulator.data.qpos[:30], simulator.data.qvel[:27])))
        # simulate car + trailer
        #pred_start = time.time()

        bb_traj_temp = copy.deepcopy(bb_trajectories[num_traj_to_compute-1])
        bb_ctrl_temp = copy.deepcopy(bb_controller)
        cur_num_traj = num_traj_to_compute
        thread = thr.Thread(target=replan_parallel, args=(bb_traj_temp, bb_ctrl_temp, car_trailer_state, 
                            simulator.time, scenario_duration-simulator.time, num_ter, cur_num_traj))
        thread.start()


        # bb_traj_temp, bb_ctrl_temp = replan_parallel(bb_traj_temp, bb_ctrl_temp, car_trailer_state, 
        #                     simulator.time, scenario_duration-simulator.time, num_ter, num_traj_to_compute)
        num_traj_to_switch = num_traj_to_compute
        num_traj_to_compute += 1
        activate_timestep = simulator.i + 75

        # output: bb_trajectory.replanner.planners[num_traj] is optimized

    if simulator.i == activate_timestep:
        # join process
        # print("Joining thread")
        thread.join()
        bb_trajectories.append(bb_traj_temp)
        bb_controller = bb_ctrl_temp
        bb.set_trajectory(bb_trajectories[-1])
        bb.set_controllers([bb_controller])
        print("Ready to join process")
    
    payload_pos.append(payload.sensor_posimeter.tolist())

simulator.close()


fig = plt.figure()
ax = plt.axes(projection="3d")
t_plot = [np.arange(0.001, 3.1, 0.01), np.arange(3.11, 5.1, 0.01), np.arange(5.11, scenario_duration-0.1, 0.01)]
x = np.array([])
y = np.array([])
z = np.array([])
vel = np.array([])
al = np.array([])
bet = np.array([])
dal = np.array([])
dbet = np.array([])
for i, traj in enumerate(bb_trajectories):
    x = np.append(x, traj.states(t_plot[i])[:, 0])
    y = np.append(y, traj.states(t_plot[i])[:, 1])
    z = np.append(z, traj.states(t_plot[i])[:, 2])
    al = np.append(al, traj.states(t_plot[i])[:, 12])
    bet = np.append(bet, traj.states(t_plot[i])[:, 13])
    dal = np.append(dal, traj.states(t_plot[i])[:, 14])
    dbet = np.append(dbet, traj.states(t_plot[i])[:, 15])
    v = traj.states(t_plot[i])[:, 3:6]
    vel = np.append(vel, np.linalg.norm(v, axis=1))

#r = np.vstack(np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(z))
q1 = -np.cos(al)*np.sin(bet)
q2 = np.sin(al)
q3 = -np.cos(al)*np.cos(bet)
L = 0.4
xL = x + L*q1
yL = y + L*q2
zL = z + L*q3

plot_trajectory(xL, yL, zL, vel, fig, ax, add_cbar=True)
ax.set_xlim(min(xL)-0.3, max(xL)+0.3)
ax.set_ylim(min(yL)-0.3, max(yL)+0.3)
ax.set_zlim(-0.1, max(zL)+0.3)
ax.set_box_aspect((np.ptp(xL), np.ptp(yL), np.ptp(zL)))
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
#car_trajectory.plot_car_path_to_axs(fig, ax)
payload_pos = np.asarray(payload_pos)
ax.plot(payload_pos[:, 0], payload_pos[:, 1], zs=payload_pos[:, 2])
plt.show()
