import matplotlib.pyplot as plt
import motioncapture
import numpy as np
from scipy.spatial.transform import Rotation
from quadcopter_hook_twodof.test.generate_nl_trajs_hookup import generate_nl_trajs_hookup
from skyc_utils.skyc_maker import Trajectory, write_skyc, XYZYaw, TrajectoryType
from skyc_utils.skyc_inspector import inspect
import pickle
from functools import partial
from aiml_virtual.trajectory.trailer_predictor import TrailerPredictor
from aiml_virtual.trajectory.car_path_point_generator import paperclip, dented_paperclip
from aiml_virtual.trajectory import CarTrajectory
from aiml_virtual.object.payload import PAYLOAD_TYPES
import sys, os

# The value of these two parameters have to match with the same parameters in the firmware.
LQR_N = 240  # number of LQR gain matrices along the trajectory
INT16_LQR_SCALING = 32766


def get_mocap_data(drone_name: str, payload_name: str, car_name: str, trailer_name: str):
    mc = motioncapture.MotionCaptureOptitrack("192.168.2.141")
    num_average = 240

    # initialize arrays
    drone_pos = np.nan * np.ones((3, num_average))
    drone_yaw = np.nan * np.ones(num_average)
    load_pos = np.nan * np.ones((3, num_average))
    load_yaw = np.nan * np.ones(num_average)
    car_pos = np.nan * np.ones((3, num_average))
    car_yaw = np.nan * np.ones(num_average)
    trailer_pos = np.nan * np.ones((3, num_average))
    trailer_yaw = np.nan * np.ones(num_average)

    for i in range(num_average):
        mc.waitForNextFrame()

        for name, obj in mc.rigidBodies.items():
            if name == drone_name:
                # have to put rotation.w to the front because the order is different
                quat = np.array([obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w])
                drone_yaw[i] = Rotation.from_quat(quat).as_euler('xyz')[2]
                drone_pos[:, i] = obj.position
            elif name == payload_name:
                quat = np.array([obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w])
                load_yaw[i] = Rotation.from_quat(quat).as_euler('xyz')[2]
                load_pos[:, i] = obj.position
            elif name == car_name:
                quat = np.array([obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w])
                car_yaw[i] = Rotation.from_quat(quat).as_euler('xyz')[2]
                car_pos[:, i] = obj.position
            elif name == trailer_name:
                quat = np.array([obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w])
                trailer_yaw[i] = Rotation.from_quat(quat).as_euler('xyz')[2]
                trailer_pos[:, i] = obj.position

    drone_pos_av = np.average(drone_pos, 1)
    drone_pos_std = np.std(drone_pos, 1)
    # print(f'Drone pos av: {drone_pos_av}')
    # print(f'Drone pos std: {drone_pos_std}')

    drone_yaw_av = np.average(drone_yaw)
    drone_yaw_std = np.std(drone_yaw)
    # print(f'Drone yaw av: {drone_yaw_av}')
    # print(f'Drone yaw std: {drone_yaw_std}')

    load_pos_av = np.average(load_pos, 1)
    load_pos_std = np.std(load_pos, 1)
    # print(f'Load pos av: {load_pos_av}')
    # print(f'Load pos std: {load_pos_std}')

    load_yaw_av = np.average(load_yaw, 0)
    load_yaw_std = np.std(load_yaw, 0)

    car_pos_av = np.average(car_pos, 1)
    car_yaw_av = np.average(car_yaw, 0)
    trailer_pos_av = np.average(trailer_pos, 1)
    trailer_yaw_av = np.average(trailer_yaw, 0)

    # with open("pickle/car_trailer_init.pickle", "wb") as file:
    #     pickle.dump([car_pos_av, car_yaw_av, trailer_pos_av, trailer_yaw_av, load_pos_av, load_yaw_av], file)

    if np.isnan(np.sum(car_pos_av + trailer_pos_av + load_pos_av)):  # no car or no trailer in optitrack, use pickle data
        print("Car or trailer data is not coming from Optitrack, using pregenerated pickle")
        with open("pickle/car_trailer_init.pickle", "rb") as file:
            data = pickle.load(file)
        car_pos_av = data[0]
        car_yaw_av = data[1]
        trailer_pos_av = data[2]
        trailer_yaw_av = data[3]
        load_pos_av = data[4]
        load_yaw_av = data[5]
    # drone_pos_av = np.zeros(3)
    # drone_yaw_av = 0

    # print(f'Load yaw av: {load_yaw_av}')
    # print(f'Load yaw std: {load_yaw_std}')
    # assert not np.isnan(np.sum(np.hstack((drone_pos_av, drone_yaw_av, load_pos_av[:, 0], load_yaw_av)))), \
    #     'Optitrack rigid body data is not coming'
    # assert np.max(np.hstack((drone_pos_std, drone_yaw_std, load_pos_std[:, 0], load_yaw_std))) < 1e-3, \
    #     'Standard deviation of Optitrack data is too high'
    return (drone_pos_av - np.array((0, 0, 0.1)), drone_yaw_av, load_pos_av, load_yaw_av,
            (car_pos_av, car_yaw_av, trailer_pos_av, trailer_yaw_av))


def predict_car_trailer_motion(load_init_pos, load_init_yaw, car_trailer_config, trajectory_type):
    predicted_obj = 'payload'  # either car or payload, mainly for debug purposes

    if trajectory_type == 'paperclip':
        path_points = np.roll(paperclip(), shift=13, axis=0)
        # path_points = np.hstack((np.atleast_2d(path_points[:, 1]).T, -np.atleast_2d(path_points[:, 0]).T))
    elif trajectory_type == 'dented_paperclip':
        path_points = dented_paperclip()
    else:
        raise RuntimeError('Trajectory type not implemented')
    car_trajectory = CarTrajectory()
    car_trajectory.build_from_points_const_speed(path_points=path_points, path_smoothing=1e-4, path_degree=5,
                                                 const_speed=0.6)
    predictor = TrailerPredictor(car_trajectory, payload_type=PAYLOAD_TYPES.Teardrop, with_graphics=False)
    car_trailer_states = compute_car_trailer_states(load_init_pos, load_init_yaw, car_trailer_config)
    load_pos, load_vel, load_yaw, load_yaw_rel = predictor.simulate(car_trailer_states, 0, 15, predicted_obj=predicted_obj)
    print(f"Load position at t=7s: {repr(load_pos(7, 0))}")
    print(load_yaw_rel)
    return load_pos, load_vel, load_yaw, load_yaw_rel


def compute_car_trailer_states(load_init_pos, load_init_yaw, car_trailer_config):
    car_pos = car_trailer_config[0]
    car_pos[2] = 0.052  # has to stay on the ground
    car_yaw = car_trailer_config[1]
    trailer_pos = car_trailer_config[2]
    trailer_yaw = car_trailer_config[3]

    car_euler = np.array([0, 0, car_yaw])
    car_quat = np.roll(Rotation.from_euler('xyz', car_euler).as_quat(), shift=1)
    car_rotmat = Rotation.from_euler('xyz', car_euler).as_matrix()
    car_omega = np.zeros(3)  # TODO: we will need it for dynamic replanning
    car_qpos = np.hstack((car_pos, car_quat))
    car_vel = np.zeros(3)  # TODO
    car_qvel = np.hstack((car_vel, car_omega))

    trailer_euler = np.array([0, 0, trailer_yaw])
    trailer_rotmat = Rotation.from_euler('xyz', trailer_euler).as_matrix()
    car_to_rod_len = 0.25
    rod_len = 0.18
    rod_to_trailer_len = 0.21
    rod_front_pos = car_qpos[:3] + car_rotmat @ np.array([-car_to_rod_len, 0, 0])
    rod_rear_pos = trailer_pos + trailer_rotmat @ np.array([rod_to_trailer_len, 0, 0])
    rod_vec = rod_front_pos - rod_rear_pos
    print(f"rod vec: {rod_vec}")
    rod_euler = np.array([0, 0, np.arctan2(rod_vec[1], rod_vec[0])])
    rod_rotmat = Rotation.from_euler('xyz', rod_euler).as_matrix()
    rod_rotmat_rel = car_rotmat.T @ rod_rotmat
    rod_yaw = Rotation.from_matrix(rod_rotmat_rel).as_euler('xyz')[2]
    print(f"rod yaw: {rod_yaw}")
    rod_yaw_rate = 0 * rod_yaw
    trailer_rotmat_rel = rod_rotmat.T @ trailer_rotmat
    trailer_yaw_rel = Rotation.from_matrix(trailer_rotmat_rel).as_euler('xyz')[2]
    print(f"trailer yaw rel: {trailer_yaw_rel}")

    load_euler = np.array([0, 0, load_init_yaw])
    load_quat = np.roll(Rotation.from_euler('xyz', load_euler).as_quat(), shift=1)
    # load_rotmat = Rotation.from_euler('xyz', load_euler).as_matrix()
    # load_pos = load_pos + load_rotmat @ np.array([0.05, 0, 0.0])
    load_vel = np.zeros(3)  # TODO

    car_trailer_states = np.hstack((car_qpos, car_qvel, rod_yaw, rod_yaw_rate, trailer_yaw_rel, 0 * trailer_yaw_rel,
                                    load_init_pos, load_quat, load_vel))
    return car_trailer_states


def plan_and_construct_pickle(drone_pos, drone_yaw, load_pos, load_vel, load_yaw, load_yaw_rel, target_load_pos,
                              target_load_yaw, load_mass, hook_mass=0.02, control_step=0.05, takeoff_height=1.1):
    # Input: payload initial position, yaw + drone initial position, yaw + payload final position

    traj_path = 'pickle/traj_moving_hookup.pickle'
    lqr_param_path = 'pickle/lqr_params_bb.pickle'

    t_scale = 5
    t0 = 0

    grasp_time, detach_time = generate_nl_trajs_hookup(drone_pos, drone_yaw, partial(load_pos, t0=t0),
                                                       partial(load_vel, t0=t0), partial(load_yaw, t0=t0), load_yaw_rel,
                                                       target_load_pos, target_load_yaw, grasp_speed=1.3,
                                                       traj_path=traj_path, lqr_param_path=lqr_param_path,
                                                       drone_mass=0.605, inertia=[1.5e-3, 1.45e-3, 2.66e-3],
                                                       hook_mass=hook_mass, payload_mass=load_mass,
                                                       control_step=control_step, takeoff_height=takeoff_height,
                                                       rod_length=0.4, t_scale=t_scale, poly_deg=7, optim_order=6)

    return grasp_time, detach_time


def construct_skyc(load_mass, hook_mass):
    traj = Trajectory(TrajectoryType.POLY4D, degree=7)
    pickle_filename = "pickle/traj_moving_hookup.pickle"
    with open(pickle_filename, "rb") as file:
        data = pickle.load(file)
    start = XYZYaw(*[float(ppoly(0)) for ppoly in data])
    traj.set_start(start)
    traj.add_ppoly(data)

    lqr_filename = "pickle/lqr_params_bb.pickle"
    with open(lqr_filename, "rb") as file:
        data = pickle.load(file)

    K_sample, func_params = data
    traj_duration = func_params[5] + 3
    grasp_time = func_params[3]
    detach_time = func_params[4]

    def K_fun(t_):
        K_lst_sections = func_params[0:3]

        K_lst = K_lst_sections[0] if t_ <= grasp_time else K_lst_sections[1] if t_ <= detach_time else \
            K_lst_sections[2]
        n = 16
        m = 4
        return np.asarray([[K_lst[i * n + j](t_) for j in range(n)] for i in range(m)]).flatten().tolist()

    # Calculate all K matrices along the trajectory, together with the upper and lower bounds
    t_eval = np.linspace(0, traj_duration, LQR_N)
    K = []
    K_min = 64 * [np.inf]
    K_max = 64 * [-np.inf]
    K_plot = []
    for t in t_eval:
        K_cur = K_fun(t)
        K_plot += [np.reshape(np.array(K_cur), (4, 16))]
        K_min = [K_cur[i] if K_cur[i] < K_min[i] else K_min[i] for i in range(64)]
        K_max = [K_cur[i] if K_cur[i] > K_max[i] else K_max[i] for i in range(64)]
        K += K_cur

    # K_to_plot = [K_[i] for i in [17, 20, 22] for K_ in K]
    # plt.figure()
    # plt.plot(t_eval, [K_[1, 12] for K_ in K_plot])
    # plt.plot(t_eval, [K_[1, 13] for K_ in K_plot])
    # plt.show()

    # normalize parameter set
    K_normed = [elem for elem in K]  # copy elementwise
    for i in range(LQR_N):
        for j in range(64):
            try:
                K_normed[i * 64 + j] = round(((K_normed[i * 64 + j] - (K_min[j] + K_max[j]) / 2) /
                                              ((K_max[j] - K_min[j]) / 2)) * INT16_LQR_SCALING)
            except ZeroDivisionError:
                K_normed[i * 64 + j] = 0

    traj.add_lqr_params(K_normed, K_min + K_max, LQR_N)

    print(traj_duration)

    traj.add_parameter(-12, "usd.logging", 0)
    traj.add_parameter(-10, "pptraj.traj_mode_drone", 0)
    traj.add_parameter(-9, "Lqr2.max_delay_time_ms", 250)
    traj.add_parameter(-8, "stabilizer.controller", 7)
    traj.add_parameter(-7, "Lqr2.max_delay", 500)
    traj.add_parameter(-6, "Lqr2.rod_length_safety", 0.58)
    traj.add_parameter(-5, "Lqr2.duration", int(traj_duration * 1000))
    # traj.add_parameter(-2, "usd.logging", 1)
    traj.add_parameter(grasp_time+1, "pptraj.payload_mass", hook_mass + load_mass)
    traj.add_parameter(detach_time, "pptraj.payload_mass", hook_mass)
    # traj.add_parameter(detach_time + 10, "usd.logging", 0)

    write_skyc([traj], 'skyc/hookup')

    sys.stdout = open(os.devnull, 'w')  # Unsafe way to suppress warnings (xD)
    inspect('skyc/hookup.skyc')
    sys.stdout = sys.__stdout__


def plan_trajectory(drone_name, payload_name, car_name, trailer_name,
                    target_load_pos, target_load_yaw, payload_mass, control_step):
    drone_pos, drone_yaw, load_init_pos, load_init_yaw, car_trailer_config = get_mocap_data(drone_name, payload_name,
                                                                                            car_name, trailer_name)
    load_pos, load_vel, load_yaw, load_yaw_rel = predict_car_trailer_motion(load_init_pos, load_init_yaw,
                                                                            car_trailer_config,
                                                                            trajectory_type='paperclip')
    grasp_times, detach_times = plan_and_construct_pickle(drone_pos, drone_yaw, load_pos, load_vel, load_yaw,
                                                          load_yaw_rel, target_load_pos, target_load_yaw, payload_mass,
                                                          control_step=control_step, takeoff_height=1.1)
    construct_skyc(payload_mass, hook_mass=0.02)


if __name__ == '__main__':
    plan_trajectory('bb3', 'payload2', 'JoeBush1', 'trailer',
                    np.array([1.2, 0.6, 0.25]), np.deg2rad(90),
                    payload_mass=0.076, control_step=0.05)
