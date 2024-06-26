import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
from aiml_virtual.object.payload import PAYLOAD_TYPES
from aiml_virtual.trajectory.trailer_predictor import TrailerPredictor
from aiml_virtual.trajectory.car_path_point_generator import paperclip, dented_paperclip
from aiml_virtual.trajectory import CarTrajectory

def load_meas_data(filename):
    # Open the CSV file
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data_lst = list(reader)

    # Extract headers and data
    headers = data_lst[0]
    data_lst = data_lst[1:]

    # Convert data to float
    data_lst = [[float(entry) for entry in row] for row in data_lst]

    # Transpose data for easier plotting
    data_transposed = list(zip(*data_lst))

    # Create dictionary
    data = {}
    for i, header in enumerate(headers):
        data[header] = np.expand_dims(np.array(data_transposed[i]), axis=1)

    # Compute states of car+trailer system
    N = len(data_transposed[0])
    car_euler = np.hstack((np.zeros((N, 2)), data['car_heading']))
    car_quat = np.roll(Rotation.from_euler('xyz', car_euler).as_quat(), shift=1, axis=1)
    car_rotmat = np.asarray([Rotation.from_euler('xyz', rpy).as_matrix() for rpy in car_euler])
    car_omega = np.hstack((np.zeros((N, 2)), data['car_ang_vel']))
    car_qpos = np.hstack((data['car_x'], data['car_y'], 0.052 * np.ones((N, 1)), car_quat))
    car_qvel = np.hstack((data['car_vx'], data['car_vy'], np.zeros((N, 1)), car_omega))

    trailer_pos = np.hstack((data['trailer_x'], data['trailer_y'], np.zeros((N, 1))))
    trailer_euler = np.hstack((np.zeros((N, 2)), data['trailer_heading']))
    trailer_rotmat = np.asarray([Rotation.from_euler('xyz', rpy).as_matrix() for rpy in trailer_euler])
    car_to_rod_len = 0.25
    rod_len = 0.18
    rod_to_trailer_len = 0.21
    rod_front_pos = car_qpos[:, :3] + car_rotmat @ np.array([-car_to_rod_len, 0, 0])
    rod_rear_pos = trailer_pos + trailer_rotmat @ np.array([rod_to_trailer_len, 0, 0])
    rod_vec = rod_front_pos - rod_rear_pos
    rod_euler = np.hstack((np.zeros((N, 2)), np.expand_dims(np.arctan2(rod_vec[:, 1], rod_vec[:, 0]), 1)))
    rod_rotmat = np.asarray([Rotation.from_euler('xyz', rpy).as_matrix() for rpy in rod_euler])
    rod_rotmat_rel = [car_rot.T @ rod_rot for car_rot, rod_rot in zip(car_rotmat, rod_rotmat)]
    rod_yaw = np.expand_dims(Rotation.from_matrix(rod_rotmat_rel).as_euler('xyz')[:, 2], 1)
    rod_yaw_rate = 0 * rod_yaw
    trailer_rotmat_rel = [rod_rot.T @ trailer_rot for rod_rot, trailer_rot in zip(rod_rotmat, trailer_rotmat)]
    trailer_yaw_rel = np.expand_dims(Rotation.from_matrix(trailer_rotmat_rel).as_euler('xyz')[:, 2], 1)

    payload_pos = np.hstack((data['payload_x'], data['payload_y'], 0.125 * np.ones((N, 1))))
    payload_euler = np.hstack((np.zeros((N, 2)), data['payload_heading']))
    payload_quat = np.roll(Rotation.from_euler('xyz', payload_euler).as_quat(), shift=1, axis=1)
    payload_rotmat = np.asarray([Rotation.from_euler('xyz', rpy).as_matrix() for rpy in payload_euler])
    #payload_pos = payload_pos + payload_rotmat @ np.array([0.05, 0, 0.0])
    payload_vel = np.hstack((data['payload_vx'], data['payload_vy'], np.zeros((N, 1))))

    car_trailer_states = np.hstack((car_qpos, car_qvel, rod_yaw, rod_yaw_rate, trailer_yaw_rel, 0*trailer_yaw_rel,
                                   payload_pos, payload_quat, payload_vel))
    car_inputs = np.hstack((data.get('d'), data.get('delta')))
    timestamp = data["time_stamp_sec"]
    timestamp -= timestamp[0]
        
    return car_trailer_states, car_inputs, timestamp, data

if __name__ == "__main__":
    trajectory_type = 'paperclip'  # for now either 'paperclip' or 'dented_paperclip'
    predicted_obj = 'payload' # either car or payload
    #csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csv', 'JoeBush1_04_24_2024_11_48_30.csv')
    csv_path = "/home/aimotion-laptop/AIMotionLab-Virtual/csv/log28.csv"
    meas_state, _, meas_time, meas_data = load_meas_data(csv_path)

    if trajectory_type == 'paperclip':
        path_points = np.roll(paperclip(), shift=13, axis=0)
    elif trajectory_type == 'dented_paperclip':
        path_points = dented_paperclip()
    else:
        print('Trajectory type not implemented')
    car_trajectory = CarTrajectory()
    car_trajectory.build_from_points_const_speed(path_points=path_points, path_smoothing=1e-4, path_degree=5,
                                                        const_speed=0.6)
    predictor = TrailerPredictor(car_trajectory, payload_type=PAYLOAD_TYPES.Teardrop, with_graphics=False)
    simu_start_idx = 0
    predictor.car.controllers[0].C_m2 *= 1.117
    predictor.car.controllers[0].C_m3 *= 0.6466
    load_pos, _, _, _ = predictor.simulate(meas_state[simu_start_idx, :], meas_time[simu_start_idx, 0], 
                                           meas_time[-1]-meas_time[simu_start_idx], predicted_obj)

    # plot
    meas_time = meas_time[simu_start_idx:, :] - meas_time[simu_start_idx, :]
    if predicted_obj == 'car':
        meas_to_plot = meas_state[simu_start_idx:, 0:2]
    else:
        meas_to_plot = meas_state[simu_start_idx:, 17:19]
    plt.figure()
    plt.plot(meas_time, meas_to_plot)
    plt.plot(meas_time, np.asarray(load_pos(meas_time, 0))[:2, :, 0].T) # 3 lista
    plt.grid('on')

    plt.figure()
    plt.plot(meas_time, meas_to_plot-np.asarray(load_pos(meas_time, 0))[:2, :, 0].T)
    plt.plot(meas_time, np.linalg.norm(meas_to_plot-np.asarray(load_pos(meas_time, 0))[:2, :, 0].T, axis=1))
    plt.grid('on')

    plt.figure()
    plt.plot(meas_to_plot[:, 0], meas_to_plot[:, 1])
    plt.plot(np.asarray(load_pos(meas_time, 0))[0, :, 0], np.asarray(load_pos(meas_time, 0))[1, :, 0])
    x,y = car_trajectory.get_traj()
    plt.plot(x,y)
    plt.legend(['meas', 'pred', 'ref'])
    plt.grid('on')
    plt.axis('equal')
    plt.show()