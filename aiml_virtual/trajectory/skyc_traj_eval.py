import os
import shutil
import json

from scipy import interpolate
import numpy as np
import bisect


import shutil
import zipfile

def unpack_skyc_file(skyc_filename: str) -> str:
    '''Function that takes a skyc file and extracts its contents neatly into a folder, as if we used winrar. Returns
    the name of this folder.'''
    folder_name = os.path.splitext(skyc_filename)[0]  # first element of the list is the file name, second is ".skyc"
    if os.path.exists(folder_name):  # if there is a leftover folder from a previous run, delete it!
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)  # make a new folder, named after the skyc file
    with zipfile.ZipFile(skyc_filename, 'r') as zip_ref:  # then extract everything into it
        zip_ref.extractall(folder_name)
    return folder_name

def cleanup(skyc_filename: str) -> None:
    '''Function that deletes the folder from which we extracted the data.'''
    folder_name = os.path.splitext(skyc_filename)[0]  # first element of the list is the file name, second is ".skyc"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

def get_traj_data(skyc_file: str):
    '''Function that extracts the contents of the trajectory.json files in the provided skyc file. Returns the
    dictionary containing this data.'''
    folder_name = unpack_skyc_file(skyc_file)  # unpack the skyc file (it's like a zip)
    drones_folder = os.path.join(folder_name, "drones")  # within it, there should be a 'drones' folder for trajectories
    traj_data = []
    for root, dirs, files in os.walk(drones_folder):
        # iterating over the files and folders in the drones folder, we are looking for trajectory files
        if 'trajectory.json' in files:
            with open(os.path.join(root, 'trajectory.json'), 'r') as json_file:
                data = json.load(json_file)
                points = data.get("points")
                assert points is not None
                data["has_yaw"] = True if len(points[0][1]) == 4 else False  # determine if there is a yaw trajectory
                traj_data.append(data)
                traj_type = data.get("type", "COMPRESSED").upper()
                # compressed trajectories can only be of degree 1, 3 and 7 as per the bitcraze documentation
                # if a trajectory is not compressed, it is poly4d, which (for now) can only have degrees up to 5
                ctrl_point_num = [0, 2, 6] if traj_type == "COMPRESSED" else [0, 1, 2, 3, 4]
                for point in points:
                    assert len(point[2]) in ctrl_point_num  # throw an error if the degree is not matching the type!
    cleanup(skyc_file)
    return traj_data


def proc_json_trajectory(data: dict) -> dict:
    points = data.get("points")
    assert points is not None
    data["has_yaw"] = True if len(points[0][1]) == 4 else False  # determine if there is a yaw trajectory
    traj_type = data.get("type", "COMPRESSED").upper()
    # compressed trajectories can only be of degree 1, 3 and 7 as per the bitcraze documentation
    # if a trajectory is not compressed, it is poly4d, which (for now) can only have degrees up to 5
    ctrl_point_num = [0, 2, 6] if traj_type == "COMPRESSED" else [0, 1, 2, 3, 4]
    for point in points:
        assert len(point[2]) in ctrl_point_num  # throw an error if the degree is not matching the type!
    
    return data


def get_traj_data_from_json(directory: str):
    
    with open(os.path.join(directory, 'trajectory.json'), 'r') as json_file:
        data = json.load(json_file)
                    
    return proc_json_trajectory(data)

def evaluate_segment(points, start_time: float, end_time: float,
                    eval_time, has_yaw: bool):
    # TODO: find a more efficient method
    '''Function that takes the control points of a bezier curve, creates an interpolate.BPoly object for each
    dimension of the curve, evaluates them at the given time and returns a tuple with the time, x, y, z and yaw.'''
    # The bernstein coefficients are simply the coordinates of the control points for each dimension.
    x_coeffs = [point[0] for point in points]
    y_coeffs = [point[1] for point in points]
    z_coeffs = [point[2] for point in points]
    x_BPoly = interpolate.BPoly(np.array(x_coeffs).reshape(len(x_coeffs), 1), np.array([start_time, end_time]))
    y_BPoly = interpolate.BPoly(np.array(y_coeffs).reshape(len(y_coeffs), 1), np.array([start_time, end_time]))
    z_BPoly = interpolate.BPoly(np.array(z_coeffs).reshape(len(z_coeffs), 1), np.array([start_time, end_time]))
    X = x_BPoly(eval_time)
    Y = y_BPoly(eval_time)
    Z = z_BPoly(eval_time)
    Vx = x_BPoly(eval_time, nu=1)
    Vy = y_BPoly(eval_time, nu=1)
    Vz = z_BPoly(eval_time, nu=1)
    # Make sure that the trajectory doesn't take the drone outside the limits of the optitrack system!
    #assert LIMITS[0][0] < X < LIMITS[0][1] and LIMITS[1][0] < Y < LIMITS[1][1] and LIMITS[2][0] < Z < LIMITS[1][1]
    retval = [float(X), float(Y), float(Z)], [float(Vx), float(Vy), float(Vz)]
    if has_yaw:
        yaw_coeffs = [point[3] for point in points]
        yaw_BPoly = interpolate.BPoly(np.array(yaw_coeffs).reshape(len(yaw_coeffs), 1), np.array([start_time, end_time]))
        retval.append(float(yaw_BPoly(eval_time)))
    return tuple(retval)


def evaluate_trajectory(trajectory, time):
    '''Function that looks at which bezier curve each timestamp falls into, then evaluates the curve at that
    timestamp, and returns the result for each timestamp.'''

    segments = trajectory.get("points")
    # check which segment the current timestamp falls into
    i = bisect.bisect_left([segment[0] for segment in segments], time)
    if i == 0:
        return segments[0][1], [0, 0, 0]
    elif i == len(segments):
        return segments[-1][1], [0, 0, 0]
    else:
        prev_segment = segments[i-1]
        start_point = prev_segment[1]
        start_time = prev_segment[0]
        segment = segments[i]
        end_point = segment[1]
        end_time = segment[0]
        ctrl_points = segment[2]
        # points will contain all points of the bezier curve, including the start and end, unlike in trajectory.json
        points = [start_point, *ctrl_points, end_point] if ctrl_points else [start_point, end_point]
        return evaluate_segment(points, start_time, end_time, time, trajectory.get("has_yaw", False))
            
    return eval

