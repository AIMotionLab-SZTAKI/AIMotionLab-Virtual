from typing import Union, Callable
import math
from aiml_virtual.object.payload import Payload, BoxPayload, TeardropPayload
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import platform
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time


def carHeading2quaternion(phi: float)-> str:
    """Converts the car heading angle (rotation around Z-axis) to quaternions that can be handled by the Mujoco Simulator

    Args:
        phi (int): Heading angle of the car (measured from the X-axis, in radians)

    Returns:
        str: Sting of the 4 quaternions
    """
    return str(math.cos( phi / 2 ))+" 0 0 "+ str(math.sin( phi / 2 ))


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def sync(i, start_time, pause_time, timestep):
    """Syncs the stepped simulation with the wall-clock.
    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.
    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.
    """
    #if timestep > .04 or i % (int(1 / (24 * timestep))) == 0:
    elapsed = time.time() - (start_time + pause_time)
    sim_time = i * timestep
    #print(sim_time - elapsed)
    if elapsed < sim_time:
        delay = sim_time - elapsed
        #print(delay)
        time.sleep(delay)


class FpsLimiter:

    def __init__(self, target_fps):
        self.fps = target_fps
        self.timestep = 1.0 / target_fps


    def begin_frame(self):
        self.t1 = time.time()

    def end_frame(self):
        frame_time = time.time() - self.t1

        if self.timestep > frame_time:
            time.sleep(self.timestep - frame_time)


def plot_payload_and_airflow_volume(payload, airflow_sampler, payload_color: str = "tab:blue"):
    if isinstance(payload, BoxPayload):
        plot_payload_and_airflow_volume_box_payload(payload, airflow_sampler, payload_color)
    elif isinstance(payload, TeardropPayload):
        plot_payload_and_airflow_volume_teardrop_payload(payload, airflow_sampler, payload_color)
    

def plot_payload_and_airflow_volume_box_payload(payload, airflow_sampler, payload_color: str = "tab:blue"):
    p, pos, n, a = payload.get_top_rectangle_data()

    payload_offset = airflow_sampler.get_payload_offset_z_meter()

    p[:, 2] += payload_offset

    col = payload_color
    alp = 0.35

    fig = plt.figure(payload.name_in_xml)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=col, alpha=alp)

    p_n, p_p, pown_n, pown_p, n_n, n_p, area_xz = payload.get_side_xz_rectangle_data()
    p_n[:, 2] += payload_offset
    p_p[:, 2] += payload_offset
    ax.scatter(p_n[:, 0], p_n[:, 1], p_n[:, 2], color=col, alpha=alp)
    ax.scatter(p_p[:, 0], p_p[:, 1], p_p[:, 2], color=col, alpha=alp)

    p_n, p_p, pown_n, pown_p, n_n, n_p, area_xz = payload.get_side_yz_rectangle_data()
    p_n[:, 2] += payload_offset
    p_p[:, 2] += payload_offset
    ax.scatter(p_n[:, 0], p_n[:, 1], p_n[:, 2], color=col, alpha=alp)
    ax.scatter(p_p[:, 0], p_p[:, 1], p_p[:, 2], color=col, alpha=alp)


    faces = []
    faces.append(np.zeros([5,3]))
    faces.append(np.zeros([5,3]))
    faces.append(np.zeros([5,3]))
    faces.append(np.zeros([5,3]))
    faces.append(np.zeros([5,3]))
    faces.append(np.zeros([5,3]))


    vs = airflow_sampler.get_transformed_vertices()


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

def plot_payload_and_airflow_volume_teardrop_payload(payload, airflow_sampler, payload_color: str = "tab:blue"):
    p, pos, n, a = payload.get_data()
    payload_offset = airflow_sampler.get_payload_offset_z_meter()
    p[:, 2] += payload_offset

    col = payload_color
    alp = 0.35

    fig = plt.figure(payload.name_in_xml)
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=col, alpha=alp)

    faces = []
    for _ in range(6):
        faces.append(np.zeros([5, 3]))

    vs = airflow_sampler.get_transformed_vertices()

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

    zoom_enabled = True
    if zoom_enabled:
        all_points = np.vstack([p, vs])
        x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
        y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
        z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]

        zoom_in_factor = 1.1

        ax.set_xlim([x_limits[0] - (x_limits[1] - x_limits[0]) * (1 - zoom_in_factor),
                    x_limits[1] + (x_limits[1] - x_limits[0]) * (1 - zoom_in_factor)])
        ax.set_ylim([y_limits[0] - (y_limits[1] - y_limits[0]) * (1 - zoom_in_factor),
                    y_limits[1] + (y_limits[1] - y_limits[0]) * (1 - zoom_in_factor)])
        ax.set_zlim([z_limits[0] - (z_limits[1] - z_limits[0]) * (1 - zoom_in_factor),
                    z_limits[1] + (z_limits[1] - z_limits[0]) * (1 - zoom_in_factor)])

    plt.show()