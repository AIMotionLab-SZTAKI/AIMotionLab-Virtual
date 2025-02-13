from __future__ import annotations

import copy
import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import interpolate
import pickle
import os

from mpl_toolkits.mplot3d.art3d import Line3DCollection


#=======================================================================================================================
# GENERAL
def save(data: any, file: str) -> None:
    pickle_out = open(file, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load(file: str) -> any:
    pickle_in = open(file, "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data


def load_scenario_and_drones(file_name: str) -> {object, list[object]}:
    data = load(f"Saves/Scenarios/{file_name}")
    scenario = data["Scenario"]
    drones = data["Drones"]
    return scenario, drones


def check_routes_save_file(file_name):
    while True:
        file_path = f"Saves/Routes/{file_name}"
        if os.path.exists(file_path):
            answer = input("File for storing the routes is already exists.\n"
                           "For deleting the current file pess: d\n"
                           "Otherwise enter a new file name (e.g., new_routes)")
            if answer == 'd' or answer == 'D':
                os.remove(file_path)
                return file_name
            file_name = answer
        else:
            return file_name


def save_drones_routes(drones, routes, file_name):
    for drone in drones:
        if not drone.returned_home:
            return

    file_path = f"Saves/Routes/{file_name}"
    if not os.path.exists(file_path):
        save(data=routes, file=file_path)
        print("Routes saved")


def c_print(message: str, color: str) -> None:
    colors = {'black': '\033[30m',
              'red': '\033[31m',
              'green': '\033[32m',
              'orange': '\033[33m',
              'blue': '\033[34m',
              'purple': '\033[35m',
              'cyan': '\033[36m',
              'lightgrey': '\033[37m',
              'darkgrey': '\033[90m',
              'lightred': '\033[91m',
              'lightgreen': '\033[92m',
              'yellow': '\033[93m',
              'lightblue': '\033[94m',
              'pink': '\033[95m',
              'lightcyan': '\033[96m'}

    print(f"{colors[color]}{message}\033[0m ")


def set_rand_seed(seed: None | int = None):
    if seed is None:
        seed = np.random.randint(low=0, high=2000000000, size=1)[0]
    np.random.seed(seed)
    print("Random seed:", seed)


def read_optitrack_take():
    file_path = 'C:/Users/Mate/PycharmProjects/Graph_based_trajectory_design/Saves/Take.csv'  # Adjust the path to the file
    data = pd.read_csv(file_path)
    return data


#=======================================================================================================================
# OBSTACLES
def handle_dynamic_obstacles(scenario, t: float, drones: list) -> None:

    colliding_drone_IDs = []
    for dynamic_obs in scenario.dynamic_obstacles:
        colliding_drone_IDs.extend(dynamic_obs.update(scenario=scenario, t=t, drones=drones)) # TODO: make the order dependent on collision time
    for ID in list(set(colliding_drone_IDs)):
        drone = next((drone for drone in drones if drone.ID == ID), None)
        drone.avoid_collision(scenario=scenario, t=t, drones=drones)
        drone.plot_trajectory()


#=======================================================================================================================
# GRAPH
def get_edges(graph: nx.Graph) -> np.ndarray:
    return np.array([(graph.nodes.data('pos')[i], graph.nodes.data('pos')[j]) for i, j in graph.edges()])


def get_vertex_position(graph: nx.Graph, v: int) -> np.ndarray[float]:
    return graph.nodes.data('pos')[v]


def plot_edge(ax, graph: nx.Graph, v1: int, v2: int, color: str):
    v1 = get_vertex_position(graph, v1)
    v2 = get_vertex_position(graph, v2)
    return ax.plot(*np.array([v1, v2]).T, color=color, alpha=1, lw=0.5)


def angle_between_points(A, B, C):
    # Convert the points to vectors
    BA = A - B
    BC = C - B

    # Compute the dot product and magnitudes
    dot_product = np.dot(BA, BC)
    mag_BA = np.linalg.norm(BA)
    mag_BC = np.linalg.norm(BC)

    # Compute the cosine of the angle
    cos_theta = dot_product / (mag_BA * mag_BC)

    # Get the angle in radians and convert to degrees
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical errors
    return np.degrees(angle_rad)


#=======================================================================================================================
# PLOT
def plot_fly_zone(dimension: np.ndarray) -> None:
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-dimension[0] / 2, dimension[0] / 2)
    ax.set_ylim3d(-dimension[1] / 2, dimension[1] / 2)
    ax.set_zlim3d(0, dimension[3])
    ax.set_aspect('equal')
    plt.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    #ax.set_xticks([-2, -1, 0, 1, 2,])
    #ax.set_yticks([-5, -4,-3,-2,-1,0,1,2,3,4,5])
    #ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([0, 1, 2])
    #ax.view_init(elev=40, azim=-100)
    ax.view_init(elev=50, azim=25)
    #ax.view_init(elev=25, azim=-150)


def sphere_surface(radius, resolution):
    theta = np.linspace(0, 2*np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    verts2 = []
    for i in range(len(phi) - 1):
        for j in range(len(theta) - 1):
            cp0 = radius * np.cos(phi[i])
            cp1 = radius * np.cos(phi[i + 1])
            sp0 = radius * np.sin(phi[i])
            sp1 = radius * np.sin(phi[i + 1])
            ct0 = np.cos(theta[j])
            ct1 = np.cos(theta[j + 1])
            st0 = radius * np.sin(theta[j])
            st1 = radius * np.sin(theta[j + 1])
            verts = [(cp0 * ct0, sp0 * ct0, st0), (cp1 * ct0, sp1 * ct0, st0), (cp1 * ct1, sp1 * ct1, st1),
                     (cp0 * ct1, sp0 * ct1, st1)]
            verts2.append(verts)
    return verts2


def color_list(idx: int) -> str:
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    return colors[idx]


def plot_optitrack_paths():
    ax = plt.gca()
    with open('C:/Users/Mate/PycharmProjects/Graph_based_trajectory_design/Saves/Take.csv', 'r') as f:
        lines = f.readlines()[1:]

    lines2 = []
    for line in lines:
        lines2 += [line.split(',')]

    lines2 = lines2[1:]
    df = np.array(lines2)

    data_dictionary = {}
    for i in range(df.shape[1]):
        # If the first line is "Rigid Body"
        if df[0, i] == "Rigid Body":
            # If we have already dealt with this concrete rigid body
            if df[1, i] in data_dictionary:
                pass
            else:
                data = df[5:, i:i + 6 + 1]
                np_data = np.array([])
                for j in range(data.shape[0]):
                    try:
                        # np_data = np.append(np_data, np.array(data[j, :], dtype = np.float64).reshape(1, -1))
                        if len(np_data) == 0:
                            np_data = np.array(data[j, :], dtype=np.float64).reshape(1, -1)
                        else:
                            np_data = np.vstack((np_data, np.array(data[j, :], dtype=np.float64).reshape(1, -1)))
                    except:
                        pass
                data_dictionary[df[1, i]] = np_data

    N = 20
    for key in data_dictionary:
        if key[:2] != 'cf':
            continue

        X = data_dictionary[key][:, 4][::N]
        Y = data_dictionary[key][:, 5][::N]
        Z = data_dictionary[key][:, 6][::N]

        points = np.array([X,-Z,Y]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        ax.add_collection3d(Line3DCollection(segments, alpha=1, lw=1, color="black"))


def set_anim_time(drones: list, frame: int, delay_between_frames: int, animation_speed: int,
                  skip_anim: bool = False) -> float:
    if skip_anim:
        drone_final_times = np.array([drone.trajectory_final_time() for drone in drones if not drone.returned_home])
        # Initial plannings
        if np.any(drone_final_times == -1):
            time = 0
        # After all drones returned to home
        elif len(drone_final_times) == 0:
            sys.exit(0)
        # During demo
        else:
            time = min(drone_final_times)

        return time

    return (frame - 1) * delay_between_frames / 1000 * animation_speed


def animate_drones(drones, time):
    for drone in drones:
        if drone.trajectory_final_time() != -1:
            drone.animate(t=time)


#=======================================================================================================================
# SPLINES
def fit_spline(points: np.ndarray) -> list:
    """
    Fit a B-spline to the given 3D coordinate sequence.
    :param points: [[x,y,z]...[x,y,z]]
    :return: spline: A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the
                     spline.
    """
    if len(points[:, 0]) == 2:
        spline_tck, *_ = interpolate.splprep([points[:, 0], points[:, 1], points[:, 2]], k=1, s=0)
    else:
        spline_tck, *_ = interpolate.splprep([points[:, 0], points[:, 1], points[:, 2]], k=2, s=0)
    return spline_tck


def evaluate_spline(spline_tck: list, der: int = 0) -> np.ndarray:
    u = np.linspace(0, spline_tck[0][-1], 100)
    spline_points = interpolate.splev(u, spline_tck, der=der)
    return spline_points


def parametrize_by_path_length(spline_tck: list) -> list:
    """
    Modify a given spline to have length accurare values.
    During evaluation the resulted slpine will give back positions for given path lengths
    """
    path = evaluate_spline(spline_tck)
    length = np.sum(np.sqrt(np.sum(np.diff(path) ** 2, axis=0)))
    s_params = np.linspace(0, length, 100)
    spline_tck, *_ = interpolate.splprep(path, k=2, s=0, u=s_params)
    return spline_tck


def get_trajectories(add_in_area: list | None = None, file_name: str | None = None) -> list:
    """
    Draw the trajectories for the obstacles and save them and/or load previously saved trajectories

    :param add_in_area: the extreme values of the added trajectories as: [x_min,x_max,y_min,y_max,z_min,z_max,]
    :param file_name: the file name of the previously generated trajectories
    :return: list of trajectories
    """
    trajectories = []
    if file_name:
        file_path = "Saves/Dynamic_obstacle_trajectories/"
        trajectories = load(file_path+file_name)
        print("Traj num:", len(trajectories))
        ax = plt.gca()
        for i, trajectory_set in enumerate(trajectories):
            for j, trajectory in enumerate(trajectory_set):
                spline_points = evaluate_spline(trajectory)
                if j == 0:
                    ax.plot(spline_points[0], spline_points[1], spline_points[2], lw=2, color="black", alpha=0.5)
                    ax.scatter(spline_points[0][0], spline_points[1][0], spline_points[2][0], color="black", marker=f"${i}$")
                else:
                    ax.plot(spline_points[0], spline_points[1], spline_points[2], lw=2, color="black")
                ax.scatter(spline_points[0][-1], spline_points[1][-1], spline_points[2][-1], color="black", alpha=0.5)

    if add_in_area is not None:
        if len(add_in_area) != 6 or add_in_area[0] >= add_in_area[1] or add_in_area[2] >= add_in_area[3] or add_in_area[4] >= add_in_area[5]:
            c_print("!!! WARNING !!! "
                    "\nPaths cannot be added because dimension values were defined incorrectly."
                    "\nPlease define: add_in_area=[x_min, x_max, y_min, y_max, z_min, z_max,]", "yellow")
            return trajectories

        spline_tcks = ask_for_paths(extreme_values=add_in_area)
        trajectories.extend(spline_tcks)

    return trajectories


def save_trajectories(trajectories: list, file_name: str) -> None:
    file_path = "Saves/Dynamic_obstacle_trajectories/"
    save(data=trajectories, file=file_path + file_name)


def ask_for_paths(extreme_values: list[float]) -> list:
    """
    Ask the user for the dynamic obstacle paths.

    :param extreme_values: [x_min, x_max, y_min, y_max,z_min, z_max]
    :return: paths_points: [[p]...[p]], where p=[[x,y,z]...[x,y,z]] -> points of the paths of the obstacles
    """
    paths_tck_u = []
    sub_paths_tck_u = []
    marked = []
    points = []

    ax = plt.gca()
    plt.subplots_adjust(left=0.25)
    slider_ax = plt.axes((0.05, 0.25, 0.0225, 0.63))  # left, right, height, width
    if extreme_values[1] < extreme_values[0]:
        c_print("!!! WARNING !!! "
                "\nWithin the extreme values x_min > x_max."
                "\nPlease define extreme_values=[x_min,x_max,y_min,y_max,z_min,z_max,]", "yellow")
        return paths_tck_u
    x_slider = Slider(ax=slider_ax, label='X', orientation="vertical",
                      valmin=extreme_values[0], valmax=extreme_values[1], valinit=0)
    slider_ax = plt.axes((0.12, 0.25, 0.0225, 0.63))  # left, right, height, width
    if extreme_values[3] < extreme_values[2]:
        c_print("!!! WARNING !!! "
                "\nWithin the extreme values y_min > y_max."
                "\nPlease define extreme_values=[x_min,x_max,y_min,y_max,z_min,z_max,]", "yellow")
        return paths_tck_u
    y_slider = Slider(ax=slider_ax, label='Y', orientation="vertical",
                      valmin=extreme_values[2], valmax=extreme_values[3], valinit=0)
    slider_ax = plt.axes((0.19, 0.25, 0.0225, 0.63))  # left, right, height, width
    if extreme_values[5] < extreme_values[4]:
        c_print("!!! WARNING !!! "
                "\nWithin the extreme values z_min > z_max."
                "\nPlease define extreme_values=[x_min,x_max,y_min,y_max,z_min,z_max,]", "yellow")
        return paths_tck_u
    z_slider = Slider(ax=slider_ax, label='Z', orientation="vertical",
                      valmin=extreme_values[4], valmax=extreme_values[5], valinit=0)
    button_ax = plt.axes((0.05, 0.16, 0.15, 0.04))
    set_point_button = Button(button_ax, 'Set point', hovercolor='white', color="lime")
    button_ax = plt.axes((0.05, 0.1, 0.15, 0.04))
    set_button = Button(button_ax, 'Set traj', hovercolor='grey', color="grey")
    button_ax = plt.axes((0.05, 0.04, 0.15, 0.04))
    save_button = Button(button_ax, 'Save traj set', hovercolor='grey', color="grey")
    active_set = {"Set traj": False,
                  "Save traj set": False}

    surface = sphere_surface(radius=0.05, resolution=10)
    marker = ax.add_collection3d(Poly3DCollection(surface, facecolor='red'))

    def activate(button: Button):
        button.color = "lime"
        button.hovercolor = "white"
        active_set[button.label.get_text()] = True

    def deactivate(button: Button):
        button.color = "grey"
        button.hovercolor = "grey"
        active_set[button.label.get_text()] = False

    def update(_):
        marker.set_verts(surface + np.array([x_slider.val, y_slider.val, z_slider.val]))
        deactivate(set_button)

    def set_point(_):
        marked.append([ax.scatter(x_slider.val, y_slider.val, z_slider.val, color="black")])
        points.append([x_slider.val, y_slider.val, z_slider.val])
        if len(points) >= 2:
            activate(set_button)

    def set_traj_fragment(_):
        if active_set["Set traj"]:
            sub_paths_tck_u.append(parametrize_by_path_length(fit_spline(np.array(points))))
            spline_points = evaluate_spline(sub_paths_tck_u[-1])
            ax.plot(spline_points[0], spline_points[1], spline_points[2], 'black', lw=2)

            points.clear()
            marked.clear()
            activate(save_button)
            deactivate(set_button)
            marked.append([ax.scatter(x_slider.val, y_slider.val, z_slider.val, color="grey")])
            points.append([x_slider.val, y_slider.val, z_slider.val])

    def save_traj(_):
        if active_set["Save traj set"]:
            paths_tck_u.append(copy.copy(sub_paths_tck_u))

            sub_paths_tck_u.clear()
            points.clear()
            for mark in marked:
                mark[0].remove()
            marked.clear()
            deactivate(save_button)

    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)
    set_point_button.on_clicked(set_point)
    set_button.on_clicked(set_traj_fragment)
    save_button.on_clicked(save_traj)
    plt.show()

    return paths_tck_u


#=======================================================================================================================
# PLANNING
def set_delay(drones, current_drone_ID, time, zero_delay):
    """
    The delay caused by the current path planning.

    e.g. path planning starts at t=10, take T=1. The next planning can not hapen before t+T
    """
    if zero_delay:
        return

    for d in drones:
        if d == drones[current_drone_ID]:
            d.wait_others = 0
        else:
            d.wait_others = max(0, time + drones[current_drone_ID].traj_gen_time)


def min_distance_between_drones(drones: list, t: float) -> float:
    min_dist = np.nan
    for drone in drones:
        if drone.trajectory is None:
            continue
        for other_drone in drones:
            if drone.ID == other_drone.ID or other_drone.trajectory is None:
                continue
            d = np.linalg.norm(drone.move(np.array(t)) - other_drone.move(np.array(t)))
            if d < min_dist or min_dist is np.nan:
                min_dist = d
    return min_dist
