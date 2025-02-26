from __future__ import annotations

import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import motioncapture
import networkx as nx
from typing import Tuple
import numpy as np
import math
import sys

from Utils.Utils import load, save


def obstacle_layout_set(layout_id: int) -> np.ndarray:
    if layout_id == 0:
        return np.array([])
    elif layout_id == 1:
        obstacle_positions = np.array([[1.2, 1.3, 2]])
        obstacle_dimensions = np.array([[0.6, 0.6]])/2
    elif layout_id == 2:
        obstacle_positions = np.array([[0, 0, 2.5], [0.5, 0.5, 2.5]])
        obstacle_dimensions = np.array([[0.75, 0.2], [0.5, 0.5]])/2
    elif layout_id == 3:
        N_obs = 30
        np.random.seed(100)
        obstacle_positions = np.multiply(np.array([8, 8]), np.random.rand(N_obs, 2)) - [4, 4]
        obstacle_positions = np.column_stack((obstacle_positions, np.ones((N_obs, 1)) * 4))
        obstacle_dimensions = np.array([[0.2, 0.2]]) / 2
    elif layout_id == 4:
        N_obs = 10
        np.random.seed(100)
        obstacle_positions = np.multiply(np.array([4, 4]), np.random.rand(N_obs, 2)) - [2, 2]
        obstacle_positions = np.column_stack((obstacle_positions, np.ones((N_obs, 1)) * 3))
        obstacle_dimensions = np.array([[0.5, 0.5]]) / 2

    elif layout_id == 5:
        obstacle_positions = np.array([[-2.25, 2.25, 1.0], [2.0, -1.9, 0.8],
                                       [-1.8, -1.9, 0.36], [4, 0, 1]])
        obstacle_dimensions = np.array([[0.5, 0.5]])/2

    elif layout_id == 6:
        "AR city"
        obstacle_positions = np.array([#[-2, -2, 1.0], [2, 2, 1], [-2, 2, 1], [2, -2, 1],
                                       [1, 2, 1], [-1, 2, 1], [-1, -2, 1], [1, -2, 1],
                                       [2, 1, 1], [-2, 1, 1], [-2, -1, 1], [2, -1, 1]])
        obstacle_dimensions = np.array([[0.2, 0.2]]) / 2
    else:
        return np.array([])

    if len(obstacle_dimensions) == 1 and len(obstacle_positions) > 1:
        obstacle_dimensions = obstacle_dimensions * np.ones((len(obstacle_positions), 1))
    elif 1 < len(obstacle_dimensions) != len(obstacle_positions):
        sys.exit("Not matching obstacle positions and dimensions")

    return np.column_stack((obstacle_positions, obstacle_dimensions))


class StaticObstacleS:
    def __init__(self,
                 layout_id: int = 0,
                 safety_distance: float = 0.0,
                 target_elevation: None | float = None,
                 new_measurement: bool = False,
                 measurement_file: None | str = None,
                 real_obs_base: None | dict = None):

        self.enclosed_spaces = obstacle_layout_set(layout_id=layout_id)
        self.safety_distance: float = safety_distance
        self.target_elevation: None | float = target_elevation
        self.num_of_real = 0

        if not new_measurement and measurement_file is not None and real_obs_base is not None:
            real_obs = self.load_real_obstacles(measurement_file, real_obs_base)
        elif new_measurement and real_obs_base is not None:
            real_obs = self.get_real_obstacles(measurement_file, real_obs_base) # TODO: implement it
        else:
            real_obs = None

        if len(self.enclosed_spaces) == 0 and real_obs is not None:
            self.num_of_real = len(real_obs)
            self.enclosed_spaces = real_obs
        elif real_obs is not None:
            self.num_of_real = len(real_obs)
            self.enclosed_spaces = np.vstack((real_obs, self.enclosed_spaces))

    def safe_enclosed_spaces(self) -> np.ndarray:
        if len(self.enclosed_spaces) == 0:
            return self.enclosed_spaces
        safe_ensclosed_spaces = copy.copy(self.enclosed_spaces)
        safe_ensclosed_spaces[:, 2:] = safe_ensclosed_spaces[:, 2:] + self.safety_distance if self.enclosed_spaces.size != 0 else np.array([])
        return safe_ensclosed_spaces

    def corners(self, safe: bool) -> np.ndarray:
        corners_of_static_obstacles = []
        if safe:
            obstacles = self.safe_enclosed_spaces()
        else:
            obstacles = self.enclosed_spaces
        for obstacle in obstacles:
            obstacle_corners = []
            for corner_x, corner_y, corner_z in zip([-1, 1, 1, -1, -1, -1, 1, 1], [-1, -1, 1, 1, 1, -1, -1, 1],
                                                    [-1, -1, -1, -1, 0, 0, 0, 0]):
                obstacle_corners.append([obstacle[0] + corner_x * obstacle[3], obstacle[1] + corner_y * obstacle[4],
                                         obstacle[2] + corner_z * obstacle[2]])
            corners_of_static_obstacles.append(obstacle_corners)
        return np.array(corners_of_static_obstacles)

    def plot(self, alpha: float, safety_zone: bool = False) -> None:
        ax = plt.gca()
        corners = self.corners(safe=safety_zone)
        for i in range(corners.shape[0]):
            p1 = corners[i][0]
            p2 = corners[i][1]
            p3 = corners[i][2]
            p4 = corners[i][3]
            p5 = corners[i][4]
            p6 = corners[i][5]
            p7 = corners[i][6]
            p8 = corners[i][7]

            if p8[2] > 0.01:
                x = [p1[0], p2[0], p3[0], p4[0]]
                y = [p1[1], p2[1], p3[1], p4[1]]
                z = [p1[2], p2[2], p3[2], p4[2]]
                verts = [list(zip(x, y, z))]
                ax.add_collection3d(
                    Poly3DCollection(verts, facecolors='black', alpha=alpha, edgecolor='k', linewidths=.1))
                x = [p1[0], p6[0], p5[0], p4[0]]
                y = [p1[1], p6[1], p5[1], p4[1]]
                z = [p1[2], p6[2], p5[2], p4[2]]
                verts = [list(zip(x, y, z))]
                ax.add_collection3d(
                    Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
                x = [p1[0], p6[0], p7[0], p2[0]]
                y = [p1[1], p6[1], p7[1], p2[1]]
                z = [p1[2], p6[2], p7[2], p2[2]]
                verts = [list(zip(x, y, z))]
                ax.add_collection3d(
                    Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
                x = [p3[0], p8[0], p5[0], p4[0]]
                y = [p3[1], p8[1], p5[1], p4[1]]
                z = [p3[2], p8[2], p5[2], p4[2]]
                verts = [list(zip(x, y, z))]
                ax.add_collection3d(
                    Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
                x = [p8[0], p7[0], p2[0], p3[0]]
                y = [p8[1], p7[1], p2[1], p3[1]]
                z = [p8[2], p7[2], p2[2], p3[2]]
                verts = [list(zip(x, y, z))]
                ax.add_collection3d(
                    Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
            x = [p5[0], p6[0], p7[0], p8[0]]
            y = [p5[1], p6[1], p7[1], p8[1]]
            z = [p5[2], p6[2], p7[2], p8[2]]
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='white', alpha=alpha, edgecolor='k', linewidths=.1))

    def delete_vertices_inside(self, vertices: np.ndarray, return_deleted: bool = False
                               ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        deleted = []
        for occupied_space in self.safe_enclosed_spaces():
            # Sides of obstacle
            front_side = occupied_space[0] + occupied_space[3]
            back_side = occupied_space[0] - occupied_space[3]
            rigth_side = occupied_space[1] + occupied_space[4]
            left_side = occupied_space[1] - occupied_space[4]
            top = occupied_space[2]

            # check verteces outside from obstacle
            in_front_of = front_side < vertices[:, 0]
            behinde = back_side > vertices[:, 0]
            to_the_rigth = rigth_side < vertices[:, 1]
            to_the_left = left_side > vertices[:, 1]
            above = top < vertices[:, 2]

            outside = np.bitwise_or(in_front_of, behinde)
            outside = np.bitwise_or(outside, to_the_rigth)
            outside = np.bitwise_or(outside, to_the_left)
            outside = np.bitwise_or(outside, above)
            if return_deleted and vertices[np.logical_not(outside)].size != 0:
                deleted += [vertices[np.logical_not(outside)]]
            vertices = vertices[outside]

        if return_deleted:
            return vertices, np.array(deleted)
        return vertices

    def delete_edges_inside(self, graph: nx.Graph) -> nx.Graph:
        for edge in graph.edges():
            if self.intersect(graph.nodes.data('pos')[edge[0]], graph.nodes.data('pos')[edge[1]],
                              self.safe_enclosed_spaces()):
                graph.remove_edge(edge[0], edge[1])

        return graph

    @staticmethod
    def intersect(v1: np.ndarray, v2: np.ndarray, enclosed_spaces: np.ndarray) -> bool:
        """
        Check if an edge is intersecting a static obstacle or not.

        :param v1: array([x,y,z]) coordinates of a vertex
        :param v2: array([x,y,z]) coordinates of a vertex
        :param enclosed_spaces: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                                -> the positions of the obstacles and their increased height and width
        :return: True if the edg intersect with an obstacle, False otherwise.
        """
        ds = 0.01  # min distance between points of the edge

        xmin = min(v1[0], v2[0])
        xmax = max(v1[0], v2[0])
        ymin = min(v1[1], v2[1])
        ymax = max(v1[1], v2[1])
        zmin = min(v1[2], v2[2])

        inside = False

        for i, occupied_space in enumerate(enclosed_spaces):
            front_side = occupied_space[0] + occupied_space[3]
            back_side = occupied_space[0] - occupied_space[3]
            rigth_side = occupied_space[1] + occupied_space[4]
            left_side = occupied_space[1] - occupied_space[4]
            top = occupied_space[2]

            # if the edge is allocated far from the obstacle skip further calculations
            if xmax < back_side or xmin > front_side or ymax < left_side or ymin > rigth_side or zmin > top:
                continue

            p = np.linspace(v1, v2, math.ceil(max(abs(v2 - v1)) / ds))
            inside_x = np.logical_and(p[:, 0] >= back_side,
                                      p[:, 0] <= front_side)
            inside_y = np.logical_and(p[:, 1] >= left_side,
                                      p[:, 1] <= rigth_side)
            inside_z = p[:, 2] <= occupied_space[2]
            inside_obs = np.logical_and(inside_z, (np.logical_and(inside_x, inside_y))).any()
            inside = inside_obs or inside

        return inside

    @staticmethod
    def generate_obstacles_from_measurements(real_obs_data, real_obs_base):
        enclosed_spaces = []

        # cf 'landing pads' to the front for easy home position definition'
        for obstacle in real_obs_data:
            if obstacle[0:2] == "cf":
                enclosed_spaces.append(np.append(real_obs_data[obstacle], 2*[real_obs_base["cf"]/2]))

        # add the rest of the measured obstacles
        for obstacle in real_obs_data:
            if obstacle[0:2] != "cf" and obstacle[0:2] in real_obs_base:
                enclosed_spaces.append(np.append(real_obs_data[obstacle], 2*[real_obs_base[obstacle[0:2]]/2]))

        return np.array(enclosed_spaces)

    def load_real_obstacles(self, measurement_file: str, real_obs_base: dict) -> np.ndarray:
        # top center of the obstacles:
        real_obs_data = load(f"Saves/Static_obstacle_mocap_measurements/{measurement_file}.pickle")

        return self.generate_obstacles_from_measurements(real_obs_data, real_obs_base)

    def get_real_obstacles(self, measurement_file: None | str, real_obs_base: dict) -> np.ndarray:
        # ----- MEASURE OPTITRACK DATA -----
        mc = motioncapture.MotionCaptureOptitrack("192.168.2.141")
        sample_size = 100
        obstacles_dict = {}
        for _ in range(sample_size):
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name in obstacles_dict:
                    obstacles_dict[name] = np.vstack((obstacles_dict[name], obj.position))
                else:
                    obstacles_dict[name] = obj.position

        for name in obstacles_dict:
            obstacles_dict[name] = np.sum(obstacles_dict[name], axis=0) / len(obstacles_dict[name])

        if measurement_file is not None:
            save(obstacles_dict, f"Saves/Static_obstacle_mocap_measurements/{measurement_file}.pickle")

        return self.generate_obstacles_from_measurements(obstacles_dict, real_obs_base)


# =============== SELF CHECK ======================
if __name__ == '__main__':

    def plot_obs(obs: StaticObstacleS, dims: list | None = None) -> None:
        ax = plt.axes(projection='3d')
        obs.plot(alpha=1)
        obs.plot(alpha=0.3, safety_zone=True)
        if dims is not None:
            if len(dims) != 6 or dims[0] >= dims[1] or dims[2] >= dims[3] or dims[4] >= dims[5]:
                print("Please give the limits of the axes as [x_min, x_max, y_min, y_max, z_min, z_max]")
                ax.axis('equal')
            else:
                ax.set_xlim3d(dims[0], dims[1])
                ax.set_ylim3d(dims[2], dims[3])
                ax.set_zlim3d(dims[4], dims[5])
                ax.set_aspect('equal')
        else:
            ax.axis('equal')
        plt.show()

    obs_ = StaticObstacleS(layout_id=6,
                           new_measurement=True,
                           real_obs_base={'bu': 0.3,
                                          'cf': 0.1,
                                          'ob': 0.1},
                           safety_distance=0.2)
    plot_obs(obs=obs_, dims=[-4, 4, -4, 4, 0, 3])
