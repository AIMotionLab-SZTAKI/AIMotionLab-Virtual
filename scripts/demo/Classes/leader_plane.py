from __future__ import annotations

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Utils.Utils import sphere_surface, c_print


class Leader_plane:
    def __init__(self,
                 radius: float,
                 trajectory: list,
                 revard_radius: float,
                 revard: float,
                 scenario,
                 start_time: float = 0.0,
                 safety_distance: float = 0.0,
                 silent: bool = False
                 ):
        self.trajectory: list = trajectory # tck
        self.radius: float = radius
        self.bounding_box: list[float] = [2 * (radius + safety_distance), 2 * (radius + safety_distance)]
        self.start_time: float = start_time
        self.revard_radius: float = revard_radius
        self.revard: float = revard  # [max, min]
        self.silent: bool = silent

        self.position: np.ndarray = self.move(np.array(start_time))
        self.vertex_revards, self.edge_revards = self.get_revard_matrices(scenario)

        self.facecolors: str = 'green'
        self.alpha: float = 0.4
        self.surface = sphere_surface(radius=self.revard_radius, resolution=11)
        self.my_plot = None

    def get_revard_matrices(self, scenario) -> (np.ndarray, np.ndarray):
        t_grid = np.arange(self.start_time, self.trajectory_final_time(), scenario.time_step)
        positions = self.move(t_grid)

        # -------- VERTICES --------
        vertices = np.array([scenario.graph.nodes.data('pos')[i] for i in scenario.graph.nodes])
        distances = np.linalg.norm(positions[:, np.newaxis]-vertices, axis=2)
        revard_v = distances <= self.revard_radius

        # -------- EDGES --------
        distances = np.linalg.norm(positions[:, np.newaxis, np.newaxis] - scenario.point_cloud, axis=3)
        distances = np.max(distances, axis=2)
        revard_e = distances <= self.revard_radius

        return revard_v, revard_e

    def trajectory_final_time(self) -> int | float:
        return round(self.trajectory_duration() + self.start_time, 5)

    def trajectory_duration(self) -> int | float:
        return round(self.trajectory[0][-1], 5)

    def move(self, t: np.ndarray) -> np.ndarray:
        t = t - self.start_time
        t = np.where(t > 0, t, 0) # set to 0 the negative time values
        t = np.where(t < self.trajectory_duration(), t, self.trajectory_duration())
        position = np.transpose(interpolate.splev(t, self.trajectory))
        return position

    def plot(self) -> None:
        ax = plt.gca()
        self.my_plot = ax.add_collection3d(Poly3DCollection(self.surface + self.position, alpha=self.alpha,
                                                            facecolors=self.facecolors, edgecolors='black',
                                                            linewidths=.1))

    def animate(self, t: float) -> None:
        if self.my_plot is None:
            self.plot()
        self.position = self.move(np.array(t))
        self.my_plot.set_verts(self.surface + self.position)

    def print_info(self, info: str) -> None:
        if not self.silent:
            c_print(f"Leader plane: {info}", 'green')
