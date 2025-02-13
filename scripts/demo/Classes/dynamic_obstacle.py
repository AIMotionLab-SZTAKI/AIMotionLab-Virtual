from __future__ import annotations

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from Utils.Utils import sphere_surface, c_print


class DynamicObstacle:
    def __init__(self,
                 radius: float,
                 height: float,
                 trajectory_list: list,
                 start_time: float = 0.0,
                 rest_time: float = 0.0,
                 safety_distance: float = 0.0,
                 repeate_movement: bool = False, # uses only the first trajectory of the traj. list
                 silent: bool = False
                 ):
        self.radius: float = radius
        self.height: float = height
        self.bounding_box = [2 * (radius + safety_distance), height + safety_distance]
        self.start_time: float = start_time
        self.rest_time: float = rest_time
        self.trajectory_list: list = trajectory_list # tck list
        self.repeate_movement: bool = repeate_movement
        self.silent: bool = silent

        self.active_trajectory_idx: int = 0
        self.speed_multiplier: float = 1.0 # TODO: implement it
        self.position: np.ndarray = self.move(np.array(start_time))
        self.ID: None | int = None

        self.facecolors: str = 'grey'
        self.alpha: float = 1.0
        self.surface = self.set_surface(resolution=7)
        self.my_plot = None
        self.trajectory_plot = []

    def update(self, scenario, t: float, drones: list) -> list[int]:
        if t >= self.start_time and self.ID is None:
            self.enter_scenario(scenario=scenario, t=t)
            return self.check_collisions(t0=t, T=scenario.occupancy_matrix_time_interval, ts=scenario.time_step,
                                         drones=drones)

        elif not self.repeate_movement and t > self.trajectory_final_time() and self.active_trajectory_idx < len(self.trajectory_list)-1:
            self.start_time = self.trajectory_final_time() + self.rest_time
            self.active_trajectory_idx += 1
            if scenario.use_GPU and t < scenario.ocm_times[-1]:
                scenario.update_OCM_GPU(dynamic_obj=self, t_update=t)
            elif t < scenario.ocm_times[-1]:
                scenario.update_OCM_CPU(dynamic_obj=self, t_update=t)
            return self.check_collisions(t0=t, T=scenario.occupancy_matrix_time_interval, ts=scenario.time_step,
                                         drones=drones)
        return[]

    def enter_scenario(self, scenario, t: float):
        self.ID = scenario.occupied_vertices.shape[0]
        scenario.occupied_vertices = np.concatenate((scenario.occupied_vertices,
                                                     np.expand_dims(np.full((len(scenario.ocm_times),
                                                                             scenario.occupied_vertices.shape[2]),
                                                                            False), axis=0)), axis=0)
        scenario.occupied_edges = np.concatenate((scenario.occupied_edges,
                                                  np.expand_dims(np.full((len(scenario.ocm_times),
                                                                          scenario.occupied_edges.shape[2]),
                                                                         False), axis=0)), axis=0)
        if scenario.use_GPU and t < scenario.ocm_times[-1]:
            scenario.update_OCM_GPU(dynamic_obj=self, t_update=t)
        elif t < scenario.ocm_times[-1]:
            scenario.update_OCM_CPU(dynamic_obj=self, t_update=t)

    def check_collisions(self, t0: float, T: float, ts: float, drones: list) -> list[int]:
        t_grid = np.arange(t0, t0+T+ts, ts)
        drones_pos = np.array([drone.move(t_grid) for drone in drones if drone.trajectory_start_time() >= 0])
        if len(drones_pos) == 0:
            return []

        obs_pos = self.move(t_grid)
        box = np.array([self.bounding_box[0], self.bounding_box[0], self.bounding_box[1]]) / 2

        pos_diff = np.abs(drones_pos - obs_pos)
        collisions = (pos_diff <= box).all(axis=2).any(axis=1)
        return [drones[i].ID for i, collision in enumerate(collisions) if collision]

    def trajectory_start_time(self) -> int | float:
        if self.trajectory_list:
            return round(self.trajectory_list[self.active_trajectory_idx][0][0], 5)
        return -1

    def trajectory_final_time(self) -> int | float:
        return round(self.trajectory_duration() + self.start_time, 5)

    def trajectory_duration(self) -> int | float:
        return round(self.trajectory_list[self.active_trajectory_idx][0][-1], 5)

    def move(self, t: np.ndarray) -> np.ndarray:
        if self.repeate_movement:
            if np.ndim(t) > 0:
                ts = t[1]-t[0]
                tf = round(round(self.trajectory_duration()/ts)*ts,5)
            else:
                tf = self.trajectory_duration()
            t = np.round(self.start_time + (t - self.start_time) % tf, 5)
        else:
            t = t - self.start_time
            t = np.where(t > 0, t, 0) # set to 0 the negative time values
            t = np.where(t < self.trajectory_duration(), t, self.trajectory_duration())
        position = np.transpose(interpolate.splev(t, self.trajectory_list[self.active_trajectory_idx]))
        #if np.ndim(position) == 1:
        #    position[2] = self.height/2
        #else:
        #    position[:, 2] = self.height/2
        return position

    def set_surface(self, resolution: int) -> np.ndarray:
        phi = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(-self.height/2, self.height/2, 2)

        top = np.array(sphere_surface(self.radius, resolution)) + [0, 0, self.height/2]

        PHI, Z = np.meshgrid(phi, z)
        CP = self.radius * np.cos(PHI)
        SP = self.radius * np.sin(PHI)
        XYZ = np.dstack([CP, SP, Z])
        verts = np.stack([XYZ[:-1, :-1], XYZ[:-1, 1:], XYZ[1:, 1:], XYZ[1:, :-1]], axis=-2).reshape(-1, 4, 3)

        return np.row_stack([top, verts])

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
            c_print(f"Obstacle {self.ID}: {info}", 'yellow')

    def plot_trajectory(self) -> None:
        fig = plt.gcf()
        ax = fig.gca()

        dt = 0.1
        t = np.arange(self.trajectory_start_time(), self.trajectory_final_time() + dt, dt)
        pos = self.move(t)

        points = np.array([pos[:, 0], pos[:, 1], pos[:, 2]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        for axes in self.trajectory_plot:
            axes.remove()
        self.trajectory_plot = [
            ax.add_collection3d(Line3DCollection(segments, alpha=0.3, lw=0.5, color="black"))]