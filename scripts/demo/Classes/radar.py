from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx


def radar_layout_set(layout_id: int) -> list:
    if layout_id == 0:
        return []
    elif layout_id == 1:
        return [{'position': [-2.25, 2.25, 1.0],
                 'a': 1.0,
                 'exp': 1.5,
                 'tilt': 0.3,
                 'height_scale': 1,
                 'risk':[5, 1]},

                {'position': [1.0, 1.9, 0.8],
                 'a': 1.0,
                 'exp': 1.5,
                 'tilt': 0.0,
                 'height_scale': 1,
                 'risk': [5, 1]},

                {'position': [-1.8, -1.9, 0.36],
                 'a': 1.0,
                 'exp': 2.5,
                 'tilt': 0.3,
                 'height_scale': 1,
                 'risk':[5, 1]},

                {'position': [4, 0, 1],
                 'a': 0.75,
                 'exp': 2.5,
                 'tilt': 0.,
                 'height_scale': 1,
                 'risk': [1, 0.5]}
                ]
    elif layout_id == 2:
        return [{'position': [0, 0, 0],
                 'a': 0.75,
                 'exp': 2.5,
                 'tilt': 0.,
                 'height_scale': 1,
                 'risk':[5, 1]}]
    elif layout_id == 3:
        return [{'position': [4, 0, 1],
                 'a': 0.75,
                 'exp': 2.5,
                 'tilt': 0.,
                 'height_scale': 1,
                 'risk':[1, 0.5]}]
    elif layout_id == 4:
        c_min = 0.5
        c_max = 5
        return [
                {'position': [4, 0, 1],'a': 1.0,'exp': 2.5,'tilt': 0.,'height_scale': 1,'risk': [c_max/2, c_min/2]},

                {'position': [-2.25, 2.25, 1.0],'a': 1.0,'exp': 1.5,'tilt': 0.0,'height_scale': 1,'risk':[c_max, c_min]},
                {'position': [2.0, -1.9, 0.8],'a': 1.0,'exp': 1.5,'tilt': 0.0,'height_scale': 1,'risk': [c_max, c_min]},
                {'position': [-1.8, -1.9, 0.36],'a': 1.0,'exp': 2.5,'tilt': 0.0,'height_scale': 1,'risk':[c_max, c_min]}
                ]
    else:
        return []


class RadarS:
    def __init__(self, layout_id: int) -> None:
        self.positions: np.ndarray = np.array([radar['position'] for radar in radar_layout_set(layout_id)])  # center of the radar field
        self.a: np.ndarray = np.array([radar['a'] for radar in radar_layout_set(layout_id)])  # size of the radar field (radius = 2 * a)
        self.exp: np.ndarray = np.array([radar['exp'] for radar in radar_layout_set(layout_id)])  # shape of the radar field
        self.tilt: np.ndarray = np.array([radar['tilt'] for radar in radar_layout_set(layout_id)])  # how much the teardrop is tilted before being rotated by 360°
        self.height_scale: np.ndarray = np.array([radar['height_scale'] for radar in radar_layout_set(layout_id)])  # scale the height of the model
        self.risk: np.ndarray = np.array([radar['risk'] for radar in radar_layout_set(layout_id)]).transpose()  # scale the height of the model [[max][min]]

    def add_risk(self, graph: nx.Graph, point_cloud: np.ndarray) -> nx.Graph:
        def move_points_on_sphere(points: np.ndarray, delta_theta: float):
            # convert to polar coordinates
            r = np.sqrt(points[0] ** 2 + points[1] ** 2 + points[2] ** 2)
            theta = np.arccos(points[2] / r)
            phi = np.arctan2(points[1], points[0])

            theta_n = theta + delta_theta

            # convert back to cartesian
            x_n = r * np.sin(theta_n) * np.cos(phi)
            y_n = r * np.sin(theta_n) * np.sin(phi)
            z_n = r * np.cos(theta_n)

            return np.array((x_n, y_n, z_n))

        # ---- VERTICES ----
        for i, position in enumerate(self.positions):
            print('radar:', position)
            for j, vertex in enumerate(graph.nodes.data()):
                dx, dy, dz = vertex[1]['pos'] - position

                p = move_points_on_sphere(points=np.array((dx, dy, dz)), delta_theta=self.tilt[i])
                d = np.sqrt(p[0] ** 2 + p[1] ** 2)

                if ((self.a[i] - d) / self.a[i]) < -1 or 1 < ((self.a[i] - d) / self.a[i]):
                    continue

                z_lim = self.height_scale[i] * self.a[i] * np.sin(np.arccos((self.a[i] - d) / self.a[i])) * np.sin(
                    np.arccos((self.a[i] - d) / self.a[i]) / 2.0) ** self.exp[i]

                p[2] += position[2]

                if d <= 2 * self.a[i] and abs(p[2] - position[2]) <= z_lim:
                    c_a = (2 * self.a[i] - d) / (2 * self.a[i])
                    c_a = max(0, c_a)
                    c_z = (z_lim - abs(p[2] - position[2])) / z_lim
                    c_z = max(0, c_z)
                    vertex[1]['detection_risk'] += (self.risk[0, i]-self.risk[1, i]) * c_a * c_z + self.risk[1, i]

        # ---- EDGES ----
        for i, position in enumerate(self.positions):
            dx = np.array(point_cloud[:, :, 0] - position[0])
            dy = np.array(point_cloud[:, :, 1] - position[1])
            dz = np.array(point_cloud[:, :, 2] - position[2])

            p = move_points_on_sphere(points=np.array((dx, dy, dz)), delta_theta=self.tilt[i])
            d = np.sqrt(p[0] ** 2 + p[1] ** 2)
            z_lim = self.height_scale[i] * self.a[i] * np.sin(np.arccos((self.a[i] - d) / self.a[i])) * np.sin(np.arccos((self.a[i] - d) / self.a[i]) / 2.0) ** self.exp[i]

            p[2] += position[2]

            for j, edge in enumerate(graph.edges.data()):
                if np.any(np.logical_and(d[j] <= 2 * self.a[i], abs(p[2, j] - position[2]) <= z_lim[j])):
                    c_a = (2 * self.a[i] - d[j]) / (2 * self.a[i])
                    c_a[c_a < 0] = 0
                    c_z = (z_lim[j] - abs(p[2, j] - position[2])) / z_lim[j]
                    c_z[c_z < 0] = 0

                    edge[2]['detection_risk'] += (self.risk[0, i] - self.risk[1, i]) * np.nanmax(c_a * c_z) + self.risk[1, i]
        return graph

    def plot(self):
        def quat_array_quat_multiply(quat_array, quat):
            w0, x0, y0, z0 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
            w1, x1, y1, z1 = quat
            return np.stack((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), axis=1)

        def quat_quat_array_multiply(quat, quat_array):

            w0, x0, y0, z0 = quat
            w1, x1, y1, z1 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
            return np.stack((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), axis=1)

        def q_conjugate(q):
            w, x, y, z = q
            return [w, -x, -y, -z]

        def quaternion_from_euler(roll, pitch, yaw):
            """
            Convert an Euler angle to a quaternion.

            Input
              :param roll: The roll (rotation around x-axis) angle in radians.
              :param pitch: The pitch (rotation around y-axis) angle in radians.
              :param yaw: The yaw (rotation around z-axis) angle in radians.

            Output
              :return qw, qx, qy, qz: The orientation in quaternion [w,x,y,z] format
            """
            qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - \
                 np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
            qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + \
                 np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
            qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - \
                 np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
            qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + \
                 np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

            return [qw, qx, qy, qz]

        def quat_vect_array_mult(q, v_array):
            q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
            return quat_quat_array_multiply(q_conjugate(q), quat_array_quat_multiply(q_array, q))[:, 1:]

        ax = plt.gca()
        resolution = 100
        rot_resolution = 51
        for i, position in enumerate(self.positions):
            xs = np.linspace(0., 2 * self.a[i], resolution)
            zps = self.height_scale[i] * self.a[i] * np.sin(np.arccos((-xs + self.a[i]) / self.a[i])) * (
                np.sin(np.arccos((-xs + self.a[i]) / self.a[i]) / 2.)) ** self.exp[i]
            ys = np.zeros(2 * resolution - 2)
            zns = -zps[1:-1].copy()
            zs = np.append(zps, np.flip(zns))
            xs = np.append(xs, np.flip(xs[1:-1]))
            points = np.vstack((xs, ys, zs)).T
            euler_tilt = np.array((0.0, -self.tilt[i], 0.0))
            quat_tilt = quaternion_from_euler(*euler_tilt)
            points = quat_vect_array_mult(quat_tilt, points)

            rot_step = 2 * math.pi / rot_resolution
            current_rotation = rot_step
            for _ in range(rot_resolution - 1):
                euler = np.array((0.0, 0.0, current_rotation))
                quat = quaternion_from_euler(*euler)
                rot_points = quat_vect_array_mult(quat, points) + position
                ax.plot(rot_points[:, 0], rot_points[:, 1], rot_points[:, 2], color='red', alpha=0.2)
                current_rotation += rot_step


def plot_radars(radars: RadarS, dims: list | None = None) -> None:
    """
    Plot the radar fields.
    :param radars: radars obj
    :param dims: dimensions of the plot
    """
    ax = plt.axes(projection='3d')
    radars.plot()
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


if __name__ == '__main__':
    radars_ = RadarS(layout_id=1)
    plot_radars(radars_, dims=[-4, 4, -4, 4, 0, 3])
