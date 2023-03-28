import mosek
import numpy as np
import scipy.interpolate as si
import cvxopt as opt
import os
import pickle
import sys
from classes.trajectory_base import TrajectoryBase


class HookedDroneTrajectory(TrajectoryBase):
    def __init__(self):
        super().__init__()
        self.a_max = 0.3
        self.v_max = 1.46
        self.lam = 0.0875
        self.control_step = 0.01  # TODO: how do we know here the control step?
        self.num_lqr_steps = 300
        self.load_mass = 0.1
        self.rod_len = 0.4  # TODO: how do we know the length of the rod?
        self.traj = {}

    def evaluate(self, state, i, time, control_step) -> dict:
        planar_rotation = np.array([[np.cos(self.traj['yaw'][i]), -np.sin(self.traj['yaw'][i])],
                                    [np.sin(self.traj['yaw'][i]), np.cos(self.traj['yaw'][i])]])
        target_pos_ = self.traj['pos'][i].copy()
        target_pos_[0:2] = planar_rotation.T @ target_pos_[0:2]
        target_pos_load = np.take(target_pos_, [0, 2]) - np.array([0, self.rod_len])
        self.output = {
            "load_mass": self.load_mass,
            "target_pos": self.traj['pos'][i],
            "target_rpy": np.array([0, 0, self.traj['yaw'][i]]),
            "target_vel": self.traj['vel'][i],
            "target_acc": None,
            "target_ang_vel": np.zeros(3),
            "target_quat": None,
            "target_quat_vel": None,
            "target_pos_load": target_pos_load
        }
        return self.output

    @staticmethod
    def __plot_3d_trajectory(x, y, z, vel, title, load_target):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, vel.max())
        lc = Line3DCollection(segments, cmap='jet', norm=norm)
        # Set the values used for colormapping
        lc.set_array(vel)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, pad=0.15)
        cbar.set_label("velocity (m/s)")
        idx = {}
        idx['A'] = 0
        idx['B'] = (np.linalg.norm(points[:, 0, 0:2] - np.array([-0.3, 0]), axis=1)).argmin()
        idx['C'] = (np.linalg.norm(points[:, 0, :], axis=1)).argmin()
        idx['D'] = (np.linalg.norm(points[:, 0, :] - (load_target + np.array([0, 0, 0.5])), axis=1)).argmin()
        idx['E'] = (np.linalg.norm(points[:, 0, :] - load_target, axis=1)).argmin()
        idx['F'] = len(x) - 1
        for (k, v) in idx.items():
            ax.scatter(x[v], y[v], z[v], marker='x', color='black')
            ax.text(x[v], y[v], z[v]+0.2, k)
        traj_break_idx = np.argmax(np.abs(y) < 1e-4)
        # ax.scatter(x[traj_break_idx], y[traj_break_idx], z[traj_break_idx])
        # ax.scatter(0, 0, 0, marker='*')
        # ax.text(x[0], y[0], z[0], str([x[0], y[0], z[0]]))
        # ax.text(x[-1], y[-1], z[-1], "[{:.1f}, {:.1f}, {:.1f}]".format(x[-1], y[-1], z[-1]))
        ax.set_xlim(min(x)-0.3, max(x)+0.3)
        ax.set_ylim(min(y)-0.3, max(y)+0.3)
        ax.set_zlim(min(z)-0.1, max(z)+0.3)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(title)
        plt.show(block=False)

    @staticmethod
    def __compute_yaw_setpoints(init_yaw, final_yaw, duration):
        T = 0.5 * duration
        A = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 2, 0, 0, 0],
                      [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                      [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([init_yaw, 0, 0, final_yaw, 0, 0])
        x = np.linalg.inv(A) @ b
        t = np.arange(0, T, 0.01)
        yaw = np.sum(np.array([x[i] * t**i for i in range(6)]), axis=0)
        return np.hstack((yaw, final_yaw * np.ones(int((duration - T) / 0.01) + 1)))

    @staticmethod
    def __compute_yaw_spline(t, yaw_setpoints):
        t_span = t[-1] - t[0]
        num_elem = min([len(t), len(yaw_setpoints)])
        knots = np.linspace(t[0] + t_span / 7, t[-1] - t_span / 7, 7)
        spl = si.splrep(t[0:num_elem], yaw_setpoints[0:num_elem], k=5, task=-1, t=knots)
        return spl

    @staticmethod
    def __insert_wait_spl(spl, yaw_spl, wait_spl, wait_yaw_spl):
        spl = list(spl)
        spl.insert(3, wait_spl)
        yaw_spl.insert(3, wait_yaw_spl)
        spl = list(zip(*[spl_ + [yaw_spl_] for spl_, yaw_spl_ in zip(spl, yaw_spl)]))
        return spl

    def construct(self, init_pos, load_target, plot_result=False, save_result=False):
        save_splines = save_result
        enable_plotting = plot_result
        invert_traj = False
        hook_yaw = 0
        final_yaw = 0
        final_pos = [p + o for p, o in zip(load_target, [-0.3 * np.cos(final_yaw), -0.3 * np.sin(final_yaw), 0])]
        if init_pos[0] > 0:
            hook_yaw = np.pi
            invert_traj = True
            init_pos[0] = -1 * init_pos[0]
            init_pos[1] = -1 * init_pos[1]
            load_target[0] = -1 * load_target[0]
            load_target[1] = -1 * load_target[1]
            final_pos = [p + o for p, o in zip(load_target, [0.3 * np.cos(final_yaw), 0.3 * np.sin(final_yaw), 0])]
        # init_pos_list = [[-1.5, 2, 0.6], [-1, 0, 1.2], [0, 1.2, 0.6], [-0.3, 0.3, 1]]
        # load_target_list = [[1.5, 1.5, -0.01], [0.5, 2, -0.01], [1.8, 0, -0.01], [1, -1, -0.01]]
        # for init_pos, load_target in zip(init_pos_list, load_target_list):
        # init_pos = [-1.5, 2, 1]  # initial position compared to the load
        # load_target = [1.5, 1, -0.05]  # target position of the load (compared to initial position)
        num_sec = 5  # number of trajectory sections
        n = num_sec * [12]
        k = num_sec * [5]
        K = num_sec * [15]
        w = num_sec * [0.5]
        rho = num_sec * [0.1]
        bc = num_sec * [{}]
        a_max = num_sec*[self.a_max]
        v_max = num_sec*[self.v_max]
        lam = num_sec*[self.lam]
        planner = num_sec * [self._TrajectoryPlanner(*([None]*9))]
        xyz = num_sec * [np.array(())]
        yaw = num_sec * [np.array(())]
        # define safety distance from hook
        xs = -0.3
        zs = 0.1
        dzs = 0.12
        bc[0] = {'init_pos': init_pos[0:3], 'init_vel': [0, 0, 0], 'init_acc': [0, 0, 0],
                 'final_pos': [xs, 0, [0, zs]], 'final_vel': [[None, 0.3], 0, [None, 0.2]],
                 'final_dir': [[0.2, None], 0, [-dzs, dzs]],
                 'final_curve': [None, 0, None], 'init_yaw': init_pos[3], 'final_yaw': hook_yaw}
        params = [bc, n, k, rho, w, K, a_max, v_max, lam]
        params = [list(x) for x in zip(*params)]
        planner[0] = self._TrajectoryPlanner(*params[0])
        planner[0].construct_trajectory()
        xyz[0] = np.array(si.splev(planner[0].s_arr, planner[0].spl)).T
        yaw[0] = self.__compute_yaw_setpoints(bc[0]['init_yaw'], bc[0]['final_yaw'], planner[0].t_arr[-1])
        # t_0 = np.linspace(0, planner[0].t_arr[-1], 100)
        # vel_0 = planner[0].eval_trajectory(t_0, der=2)
        # plt.figure()
        # plt.plot(t_0, vel_0)
        # plt.show()

        final_der = np.array(si.splev(planner[0].s_arr[-1], planner[0].spl, der=1))
        final_der[1] = 0
        bc[1] = {'init_pos': [xyz[0][-1, 0], 0, xyz[0][-1, 2]], 'final_vel': 3 * [[None, 0.2]],
                 'init_vel': planner[0].vel_traj[-1] * final_der / np.linalg.norm(final_der),
                 'final_pos': [0, 0, 0], 'final_dir': [[0, None], 0, 0], 'final_curve': [None, None, 0],
                 'init_dir': final_der, 'init_yaw': hook_yaw, 'final_yaw': hook_yaw}
        params = [bc, n, k, rho, w, K, a_max, v_max, lam]
        params = [list(x) for x in zip(*params)]
        planner[1] = self._TrajectoryPlanner(*params[1])
        planner[1].construct_trajectory()
        xyz[1] = np.array(si.splev(planner[1].s_arr, planner[1].spl)).T
        yaw[1] = self.__compute_yaw_setpoints(bc[1]['init_yaw'], bc[1]['final_yaw'], planner[1].t_arr[-1])
        # t_1 = np.linspace(0, planner[1].t_arr[-1], 100)
        # vel_1 = planner[1].eval_trajectory(t_1, der=2)
        # plt.figure()
        # plt.plot(t_1, vel_1)
        # plt.show()

        final_der = np.array(si.splev(planner[1].s_arr[-1], planner[1].spl, der=1))
        final_der[1] = 0
        bc[2] = {'init_pos': [xyz[1][-1, 0], 0, xyz[1][-1, 2]],
                 'init_vel': planner[1].vel_traj[-1] * final_der / np.linalg.norm(final_der), 'final_vel': 3 * [0.0],
                 'final_pos': load_target + np.array([0.0, 0.0, 0.45]), 'init_curve': [None, None, [0.2, None]], 'final_dir': [0.0, 0.0, [None, -0.2]],#'final_curve': [0.0, 0.0, 0.0],
                 'init_dir': [[0.5, None], 0, 0], 'init_yaw': hook_yaw, 'final_yaw': 0}
        params = [bc, n, k, rho, w, K, a_max, v_max, lam]
        params = [list(x) for x in zip(*params)]
        planner[2] = self._TrajectoryPlanner(*params[2])
        planner[2].construct_trajectory()
        xyz[2] = np.array(si.splev(planner[2].s_arr, planner[2].spl)).T
        yaw[2] = self.__compute_yaw_setpoints(bc[2]['init_yaw'], bc[2]['final_yaw'], planner[2].t_arr[-1])

        bc[3] = {'init_pos': [xyz[2][-1, 0], xyz[2][-1, 1], xyz[2][-1, 2]],
                 'init_vel': 3 * [0], 'final_vel': 3 * [0],
                 'final_pos': load_target, 'init_dir': [0.0, 0.0, -0.01],
                 'init_yaw': 0}
        w[3] = 1e-5
        params = [bc, n, k, rho, w, K, a_max, v_max, lam]
        params = [list(x) for x in zip(*params)]
        planner[3] = self._TrajectoryPlanner(*params[3])
        planner[3].v_max = 0.3
        planner[3].construct_trajectory()
        xyz[3] = np.array(si.splev(planner[3].s_arr, planner[3].spl)).T
        yaw[3] = self.__compute_yaw_setpoints(bc[3]['init_yaw'], 0, planner[3].t_arr[-1])

        bc[4] = {'init_pos': [xyz[3][-1, 0], xyz[3][-1, 1], xyz[3][-1, 2]],
                 'init_vel': 3 * [0], 'final_vel': 3 * [0],
                 'final_pos': final_pos, 'init_curve': [None, None, 0],
                 'init_dir': [2 * np.cos(yaw[2][-1]), 2 * np.sin(yaw[2][-1]), 0], 'init_yaw': yaw[2][-1]}
        params = [bc, n, k, rho, w, K, a_max, v_max, lam]
        params = [list(x) for x in zip(*params)]
        planner[4] = self._TrajectoryPlanner(*params[4])
        planner[4].v_max = 0.2
        planner[4].construct_trajectory()
        xyz[4] = np.array(si.splev(planner[4].s_arr, planner[4].spl)).T
        yaw[4] = self.__compute_yaw_setpoints(bc[4]['init_yaw'], bc[4]['init_yaw'], planner[4].t_arr[-1])

        xyz = np.vstack([xyz_ for xyz_ in xyz])
        vel_traj = np.hstack([planner_.vel_traj for planner_ in planner])
        T = [planner_.t_arr[-1] for planner_ in planner]

        t = [np.arange(0, T_, 0.01) for T_ in T]
        pos, spl = list(zip(*[planner_.eval_trajectory(t_, 0) for planner_, t_ in zip(planner, t)]))
        pos = list(pos)
        yaw_spl = [self.__compute_yaw_spline(t_, yaw_) for t_, yaw_ in zip(t, yaw)]
        t_wait = self.num_lqr_steps * self.control_step
        knots = np.linspace(t_wait / 7, t_wait - t_wait / 7, 7)
        wait_spl = [si.splrep(np.linspace(0, t_wait, 20), pos[2][-1, i]*np.ones(20), k=5, task=-1, t=knots) for i in range(3)]
        wait_yaw_spl = si.splrep(np.linspace(0, t_wait, 20), yaw[2][-1]*np.ones(20), k=5, task=-1, t=knots)
        spl = self.__insert_wait_spl(spl, yaw_spl, wait_spl, wait_yaw_spl)



        vel = [planner_.eval_trajectory(t_, 1)[0] for planner_, t_ in zip(planner, t)]
        acc = [planner_.eval_trajectory(t_, 2)[0] for planner_, t_ in zip(planner, t)]
        ctrl_type = sum([len(pos[i]) for i in range(2)]) * ['geom'] + len(pos[2]) * ['geom_load' + "{:.3f}".format(self.load_mass)] + \
                    self.num_lqr_steps * ['lqr' + "{:.3f}".format(self.load_mass)] + len(pos[3]) * ['geom_load' + "{:.3f}".format(self.load_mass)] + len(pos[4]) * ['geom']
        pos = pos[0:3] + self.num_lqr_steps*[pos[2][-1, :]] + pos[3:]
        pos = np.vstack(pos)
        vel = vel[0:3] + self.num_lqr_steps * [np.zeros(3)] + vel[3:]
        vel = np.vstack(vel)
        acc = np.vstack(acc)
        yaw = yaw[0:3] + self.num_lqr_steps*[yaw[2][-1]] + yaw[3:]
        yaw = np.hstack(yaw)

        if save_splines:
            from datetime import datetime
            now = datetime.now()
            self.__save(spl, '../pickle/hook_up_spline_' + now.strftime("%H_%M_%S") + '.pickle')

        if enable_plotting:
            tle = 'Duration of trajectory: ' + "{:.2f}".format(sum(T)) + ' seconds'
            self.__plot_3d_trajectory(xyz[:, 0], xyz[:, 1], xyz[:, 2], vel_traj, tle, load_target)
            import matplotlib.pyplot as plt
            t = np.arange(0, sum(T) + 0.01, 0.01)
            plot_len = min((t.shape[0], pos.shape[0])) - 1
            fig = plt.figure()
            # plt.plot(t[0:plot_len], pos[0:plot_len, 0])
            # plt.plot(t[0:plot_len], pos[0:plot_len, 1])
            # plt.plot(t[0:plot_len], pos[0:plot_len, 2])
            acc = np.clip(acc, -0.4, 0.4)
            plt.plot(t[0:plot_len], acc[0:plot_len, 0])
            plt.plot(t[0:plot_len], acc[0:plot_len, 1])
            plt.plot(t[0:plot_len], acc[0:plot_len, 2])
            plt.xlabel('time (s)')
            plt.ylabel('acceleration (m/s$^2$)')
            plt.legend(('x', 'y', 'z'), loc='upper right')
            fig.subplots_adjust(left=0.18,
                                bottom=0.27,
                                right=0.99,
                                top=0.98,
                                wspace=0.5,
                                hspace=0.5
                                )
            # plt.ylim((-0.45, 0.45))
            # fig = plt.figure()
            # plt.plot(t[0:plot_len], vel[0:plot_len, 0])
            # plt.plot(t[0:plot_len], vel[0:plot_len, 1])
            # plt.plot(t[0:plot_len], vel[0:plot_len, 2])
            # plt.xlabel('time (s)')
            # plt.ylabel('velocity (m/s)')
            # plt.legend(('x', 'y', 'z'), loc='upper right')
            # fig.subplots_adjust(left=0.18,
            #                     bottom=0.27,
            #                     right=0.99,
            #                     top=0.98,
            #                     wspace=0.5,
            #                     hspace=0.5
            #                     )
            # plt.show(block=False)
            # plt.figure()
            # plt.plot(acc[:, 0])
            # plt.plot(acc[:, 1])
            # plt.plot(acc[:, 2])
            plt.show(block=True)
            # plt.figure()
            # plt.plot(t, acc)
            # plt.plot(t, np.linalg.norm(pos, axis=1))
            # plt.show()
        if invert_traj:
            pos[:, 0] = -1*pos[:, 0]
            pos[:, 1] = -1*pos[:, 1]
            vel[:, 0] = -1*vel[:, 0]
            vel[:, 1] = -1*vel[:, 1]
        T[3] = T[3] + self.num_lqr_steps * self.control_step
        self.traj = {'pos': pos, 'vel': vel, 'yaw': yaw, 'ctrl_type': ctrl_type}

    @staticmethod
    def __save(data, filename='pickle/optimal_trajectory.pickle'):
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    class _TrajectoryPlanner:
        def __init__(self, boundary_condition, num_sec, spl_degree, input_weight,
                     arc_length_weight, num_grid, a_max, v_max, lam):
            self.bc = boundary_condition
            self.n = num_sec
            self.k = spl_degree
            self.rho = input_weight
            self.w = arc_length_weight
            self.K = num_grid
            self.spl = []
            self.v_max = v_max
            self.a_max = a_max
            self.lam = lam
            self.s_arr = []
            self.t_arr = []
            self.vel_traj = []
            self.plot = False
            self.log_optim = False
            self.z_max = 10

        def eval_trajectory(self, t, der=0):
            t_span = t[-1] - t[0]
            knots = np.linspace(t[0] + t_span / 7, t[-1] - t_span / 7, 7)
            if der == 0:
                y = si.splev(self.s_arr, self.spl)
                # spl = [si.splrep(self.t_arr, y_, s=0, k=5) for y_ in y]
                spl = [si.splrep(self.t_arr, y_, k=5, task=-1, t=knots) for y_ in y]
                arr = np.array([si.splev(t, spl_) for spl_ in spl]).T
            elif der == 1:
                m = self.n
                idx = np.hstack([np.arange(i * (self.K + 1), (i + 1) * (self.K + 1) - 1) for i in range(m + 1)])
                idx = np.hstack((idx, (self.K + 1) * (m + 1) - 1))
                y = [si.splev(self.s_arr, self.spl, der=1)[i] * np.sqrt(self.b[idx].flatten()) for i in range(3)]
                # spl = [si.splrep(self.t_arr, y_, s=0, k=5) for y_ in y]
                spl = [si.splrep(self.t_arr, y_, k=5, task=-1, t=knots) for y_ in y]
                arr = np.array([si.splev(t, spl_) for spl_ in spl]).T
            elif der == 2:
                m = self.n
                idx = np.hstack([np.arange(i * (self.K + 1), (i + 1) * (self.K + 1) - 1) for i in range(m + 1)])
                s_arr = self._moving_average(self.s_arr, 2)
                # s_arr = np.hstack((s_arr, s_arr[-1]))
                b = 0.5 * (self.b[idx].flatten()[1:] + self.b[idx].flatten()[:-1])
                y = [si.splev(s_arr, self.spl, der=1)[i][:-1] * self.a.flatten()[:-1] +
                     si.splev(s_arr, self.spl, der=2)[i][:-1] * b for i in range(3)]
                spl = [si.splrep(self.t_arr[:-2], y_, s=0, k=5) for y_ in y]
                # spl = [si.splrep(self.t_arr[:-2], y_, k=5, task=-1, t=knots) for y_ in y]
                arr = np.array([si.splev(t, spl_) for spl_ in spl]).T
            else:
                raise NotImplementedError
            return arr, spl

        def eval_trajectory_from_load(self, t, der=0):
            L = 0.4
            rL = si.splev(self.s_arr, self.spl)
            # rL_spl = [si.splrep(self.t_arr, r_, s=1e-8, k=5) for r_ in rL]  # fit spline
            # ddrL = np.array([si.splev(self.t_arr, r_, der=2) for r_ in rL_spl]).T  # evaluate second derivative
            ddrL, _ = self.eval_trajectory(self.t_arr, der=2)
            p = ((ddrL - np.array([0, 0, 9.81])) /
                 np.tile(np.linalg.norm(ddrL - np.array([0, 0, 9.81]), axis=1), (3, 1)).T).T
            rL = np.array(rL)
            r = rL - p * L
            if der > 0:
                s = 1e-3
            else:
                s = 0
            r_spl = [si.splrep(self.t_arr, r_, s=s, k=5) for r_ in r]  # fit spline
            # plt.figure()
            # plt.plot(self.t_arr, ddrL)
            # plt.show()
            arr = np.array([si.splev(t, pos_spl_, der=der) for pos_spl_ in r_spl]).T - (der == 0) * np.array([0, 0, 0.4])
            return arr

        def construct_trajectory(self):
            if not self.log_optim:
                self.__blockPrint()
            self.spl = self._plan_spatial_trajectory()
            b = self._plan_temporal_trajectory()
            self.s_arr, self.t_arr, v_arr = self._compute_time_allocation(b)
            # self.vel_traj = (np.linalg.norm(np.array(si.splev(self.s_arr, self.spl, der=1)), axis=0) * np.gradient(self.s_arr, self.t_arr))[:-1]
            self.vel_traj = np.linalg.norm(self.eval_trajectory(self.t_arr, der=1)[0], axis=1)
            if not self.log_optim:
                self.__enablePrint()

        def _plan_spatial_trajectory(self):
            # Construct the spatial trajectory as B-splines
            s_max = (self.n + self.k - 2 * self.k)
            t = np.zeros(self.n + self.k + 1)
            t[0:self.k] = np.zeros(self.k)
            t[-self.k:] = s_max * np.ones(self.k)
            t[self.k:-self.k] = np.linspace(0, s_max, s_max + 1)

            # Cost function: minimize jerk and arc length on a grid
            T1 = self._derivative_transformation(t, self.k, self.n)
            T2 = self._derivative_transformation(t, self.k - 1, self.n + 1)
            T3 = self._derivative_transformation(t, self.k - 2, self.n + 2)
            T4 = self._derivative_transformation(t, self.k - 3, self.n + 3)
            P_eval = np.linspace(0, s_max, 1000)
            P_chol_jerk = 1e-2 * si.BSpline.design_matrix(P_eval, t, self.k - 3).toarray() @ T3 @ T2 @ T1
            P_chol_vel = 1e-3 * si.BSpline.design_matrix(P_eval, t, self.k - 1).toarray() @ T1
            P = self.w * np.kron(np.eye(3, dtype=int), P_chol_jerk.T @ P_chol_jerk) + \
                (1 - self.w) * np.kron(np.eye(3, dtype=int), P_chol_vel.T @ P_chol_vel)
            q = np.zeros(3 * self.n)

            A = np.zeros((1, 3*self.n))
            b = np.zeros((1, 1))
            G = np.zeros_like(A)
            h = np.zeros_like(b)

            A, b, G, h = self._set_pos_constraints(A, b, G, h, s_max, t)
            A, b, G, h = self._set_dir_constraints(A, b, G, h, s_max, t, T1)
            A, b, G, h = self._set_curve_constraints(A, b, G, h, s_max, t, T1, T2)

            A, b, G, h = A[1:, :], b[1:, :], G[1:, :], h[1:, :]
            A[np.abs(A) < 1e-10] = 0
            idx = np.linalg.norm(A, axis=1) > 1e-10
            A = A[idx, :]
            b = b[idx, :]

            G = opt.matrix(G, tc='d')
            h = opt.matrix(h, tc='d')

            P = opt.matrix(P)
            q = opt.matrix(q)
            A = opt.matrix(A, tc='d')
            b = opt.matrix(b, tc='d')
            opt.solvers.options['show_progress'] = self.log_optim
            opt.solvers.options['mosek'] = {mosek.iparam.log: self.log_optim,
                                            mosek.iparam.max_num_warnings: self.log_optim * 10}
            sol = opt.solvers.qp(P, q, A=A, b=b, G=G, h=h, solver='mosek')

            # exec_time = timeit.timeit("opt.solvers.qp(P, q, A=A, b=b, G=G, h=h, kktsolver='ldl')", number=200,
            #                           globals={'opt': opt, 'P': P, 'q': q, 'A': A, 'b': b, 'G': G, 'h': h})
            # print(exec_time/200)
            c = np.reshape(np.array(sol['x']), (3, self.n))
            # print(c)
            spl = (t, c, self.k)
            if self.plot:
                import matplotlib.pyplot as plt
                x = np.linspace(0, s_max, 1000)
                y = si.splev(x, spl)
                plt.figure()
                ax = plt.axes(projection="3d")
                ax.plot3D(*y)
                # plt.plot(x, y[0], x, y[1], x, y[2])
                plt.show(block=True)
            return spl

        def _set_pos_constraints(self, A, b, G, h, s_max, t):
            # Position
            pos_types = ['init_pos', 'final_pos']
            for pos_type in pos_types:
                if pos_type not in self.bc:
                    continue
                for i in range(3):
                    if not isinstance(self.bc[pos_type][i], list):
                        if self.bc[pos_type][i] is not None:
                            # Equality constraint
                            A_ = np.hstack((np.zeros((1, i * self.n)),
                                            si.BSpline.design_matrix(s_max * pos_types.index(pos_type), t,
                                                                     self.k).toarray(),
                                            np.zeros((1, (2 - i) * self.n))))
                            b_ = np.array([[self.bc[pos_type][i]]])
                            A, b = np.vstack((A, A_)), np.vstack((b, b_))
                    else:
                        # Inequality constraint
                        for j in range(2):
                            if self.bc[pos_type][i][j] is not None:
                                G_ = (-1) ** (j + 1) * np.hstack((np.zeros((1, i * self.n)),
                                                                  si.BSpline.design_matrix(
                                                                      s_max * pos_types.index(pos_type), t,
                                                                      self.k).toarray(),
                                                                  np.zeros((1, (2 - i) * self.n))))
                                h_ = (-1) ** (j + 1) * np.array([[self.bc[pos_type][i][j]]])
                                G, h = np.vstack((G, G_)), np.vstack((h, h_))
            G_ = np.hstack((np.zeros((100, 2 * self.n)), si.BSpline.design_matrix(np.linspace(0, s_max, 100), t,
                                                                                self.k).toarray()))
            h_ = self.z_max * np.ones((G_.shape[0], 1))
            G, h = np.vstack((G, G_)), np.vstack((h, h_))
            return A, b, G, h

        def _set_dir_constraints(self, A, b, G, h, s_max, t, T1):
            # Velocity (spatial)
            vel_types = ['init_dir', 'final_dir']
            for vel_type in vel_types:
                if vel_type not in self.bc:
                    continue
                s_eval = s_max * vel_types.index(vel_type) + (-1)**vel_types.index(vel_type) * 1e-6
                idx = [[0, 1], [0, 2], [1, 2]]
                for i in range(3):
                    if not isinstance(self.bc[vel_type][i], list):
                        if self.bc[vel_type][i] == 0:
                            # Equality constraint
                            A_ = np.hstack((np.zeros((1, i * self.n)),
                                            si.BSpline.design_matrix(s_eval, t,
                                                                     self.k - 1).toarray() @ T1,
                                            np.zeros((1, (2 - i) * self.n))))
                            b_ = np.array([[0]])
                            A, b = np.vstack((A, A_)), np.vstack((b, b_))
                            idx = [elem for elem in idx if i not in elem]
                    else:
                        # Inequality constraint
                        for j in range(2):
                            if self.bc[vel_type][i][j] is not None:
                                G_ = (-1) ** (j + 1) * np.hstack((np.zeros((1, i * self.n)),
                                                                  si.BSpline.design_matrix(
                                                                      s_eval, t,
                                                                      self.k - 1).toarray() @ T1,
                                                                  np.zeros((1, (2 - i) * self.n))))
                                h_ = (-1) ** (j + 1) * np.array([[self.bc[vel_type][i][j]]])
                                G, h = np.vstack((G, G_)), np.vstack((h, h_))
                for i in range(len(idx)):
                    if not isinstance(self.bc[vel_type][idx[i][0]], list) \
                            and not isinstance(self.bc[vel_type][idx[i][1]], list) \
                            and None not in [self.bc[vel_type][idx[i][0]], self.bc[vel_type][idx[i][1]]]:
                        # Equality constraint
                        A_ = np.hstack((np.zeros((1, (idx[i][0] == 1) * self.n)),
                                        self.bc[vel_type][idx[i][1]] * si.BSpline.design_matrix(s_eval, t,
                                                                                                self.k - 1).toarray() @ T1,
                                        np.zeros((1, (idx[i][1] - idx[i][0] == 2) * self.n)),
                                        -1 * self.bc[vel_type][idx[i][0]] * si.BSpline.design_matrix(s_eval, t,
                                                                                                     self.k - 1).toarray() @ T1,
                                        np.zeros((1, (idx[i][1] == 1) * self.n))))
                        b_ = np.array([[0]])
                        A, b = np.vstack((A, A_)), np.vstack((b, b_))
                        G_ = np.hstack((np.zeros((1, (idx[i][0] == 1) * self.n)),
                                        -1 * self.bc[vel_type][idx[i][0]] * si.BSpline.design_matrix(s_eval, t,
                                                                                                self.k - 1).toarray() @ T1,
                                        np.zeros((1, (2 - (idx[i][0] == 1)) * self.n))))
                        h_ = np.array([[-0.002]])
                        G, h = np.vstack((G, G_)), np.vstack((h, h_))
            return A, b, G, h

        def _set_curve_constraints(self, A, b, G, h, s_max, t, T1, T2):
            # Acceleration (spatial)
            acc_types = ['init_curve', 'final_curve']
            for acc_type in acc_types:
                if acc_type not in self.bc:
                    continue
                s_eval = s_max * acc_types.index(acc_type) + (-1)**acc_types.index(acc_type) * 1e-6
                for i in range(3):
                    if not isinstance(self.bc[acc_type][i], list):
                        if self.bc[acc_type][i] is not None:
                            # Equality constraint
                            A_ = np.hstack((np.zeros((1, i * self.n)),
                                            si.BSpline.design_matrix(s_eval, t,
                                                                     self.k - 2).toarray() @ T2 @ T1,
                                            np.zeros((1, (2 - i) * self.n))))
                            b_ = np.array([[self.bc[acc_type][i]]])
                            A, b = np.vstack((A, A_)), np.vstack((b, b_))
                    else:
                        # Inequality constraint
                        for j in range(2):
                            if self.bc[acc_type][i][j] is not None:
                                G_ = (-1) ** (j + 1) * np.hstack((np.zeros((1, i * self.n)),
                                                                  si.BSpline.design_matrix(
                                                                      s_max * acc_types.index(acc_type), t,
                                                                      self.k - 2).toarray() @ T2 @ T1,
                                                                  np.zeros((1, (2 - i) * self.n))))
                                h_ = (-1) ** (j + 1) * np.array([[self.bc[acc_type][i][j]]])
                                G, h = np.vstack((G, G_)), np.vstack((h, h_))
            return A, b, G, h

        def _set_equality_constraints(self, N, m, s_max):
            A = np.zeros((1, N))
            b = np.array([[0]])

            # Initial and final velocity
            vel_types = ['init_vel', 'final_vel']
            for vel_type in vel_types:
                if vel_type in self.bc:
                    s_eval = s_max * vel_types.index(vel_type) + (-1) ** vel_types.index(vel_type) * 1e-6
                    for dim in range(3):
                        if not isinstance(self.bc[vel_type][dim], list):
                            # Equality constraint
                            A_ = np.zeros(N)
                            coef = si.splev(s_eval, self.spl, der=1)[dim] ** 2
                            if abs(coef) < 1e-8:
                                continue
                            A_[(m + 1) * self.K * (vel_types.index(vel_type) + 1) + vel_types.index(vel_type)] = coef
                            b_ = (np.linalg.norm(self.bc[vel_type][dim])) ** 2
                            A, b = np.vstack((A, A_)), np.vstack((b, b_))

            A_ = np.zeros((m, N))
            idx1 = (self.K + 1) * np.arange(1, m + 1) - 1
            idx1 = idx1 + (m + 1) * self.K
            idx2 = idx1 + 1
            A_[np.arange(0, m), idx1] = 1
            A_[np.arange(0, m), idx2] = -1
            b_ = np.zeros((A_.shape[0], 1))
            A, b = np.vstack((A, A_)), np.vstack((b, b_))

            A_ = np.zeros(((m + 1) * self.K, N))
            idx1 = np.hstack([np.arange(i * (self.K + 1) + 1, (i + 1) * (self.K + 1)) for i in range(m + 1)])
            idx1 = idx1 + (m + 1) * self.K
            idx2 = idx1 - 1
            idx3 = np.arange(0, (m + 1) * self.K)
            A_[np.arange(0, A_.shape[0]), idx1] = 1
            A_[np.arange(0, A_.shape[0]), idx2] = -1
            A_[np.arange(0, A_.shape[0]), idx3] = -2 * 1 / self.K
            b_ = np.zeros((A_.shape[0], 1))
            A, b = np.vstack((A, A_)), np.vstack((b, b_))
            return A[1:, :], b[1:, :]

        def _set_inequality_constraints(self, N, m, s_max):
            G = np.zeros(((m + 1) * (self.K + 1), N))
            idx = np.arange((m + 1) * self.K, (m + 1) * (2 * self.K + 1))
            G[np.arange(0, G.shape[0]), idx] = -1
            h = np.zeros((G.shape[0], 1))

            G_ = np.zeros(((m + 1) * self.K, N))
            idx = np.hstack([np.arange(i * (self.K + 1), (i + 1) * (self.K + 1) - 1) for i in range(m + 1)]) + (m + 1) * self.K
            s = np.linspace(0, s_max, (m + 1) * self.K)
            h_ = self.v_max ** 2 * np.ones((G_.shape[0], 1))
            # h_[-10:] = h_[-10:] / 100
            for dim in range(3):
                coef = si.splev(s, self.spl, der=1)[dim] ** 2
                if np.max(np.abs(coef)) < 1e-8:
                    continue
                G_[np.arange(0, G_.shape[0]), idx] = coef
                G, h = np.vstack((G, G_)), np.vstack((h, h_))

            # Coefficients are similar to the last equality constraint
            G_ = np.zeros(((m + 1) * self.K, N))
            idx1 = np.hstack([np.arange(i * (self.K + 1) + 1, (i + 1) * (self.K + 1)) for i in range(m + 1)])
            idx1 = idx1 + (m + 1) * self.K
            idx2 = idx1 - 1
            idx3 = np.arange(0, (m + 1) * self.K)
            s_ = self._moving_average(s, 2)
            s_ = np.hstack((s_, s_[-1]))  # to match the dimensions

            df = si.splev(s_, self.spl, der=1)
            ddf = si.splev(s_, self.spl, der=2)
            h_ = self.a_max * np.ones((2 * G_.shape[0], 1))
            for dim in range(3):
                G_[np.arange(0, G_.shape[0]), idx1] = 1 / 2 * ddf[dim]
                G_[np.arange(0, G_.shape[0]), idx2] = 1 / 2 * ddf[dim]
                G_[np.arange(0, G_.shape[0]), idx3] = df[dim]
                if np.max(np.abs(G_)) < 1e-8:
                    continue
                G, h = np.vstack((G, G_, -G_)), np.vstack((h, h_))

            G_ = np.zeros(((m + 1) * self.K - 1, N))
            idx1 = np.hstack([np.arange(i * (self.K + 1) + 1, (i + 1) * (self.K + 1)) for i in range(m + 1)])
            idx1 = idx1[:-1] + (m + 1) * self.K  # b[k+1]
            idx2 = idx1 - 1  # b[k]
            idx3 = idx1 + 1  # b[k+2]
            idx4 = np.arange(1, (m + 1) * self.K)  # a[k+1]
            idx5 = idx4 - 1  # a[k]
            h_ = self.lam * np.ones((2 * G_.shape[0], 1))
            for dim in range(3):
                G_[np.arange(0, G_.shape[0]), idx1] = 1 / 2 * (ddf[dim][:-1] - ddf[dim][1:])
                G_[np.arange(0, G_.shape[0]), idx2] = 1 / 2 * ddf[dim][:-1]
                G_[np.arange(0, G_.shape[0]), idx3] = - 1 / 2 * ddf[dim][1:]
                G_[np.arange(0, G_.shape[0]), idx4] = - df[dim][1:]
                G_[np.arange(0, G_.shape[0]), idx5] = df[dim][:-1]
                if np.max(np.abs(G_)) < 1e-8:
                    continue
                G, h = np.vstack((G, G_, -G_)), np.vstack((h, h_))

            # Initial and final velocity
            vel_types = ['init_vel', 'final_vel']
            for vel_type in vel_types:
                if vel_type in self.bc:
                    s_eval = s_max * vel_types.index(vel_type) + (-1) ** vel_types.index(vel_type) * 1e-6
                    for dim in range(3):
                        if isinstance(self.bc[vel_type][dim], list):
                            # Inequality constraint
                            for i in range(2):
                                if self.bc[vel_type][dim][i] is not None:
                                    G_ = np.zeros(N)
                                    G_[(m + 1) * self.K * (vel_types.index(vel_type) + 1)] = (-1) ** (i + 1) * si.splev(s_eval, self.spl, der=1)[dim] ** 2
                                    if np.max(np.abs(G_)) < 1e-8:
                                        continue
                                    h_ = (-1) ** (i + 1) * self.bc[vel_type][dim][i] ** 2
                                    G, h = np.vstack((G, G_)), np.vstack((h, h_))

            # # Initial and final acceleration
            # acc_types = ['init_acc', 'final_acc']
            # for acc_type in acc_types:
            #     if acc_type in self.bc:
            #         s_eval = s_max * acc_types.index(acc_type) + (-1) ** acc_types.index(acc_type) * 1e-6
            #         df = si.splev(s_eval, self.spl, der=1)
            #         ddf = si.splev(s_eval, self.spl, der=2)
            #         G_ = np.zeros(N)
            #         for dim in range(3):
            #             if not isinstance(self.bc[acc_type][dim], list):
            #                 # Inequality constraint
            #                 idx1 = (m + 1) * self.K + acc_types.index(acc_type) * ((m + 1) * (self.K + 1) - 2)
            #                 G_[idx1] = 1 / 2 * ddf[dim]
            #                 G_[idx1 + 1] = 1 / 2 * ddf[dim]
            #                 G_[acc_types.index(acc_type) * (m + 1) * (self.K - 1)] = df[dim]
            #                 h_ = self.lam * np.ones((2, 1))
            #                 if np.max(np.abs(G_)) < 1e-8:
            #                     continue
            #                 G, h = np.vstack((G, G_, -G_)), np.vstack((h, h_))
            return G, h

        def _set_soc_constraints(self, N, m):
            G, h = [], []
            isq = 1 / np.sqrt(2)
            G_ = np.zeros(((m + 1) * self.K + 2, N))
            G_[0, -1] = -isq * 100
            G_[1, -1] = -isq * 100
            G_[2:, 0:(m + 1) * self.K] = np.eye((m + 1) * self.K)
            h_ = np.vstack((1, -isq, np.zeros(((m + 1) * self.K, 1))))
            G, h = G + [G_], h + [h_]

            for i in range((m + 1) * self.K):
                G_ = np.zeros((3, N))
                h_ = np.zeros((3, 1))
                G_[0:2, (m + 1) * (3 * self.K + 2) + i] = -isq
                G_[0:2, (m + 1) * (2 * self.K + 1) + i:(m + 1) * (2 * self.K + 1) + i + 2] = isq * np.array(
                    [[-1, -1], [1, 1]])
                h_[2] = np.sqrt(2)
                G, h = G + [G_], h + [h_]

            for i in range((m + 1) * (self.K + 1)):
                G_ = np.zeros((3, N))
                h_ = np.zeros((3, 1))
                G_[0:2, (m + 1) * self.K + i] = -1
                G_[2, (m + 1) * (2 * self.K + 1) + i] = -2
                h_[0:2, 0] = np.array([1, -1])
                G, h = G + [G_], h + [h_]
            return G, h

        def _plan_temporal_trajectory(self):
            n = self.spl[1].shape[1]
            k = self.spl[2]
            s_max = (n + k - 2 * k)
            m = n

            # The optimization variable is x = [a, b, c, d, t],
            # dim(x) = N = (m + 1) * (K + (K+1) + (K+1) + K) + 1 = (m + 1) * (4K + 2) + 1

            # Objective function
            N = (m + 1) * (4 * self.K + 2) + 1
            c = np.zeros(N)
            c[(m + 1) * (3 * self.K + 2):-1] = 2e-5
            c[-1] = self.rho * 1e-5
            # c = c

            # Equality constraints
            A, b = self._set_equality_constraints(N, m, s_max)

            # Linear matrix inequalities
            G_l, h_l = self._set_inequality_constraints(N, m, s_max)

            # Second-order cone constraints
            G_soc, h_soc = self._set_soc_constraints(N, m)

            c = opt.matrix(c)
            G_l = opt.matrix(np.vstack((A, -A, G_l)))
            h_l = opt.matrix(np.vstack((b, -b, h_l)))
            G_soc = [opt.matrix(G_k) for G_k in G_soc]
            h_soc = [opt.matrix(h_k) for h_k in h_soc]
            A = opt.matrix(A)
            b = opt.matrix(b)

            opt.solvers.options['show_progress'] = self.log_optim
            opt.solvers.options['mosek'] = {mosek.iparam.log: self.log_optim,
                                            mosek.iparam.max_num_warnings: self.log_optim * 10}
            sol = opt.solvers.socp(c=c, Gl=G_l, hl=h_l, Gq=G_soc, hq=h_soc, solver='mosek')
            print(sol)
            if sol['x'] is None:
                self.__enablePrint()
                print('Solution failed with parameters' + str([self.w, self.v_max, self.a_max, self.lam]))
            self.b = sol['x'][(m + 1) * self.K:(m + 1) * (2 * self.K + 1)]
            self.b = np.array(self.b)
            self.a = np.array(sol['x'][0:(m + 1) * self.K])
            if np.min(self.b) < -1e-10:
                print('b has negative values, something went wrong during optimization')
            return np.abs(self.b)

        def _compute_time_allocation(self, b):
            n = self.spl[1].shape[1]
            k = self.spl[2]
            s_max = (n + k - 2 * k)
            m = n
            s_arr = np.vstack([np.linspace(i * s_max / (m + 1), (i + 1) * s_max / (m + 1), self.K + 1) for i in range(m + 1)])
            ds = s_max / (m + 1) / self.K
            t_arr = np.zeros_like(s_arr)
            v_arr = []
            for i in range(m + 1):
                for k in range(0, self.K + 1):
                    if k == 0 and i == 0:
                        continue
                    elif k == 0:
                        t_arr[i, k] = t_arr[i - 1, -1]
                    else:
                        t_arr[i, k] = t_arr[i, k - 1] + 2 * ds / (np.sqrt(b[i * self.K + k - 1]) + np.sqrt(b[i * self.K + k]))
                        # t_arr[i, k] = t_arr[i, k - 1] + ds / np.sqrt(b[i * self.K + k])
                        v_arr = v_arr + [np.sum(np.array(si.splev(s_arr[i, k], self.spl, der=1)) ** 2, axis=0) * b[i * self.K + k]]
            s_arr, t_arr, v_arr = np.hstack((s_arr[:, :-1].flatten(), s_arr[-1, -1])), \
                                  np.hstack((t_arr[:, :-1].flatten(), t_arr[-1, -1])), \
                                  np.sqrt(np.array(v_arr)).flatten()
            return s_arr, t_arr, v_arr

        @staticmethod
        def _derivative_transformation(t, k, n):
            # Computes a transformation matrix T that can be used to express the
            # derivative of a B-spline by using the same coefficients that the spline has.
            def zerocheck(a):
                return int(a == 0) + a * int(a != 0)

            T = np.zeros((n + 1, n))
            for i in range(n):
                T[i, i] = k / zerocheck(t[i+k]-t[i])
                T[i+1, i] = -k / zerocheck(t[i+k+1]-t[i+1])
            return T

        @staticmethod
        def _moving_average(a, n):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        # Disable
        @staticmethod
        def __blockPrint():
            sys.stdout = open(os.devnull, 'w')

        # Restore
        @staticmethod
        def __enablePrint():
            sys.stdout = sys.__stdout__