import pickle
# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/mate/Desktop/mujoco/crazyflie-mujoco/')
sys.path.insert(2, '/home/crazyfly/Desktop/mujoco_digital_twin/crazyflie-mujoco/')

import mujoco
import glfw
import os
import mujoco
import numpy as np
from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl
import time
from assets.util import sync
from scipy.spatial.transform import Rotation
from assets.logger import Logger
from matplotlib import pyplot as plt
from hook_up_scenario.traj_opt_min_time import construct, plot_3d_trajectory
import timeit

class Trajectory:

    def __init__(self, model, data, control_step):
        # Reading model data
        self.model = model
        self.data = data

        # Trajectory parameters
        self.init_pos = [-1.5, 2, 1]  # initial position compared to the first load
        self.load_init = [[0.0, 0, 0.8], [-0.6, 0.6, 0.75], [-0.3, -0.6, 0.8]]
        self.load_target = [[2.2, 1.0, 0.77], [1.8, 1.0, 0.72], [1.4, 1.0, 0.77]]
        self.init_pos = [-0.5, 2.0, 1.8, np.pi/2]#, [load_target[0] - [0.3, 0, 0]]]
        self.load_mass = [0.15, 0.05, 0.1]

        self.pos_ref, self.vel_ref, self.yaw_ref, self.ctrl_type = None, None, None, None
        for num_sec in range(len(self.load_init)):

            self.init_pos_rel, self.load_target_rel = list(), list()
            for e1, e2, e3 in zip(self.init_pos, self.load_init[num_sec], self.load_target[num_sec]):

                self.init_pos_rel.append(e1-e2)
                self.load_target_rel.append(e3-e2)
            self.init_pos_rel.append(self.init_pos[-1])
            pos_ref_, vel_ref_, yaw_ref_, ctrl_type_ = construct(self.init_pos_rel, self.load_target_rel, self.load_mass[num_sec], plot_result=False)

            sys.stdout = sys.__stdout__
            # exec_time = timeit.timeit("construct(init_pos_rel, load_target_rel, False)", number=100,
            #                           globals={'init_pos_rel': init_pos_rel, 'load_target_rel': load_target_rel, 'construct': construct})
            # print(exec_time/200)
            pos_ref_ = pos_ref_ + np.array([self.load_init[num_sec]])
            # pos_ref, vel_ref, yaw_ref, ctrl_type = pos_ref + pos_ref_, vel_ref + vel_ref_, yaw_ref + yaw_ref_, \
            #                                        ctrl_type + ctrl_type_
            if self.pos_ref is None:
                self.pos_ref = pos_ref_
                self.vel_ref = vel_ref_
                self.yaw_ref = yaw_ref_
                self.ctrl_type = ctrl_type_
            else:
                self.pos_ref = np.vstack((self.pos_ref, pos_ref_))
                self.vel_ref = np.vstack((self.vel_ref, vel_ref_))
                self.yaw_ref = np.hstack((self.yaw_ref, yaw_ref_))
                self.ctrl_type = np.hstack((self.ctrl_type, ctrl_type_))
            if num_sec != len(self.load_init) - 1:
                self.init_pos = list()
                for e1, e2 in zip(self.load_target[num_sec], [0.3, 0, 0]):
                    self.init_pos.append(e1 - e2)
                self.init_pos.append(0)


        ## To obtain inertia matrix
        mujoco.mj_step(model, data)
        ### Controller
        self.controller = RobustGeomControl(model, data, drone_type='large_quad')
        self.controller.delta_r = 0
        self.mass = self.controller.mass
        self.controller_lqr = PlanarLQRControl(model)

        self.L = 0.4

        self.control_step = control_step

        q0 = np.roll(Rotation.from_euler('xyz', [0, 0, self.yaw_ref[0]]).as_quat(), 1)
        self.data.qpos[0:8] = np.hstack((self.pos_ref[0, :], q0, 0))
        self.episode_length = (self.pos_ref.shape[0] - 1) * control_step + 2.5

        #print(self.data.xquat)

    def evaluate(self, i, drone):
        
        # Get time and states
        simtime = self.data.time

        pos = drone.get_qpos()[:3]
        quat = drone.get_top_body_xquat()
        vel = drone.get_qvel()[0:3]
        ang_vel = drone.get_sensor_data()

        #print(self.data.sensordata[0:3])
        #print("get_sensor_data(): ", drone.get_sensor_data())
        #print()


        if simtime < 1:
            target_pos = self.pos_ref[0, :]
            target_rpy = np.array([0, 0, self.yaw_ref[0]])
            drone.set_ctrl(self.controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos, target_rpy=target_rpy))
        else:

            i_ = i - int(1 / self.control_step)

            if i_ < self.pos_ref.shape[0]:
                target_pos = self.pos_ref[i_, :]
                target_vel = self.vel_ref[i_, :]
                target_rpy = np.array([0, 0, self.yaw_ref[i_]])
                if 'lqr' in self.ctrl_type[i_]:
                    self.controller.mass = self.mass + float(self.ctrl_type[i_][-5:])
                    self.controller_lqr.mass = self.mass + float(self.ctrl_type[i_][-5:])
                    drone.set_ctrl(self.controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy))

                    alpha = drone.get_hook_qpos()
                    dalpha = drone.get_hook_qvel()
                    pos_ = pos.copy()
                    vel_ = vel.copy()
                    R_plane = np.array([[np.cos(self.yaw_ref[i_]), -np.sin(self.yaw_ref[i_])],
                                        [np.sin(self.yaw_ref[i_]), np.cos(self.yaw_ref[i_])]])
                    pos_[0:2] = R_plane.T @ pos_[0:2]
                    vel_[0:2] = R_plane.T @ vel_[0:2]
                    hook_pos = pos_ + self.L * np.array([-np.sin(alpha), 0, -np.cos(alpha)])
                    hook_vel = vel_ + self.L * dalpha * np.array([-np.cos(alpha), 0, np.sin(alpha)])
                    hook_pos = np.take(hook_pos, [0, 2])
                    hook_vel = np.take(hook_vel, [0, 2])
                    phi_Q = Rotation.from_quat(np.roll(quat, -1)).as_euler('xyz')[1]
                    dphi_Q = ang_vel[1]
                    target_pos_ = target_pos.copy()
                    target_pos_[0:2] = R_plane.T @ target_pos_[0:2]
                    target_pos_load = np.take(target_pos_, [0, 2]) - np.array([0, self.L])
                    lqr_ctrl = self.controller_lqr.compute_control(hook_pos,
                                                                hook_vel,
                                                                alpha,
                                                                dalpha,
                                                                phi_Q,
                                                                dphi_Q,
                                                                target_pos_load)
                    drone.ctrl0[0] = lqr_ctrl[0]
                    drone.ctrl2[0] = lqr_ctrl[2]
                elif 'geom_load' in self.ctrl_type[i_]:
                    self.controller.mass = self.mass + float(self.ctrl_type[i_][-5:])
                    self.controller.mass = self.mass + 0.1
                    drone.set_ctrl(self.controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy))
                else:
                    self.controller.mass = self.mass
                    drone.set_ctrl(self.controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy))
            else:
                target_pos = self.pos_ref[-1, :]
                target_vel = np.zeros(3)
                target_rpy = np.array([0, 0, self.yaw_ref[-1]])
                drone.set_ctrl(self.controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy))