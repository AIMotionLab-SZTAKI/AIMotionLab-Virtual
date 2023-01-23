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
from assets.splines.spline import BSpline, BSplineBasis


HOOK_UP_3_LOADS = 1
HOOK_UP = 2
FLY = 3
FLIP = 4

class Trajectory():
    """ Base class for Drone trajectories
    """
    def __init__(self, control_step):

        self.control_step = control_step

        # self.output needs to be updated and returned in evaluate()
        self.output = {
            "controller_name" : None,
            "load_mass" : 0.0,
            "target_pos" : None,
            "target_rpy" : np.zeros(3),
            "target_vel" : np.zeros(3),
            "target_pos" : None,
            "target_acc" : None,
            "target_quat" : None,
            "target_quat_vel" : None,
            "target_pos_load" : None
        }
    

    def evaluate(self, i, simtime) -> dict:
        # must implement this method
        raise NotImplementedError("Derived class must implement evaluate()")
        


class TestTrajectory(Trajectory):

    def __init__(self, control_step, scenario = HOOK_UP_3_LOADS):
        super().__init__(control_step)
        
        self.scenario = scenario


        if scenario == HOOK_UP_3_LOADS:
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

            self.L = 0.4

            self.q0 = np.roll(Rotation.from_euler('xyz', [0, 0, self.yaw_ref[0]]).as_quat(), 1)
            self.episode_length = (self.pos_ref.shape[0] - 1) * control_step + 2.5
        
        elif scenario == HOOK_UP:
            print("hook up trajectory not yet implemented")
        
        elif scenario == FLY:
            # don't have to do anything
            pass
            
        
        elif scenario == FLIP:
            self.flip_spline_params = np.loadtxt("../../crazyflie-mujoco/assets/pos_spline.csv", delimiter=',')
            self.flip_traj = [BSpline(BSplineBasis(self.flip_spline_params[i, 1:18], int(self.flip_spline_params[i, 0])), self.flip_spline_params[i, 18:]) for i in range(3)]
            self.flip_vel = [self.flip_traj[i].derivative(1) for i in range(3)]
            self.flip_acc = [self.flip_traj[i].derivative(2) for i in range(3)]
        



    def evaluate(self, i, simtime) -> dict:


        if self.scenario == HOOK_UP_3_LOADS:
        

            if simtime < 1:
                self.output["target_pos"] = self.pos_ref[0, :]
                self.output["target_rpy"] = np.array([0, 0, self.yaw_ref[0]])
                self.output["controller_name"] = "geom_pos"
                
            else:

                i_ = i - int(1 / self.control_step)

                if i_ < self.pos_ref.shape[0]:
                    self.output["target_pos"] = self.pos_ref[i_, :]
                    self.output["target_vel"] = self.vel_ref[i_, :]
                    self.output["target_rpy"] = np.array([0, 0, self.yaw_ref[i_]])

                    if 'lqr' in self.ctrl_type[i_]:
                        self.output["load_mass"] = float(self.ctrl_type[i_][-5:])
                        self.output["controller_name"] = "lqr"

                        R_plane = np.array([[np.cos(self.yaw_ref[i_]), -np.sin(self.yaw_ref[i_])],
                                            [np.sin(self.yaw_ref[i_]), np.cos(self.yaw_ref[i_])]])
                                            
                        target_pos_ = self.output["target_pos"].copy()
                        target_pos_[0:2] = R_plane.T @ target_pos_[0:2]
                        self.output["target_pos_load"] = np.take(target_pos_, [0, 2]) - np.array([0, self.L])

                    elif 'geom_load' in self.ctrl_type[i_]:
                        self.output["load_mass"] = float(self.ctrl_type[i_][-5:])
                        self.output["controller_name"] = "geom_pos"
                    else:
                        self.output["load_mass"] = 0
                        self.output["controller_name"] = "geom_pos"
                else:
                    self.output["target_pos"] = self.pos_ref[-1, :]
                    self.output["target_vel"] = np.zeros(3)
                    self.output["target_rpy"] = np.array([0, 0, self.yaw_ref[-1]])
                    self.output["controller_name"] = "geom_pos"
                
        
        elif self.scenario == FLY:

            self.output["controller_name"] = "geom_pos"

            if simtime < 1:
                self.output["target_pos"] = np.array([0, 0, 1])

            else:
                traj_freq = 2
                t = i * self.control_step
                self.output["target_pos"] = np.array([0.8 * np.sin(traj_freq * t), 0.8 * np.sin(2 * traj_freq * (t - np.pi / 2)),
                                    1])
                self.output["target_vel"] = np.array([0.8 * traj_freq * np.cos(traj_freq * t),
                                    0.8 * 2 * traj_freq * np.cos(2 * traj_freq * (t - np.pi / 2)), 0])

        elif self.scenario == FLIP:

            if simtime < 2:
                self.output["target_pos"] = np.array([0, 0, 0.3])
                self.output["controller_name"] = "geom_pos"
                
            elif simtime < 2.9:
                
                eval_time = (simtime - 2) / 0.9
                target_pos = np.array([self.flip_traj[0](eval_time)[0], 0, self.flip_traj[1](eval_time)[0]])
                target_pos[2] = target_pos[2] + 0.3
                self.output["target_pos"] = target_pos
                self.output["target_vel"] = np.array([self.flip_vel[0](eval_time)[0], 0, self.flip_vel[1](eval_time)[0]])
                self.output["target_acc"] = np.array([self.flip_acc[0](eval_time)[0], 0, self.flip_acc[1](eval_time)[0]])
                q0 = self.flip_traj[2](eval_time)[0]
                q2 = np.sqrt(1 - q0**2)
                self.output["target_quat"] = np.array([q0, 0, q2, 0])
                dq0 = self.flip_vel[2](eval_time)[0]
                dq2 = - dq0 * q0 / q2
                self.output["target_quat_vel"] = np.array([dq0, 0, dq2, 0])
                self.output["controller_name"] = "geom_att"

            else:
                self.output["target_pos"] = np.array([0, 0, 0.3])
        
        elif self.scenario == HOOK_UP:
            pass
            
        return self.output
