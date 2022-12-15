from ast import Pass
import math

import motioncapture
import time
import numpy as np
import mujoco
import glfw
import os
import numpy as np
import time
from util import mujoco_helper
import cv2
from gui.drone_name_gui import DroneNameGui
from util.util import sync
import scipy.signal
from util.mujoco_helper import LiveLFilter
from classes.mujoco_display import Display
import classes.drone as drone


class ActiveSimulator(Display):

    def __init__(self, xml_file_name, video_intervals, sim_step, control_step, graphics_step, connect_to_optitrack=True):

        super().__init__(xml_file_name, connect_to_optitrack)
        """
        
        Inputs:
            xml_file_name:
            video_intervals: should have even number of elements, 
        """
        self.video_intervals = ActiveSimulator.__check_video_intervals(video_intervals)

        self.sim_step = sim_step
        self.control_step = control_step
        self.graphics_step = graphics_step
        self.is_automatic_recording = False

        ## To obtain inertia matrix
        mujoco.mj_step(self.model, self.data)
        self.start_time = 0.0
        self.prev_time = time.time()
        self.vid_rec_cntr = 0
    
    @staticmethod
    def __check_video_intervals(video_intervals):

        if video_intervals is not None:
            if not isinstance(video_intervals, list):
                print("[ActiveSimulator] Error: video_intervals should be list")
                return None

            else:
                checked_video_intervals = []
                i = 0
                while i + 1 < len(video_intervals):
                    if video_intervals[i + 1] <= video_intervals[i]:
                        print("[ActiveSimulator] Error: end of video interval needs to be greater than its start. Excluding this interval.")
                    
                    else:
                        checked_video_intervals.append(video_intervals[i])
                        checked_video_intervals.append(video_intervals[i + 1])
                    
                    i += 2
                
                return checked_video_intervals

        
        return None

    
    def update(self, i):
        
        self.manage_video_recording()
        
        # getting data from optitrack server
        if self.connect_to_optitrack:

            self.mc.waitForNextFrame()
            for name, obj in self.mc.rigidBodies.items():

                # have to put rotation.w to the front because the order is different
                # only update real drones
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]
 
                drone_to_update = drone.Drone.get_drone_by_name_in_motive(self.realdrones, name)

                if drone_to_update is not None:
                    drone_to_update.set_pos(obj.position)
                    drone_to_update.set_quat(drone_orientation)

        if self.activeCam == self.camFollow and len(self.drones) > 0:
            d = self.drones[self.followed_drone_idx]
            mujoco_helper.update_follow_cam(d.get_qpos(), self.camFollow,\
                                            self.azim_filter_sin, self.azim_filter_cos,\
                                            self.elev_filter_sin, self.elev_filter_cos)
        
        for m in range(len(self.realdrones)):
            self.realdrones[m].spin_propellers()


        for l in range(len(self.virtdrones)):

            #self.spin_propellers(self.drones[l], self.control_step, 20)
            self.virtdrones[l].fake_propeller_spin(self.control_step, 20)


            self.virtdrones[l].update(i)

            # self.drones[l].controller.update()

            # self.drones[l].trajectory.eval()

            # self.data.ctrl[valami] = drones[l].controller.eval()
            #pass
        
        mujoco.mj_step(self.model, self.data, int(self.control_step / self.sim_step))

        if i % (self.graphics_step / self.control_step) == 0:


            self.viewport = mujoco.MjrRect(0, 0, 0, 0)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, pert=None, cam=self.activeCam, catmask=mujoco.mjtCatBit.mjCAT_ALL,
                                    scn=self.scn)
            mujoco.mjr_render(self.viewport, self.scn, self.con)
            if self.is_recording:
                 
                # need to create arrays with the exact size!! before passing them to mjr_readPixels()
                rgb = np.empty(self.viewport.width * self.viewport.height * 3, dtype=np.uint8)
                depth = np.empty(self.viewport.width * self.viewport.height, dtype=np.float32)

                # draw a time stamp on the rendered image
                stamp = str(time.time())
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, self.viewport, stamp, None, self.con)
                
                mujoco.mjr_readPixels(rgb, depth, self.viewport, self.con)
                
                self.image_list.append([stamp, rgb])

            glfw.swap_buffers(self.window)
            glfw.poll_events()
        sync(i, self.start_time, self.control_step)

        return self.data
    
    def manage_video_recording(self):
        time_since_start = time.time() - self.start_time

        if self.video_intervals is not None and self.vid_rec_cntr < len(self.video_intervals):
            if time_since_start >= self.video_intervals[self.vid_rec_cntr + 1] and self.is_recording and self.is_automatic_recording:
                self.is_recording = False
                self.is_automatic_recording = False
                self.vid_rec_cntr += 2
                self.reset_title()
                self.save_video_background()

            elif time_since_start >= self.video_intervals[self.vid_rec_cntr] and time_since_start < self.video_intervals[self.vid_rec_cntr + 1]:
                if not self.is_automatic_recording:
                    self.is_recording = True
                    self.is_automatic_recording = True
                    self.append_title(" (Recording automatically...)")

    
    def print_time_diff(self):
        self.tc = time.time()
        time_elapsed =  self.tc - self.prev_time
        self.prev_time = self.tc
        print(f'{time_elapsed:.4f}')

    
    def log(self):
        pass

    def plot_log(self):
        pass

    def save_log(self):
        pass

    def close(self):
        
        glfw.terminate()