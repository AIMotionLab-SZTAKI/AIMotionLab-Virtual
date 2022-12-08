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

    def __init__(self, xml_file_name, record_video, sim_step, control_step, graphics_step, connect_to_optitrack=True):

        super().__init__(xml_file_name, connect_to_optitrack)

        self.record_video = record_video
        self.sim_step = sim_step
        self.control_step = control_step
        self.graphics_step = graphics_step

        self.start = 0
    
    def update(self, i):
        
        # getting data from optitrack server
        if self.connect_to_optitrack:

            self.mc.waitForNextFrame()
            for name, obj in self.mc.rigidBodies.items():

                # have to put rotation.w to the front because the order is different
                # only update real drones
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

                try:
                    idx = self.droneNames.index(name)
                except ValueError:
                    idx = -1

                if idx >= 0:
                    mujoco_helper.update_drone(self.data, idx, obj.position, drone_orientation)

        if self.activeCam == self.camFollow and len(self.drones) > 0:
            mujoco_helper.update_follow_cam(self.drones[self.followed_drone_idx].get_qpos(), self.camFollow,\
                                            self.azim_filter_sin, self.azim_filter_cos,\
                                            self.elev_filter_sin, self.elev_filter_cos)


        for l in range(len(self.drones)):

            self.spin_propellers(self.drones[l])
            
            self.drones[l].trajectories.evaluate(i, self.drones[l])
            # self.drones[l].controller.update()

            # self.drones[l].trajectories.eval()

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
            sync(i, self.start, self.control_step)

        return self.data
    
    def log(self):
        pass

    def plot_log(self):
        pass

    def save_log(self):
        pass

    def close(self):
        
        glfw.terminate()