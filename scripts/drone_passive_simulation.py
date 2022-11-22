from ast import Pass
import math

#import motioncapture
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
from mujoco_display import Display


class PassiveDisplay(Display):

    def __init__(self, xml_file_name, connect_to_optitrack=True):

        super().__init__(xml_file_name, connect_to_optitrack)


    def run(self):
        # To obtain inertia matrix
        mujoco.mj_step(self.model, self.data)

        i = 0
        start = time.time()
        while not glfw.window_should_close(self.window):
            
            # getting data from optitrack server
            if self.connect_to_optitrack:

                self.mc.waitForNextFrame()
                for name, obj in self.mc.rigidBodies.items():

                    # have to put rotation.w to the front because the order is different
                    drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

                    try:
                        idx = self.droneNames.index(name)
                    except ValueError:
                        idx = -1

                    if idx >= 0:
                        mujoco_helper.update_drone(self.data, idx, obj.position, drone_orientation)

            if self.activeCam == self.camFollow and self.DRONE_NUM > 0:
                mujoco_helper.update_follow_cam(self.data.qpos, self.followed_drone_ID, self.camFollow,\
                                               self.azim_filter_sin, self.azim_filter_cos,\
                                               self.elev_filter_sin, self.elev_filter_cos)

            mujoco.mj_step(self.model, self.data, 1)
            self.viewport = mujoco.MjrRect(0, 0, 0, 0)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, pert=None, cam=self.activeCam, catmask=mujoco.mjtCatBit.mjCAT_ALL,
                                   scn=self.scn)
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            if self.is_recording:
                
                rgb = np.zeros(self.viewport.width * self.viewport.height * 3, dtype=np.uint8)
                depth = np.zeros(self.viewport.width * self.viewport.height, dtype=np.float32)

                stamp = str(time.time())
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, self.viewport, stamp, None, self.con)
                
                mujoco.mjr_readPixels(rgb, depth, self.viewport, self.con)
                
                self.image_list.append([stamp, rgb])

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            
            sync(i, start, self.timestep)
            i += 1
        
        if self.is_recording:
            self.save_video()

        glfw.terminate()