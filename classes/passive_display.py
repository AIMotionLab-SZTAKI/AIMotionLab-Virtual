from ast import Pass
import math

import time
import numpy as np
import mujoco
import glfw
import os
import numpy as np
import time
from util import mujoco_helper
import cv2
from gui.vehicle_name_gui import VehicleNameGui
from util.util import sync, FpsLimiter
import scipy.signal
from util.mujoco_helper import LiveLFilter
from classes.mujoco_display import Display
from classes.drone import Drone, DroneMocap
from classes.moving_object import MovingMocapObject


class PassiveDisplay(Display):

    def __init__(self, xml_file_name, graphics_step, virt_parsers: list = None, mocap_parsers: list = None, connect_to_optitrack=True):

        super().__init__(xml_file_name, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack)

    def run(self):
        # To obtain inertia matrix
        mujoco.mj_step(self.model, self.data)
    
        
        self.fps_limiter = FpsLimiter(1.0 / self.graphics_step)
        
        while not self.glfw_window_should_close():
            self.fps_limiter.begin_frame()
            
            # getting data from optitrack server
            if self.connect_to_optitrack:

                self.mc.waitForNextFrame()
                for name, obj in self.mc.rigidBodies.items():

                    # have to put rotation.w to the front because the order is different
                    vehicle_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

                    vehicle_to_update = MovingMocapObject.get_object_by_name_in_motive(self.all_real_vehicles, name)
                    #print(name)

                    if vehicle_to_update:
                        vehicle_to_update.update(obj.position, vehicle_orientation)

            if self.activeCam == self.camOnBoard and len(self.all_vehicles) > 0:
                mujoco_helper.update_onboard_cam(self.all_vehicles[self.followed_vehicle_idx].get_qpos(), self.camOnBoard,\
                                               self.azim_filter_sin, self.azim_filter_cos,\
                                               self.elev_filter_sin, self.elev_filter_cos, self.onBoard_elev_offset)


            mujoco.mj_step(self.model, self.data, 1)
            self.viewport = mujoco.MjrRect(0, 0, 0, 0)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, pert=None, cam=self.activeCam, catmask=mujoco.mjtCatBit.mjCAT_ALL,
                                   scn=self.scn)
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            if self.is_recording:
                self.append_frame_to_list()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            
            self.fps_limiter.end_frame()
        
        if self.is_recording:
            self.save_video()

        glfw.terminate()
	
    def print_optitrack_data(self):

        self.mc.waitForNextFrame()
        for name, obj in self.mc.rigidBodies.items():

            # have to put rotation.w to the front because the order is different
            drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

            print(name + str(obj.position) + " " + str(drone_orientation))
            print()