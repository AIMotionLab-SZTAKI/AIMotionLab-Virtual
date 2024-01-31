import mujoco
import glfw
import numpy as np
from aiml_virtual.util import mujoco_helper, sync
from aiml_virtual.simulator.mujoco_display import Display, INIT_WWIDTH, INIT_WHEIGHT, OPTITRACK_IP
from aiml_virtual.object.moving_object import MocapObject, MovingObject
from aiml_virtual.object.object_parser import parseMovingObjects, parseMocapObjects
from aiml_virtual.gui.vehicle_name_gui import VehicleNameGui
import os
if os.name == 'nt':
    import win_precise_time as time
else:
    import time


class ActiveSimulator(Display):

    def __init__(self, xml_file_name, video_intervals, control_step, graphics_step,
                 virt_parsers: list = [parseMovingObjects], mocap_parsers: list = [parseMocapObjects],
                 connect_to_optitrack=False, window_size=[INIT_WWIDTH, INIT_WHEIGHT],
                 optitrack_ip=OPTITRACK_IP, with_graphics=True):
        
        self._with_graphics = with_graphics
        self.control_step = control_step

        if with_graphics:
            super().__init__(xml_file_name, graphics_step, connect_to_optitrack, window_size)
            self.video_intervals = ActiveSimulator.__check_video_intervals(video_intervals)
            self.is_recording_automatically = False
            self.frame_counter = 0

            self.graphics_control_ratio = int(self.graphics_step / self.control_step)

            self.n_controlstep_sum = int(1 / control_step)
            self.actual_controlstep = self.control_step
            self.n_graphicstep_sum = int(1 / graphics_step)
            self.actual_graphicstep = self.graphics_step

            self.start_time = 0.0
            self.vid_rec_cntr = 0
            fc_target = 1.0 / control_step
            self.control_freq_warning_limit = fc_target - (0.05 * fc_target)
            self.pause_time = 0.0

            self._first_loop = True

            self.set_key_n_callback(self.set_vehicle_names)

        else:
            
            self.load_model(xml_file_name)

        self.virt_parsers = virt_parsers
        self.mocap_parsers = mocap_parsers
        

        self.i = 0
        self.time = self.data.time

        self.frame = None
        self._show_overlay = True
        self.parse_model()
    
    def parse_model(self):
        
        self.all_moving_objects = []
        self.all_mocap_objects = []


        if self.virt_parsers is not None:

            for i in range(len(self.virt_parsers)):
                self.all_moving_objects += self.virt_parsers[i](self.data, self.model)
        
        if self.mocap_parsers is not None:

            for i in range(len(self.mocap_parsers)):
                self.all_mocap_objects += self.mocap_parsers[i](self.data, self.model)

        print()
        print(str(len(self.all_moving_objects)) + " virtual object(s) found in xml.")
        print()
        print(str(len(self.all_mocap_objects)) + " mocap objects(s) found in xml.")
        print("______________________________")
        
        self.all_vehicles = self.all_moving_objects + self.all_mocap_objects
        
    def get_all_MovingObjects(self):
        return self.all_moving_objects
    
    def show_overlay(self, show):

        self._show_overlay = show


    def get_MovingObject_by_name_in_xml(self, name) -> MovingObject:

        for i in range(len(self.all_moving_objects)):

            vehicle = self.all_moving_objects[i]
            if name == vehicle.name_in_xml:

                return vehicle

        return None

    def get_MocapObject_by_name_in_xml(self, name):

        for i in range(len(self.all_mocap_objects)):

            vehicle = self.all_mocap_objects[i]
            if name == vehicle.name_in_xml:
                return vehicle
        
        return None
    
    def reload_model(self, xml_file_name, vehicle_names_in_motive = None):
        
        self.load_model(xml_file_name)

        self.parse_model()

        if vehicle_names_in_motive is not None:
            for i in range(len(self.all_mocap_objects)):
                self.all_mocap_objects[i].name_in_motive = vehicle_names_in_motive[i]
    
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

    def update(self):

        if self._with_graphics:

            if self._first_loop:
                self.start_time = time.time()
                self.prev_tc = time.time()
                self.prev_tg = time.time()
                self._first_loop = False
            
            self.manage_video_recording(self.i)
            
            # getting data from optitrack server
            if self.connect_to_optitrack:

                self.mc.waitForNextFrame()
                for name, obj in self.mc.rigidBodies.items():

                    # have to put rotation.w to the front because the order is different
                    # only update real vehicles
                    vehicle_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]
    
                    vehicle_to_update = MocapObject.get_object_by_name_in_motive(self.all_mocap_objects, name)

                    if vehicle_to_update is not None:
                        vehicle_to_update.update(obj.position, vehicle_orientation)

            if self.activeCam == self.camOnBoard and len(self.all_vehicles) > 0:
                v = self.all_vehicles[self.followed_vehicle_idx]
                mujoco_helper.update_onboard_cam(v.get_qpos(), self.camOnBoard,\
                                                self.azim_filter_sin, self.azim_filter_cos,\
                                                self.elev_filter_sin, self.elev_filter_cos, self.onBoard_elev_offset)
                

            
            if self.i % self.graphics_control_ratio == 0:

                self.frame_counter += 1
                if self.frame_counter % self.n_graphicstep_sum == 0:
                    self.actual_graphicstep = self.calc_actual_graphics_step()  
            
                fc = 1.0 / self.actual_controlstep
                fg = 1.0 / self.actual_graphicstep
                
                if not self._is_paused:
                    fst = "Control: {:10.3f} Hz\nGraphics: {:10.3f} Hz".format(fc, fg)
                else:
                    fst = "Control: ------ Hz\nGraphics: {:10.3f} Hz".format(fg)

                
                if self._show_overlay:
                    needs_warning = (fc < self.control_freq_warning_limit and not self._is_paused)
                    self.render(fst, needs_warning)
                else:
                    self.render()

            if not self._is_paused:
                
                for l in range(len(self.all_moving_objects)):

                    self.all_moving_objects[l].update(self.i, self.control_step)
                mujoco.mj_step(self.model, self.data, int(self.control_step / self.sim_step))
                if self.i % self.n_controlstep_sum == 0:
                    self.actual_controlstep = self.calc_actual_control_step()   
                self.i += 1
            
            else:
                self.tc = time.time()
                self.tg = time.time()
            
            sync(self.i, self.start_time, self.pause_time, self.control_step)

        else:
            # no window, no graphics
            self.update_()
        
        self.time = self.data.time

    def update_(self):
        if self.i == 0:
            self.start_time = time.time()
        

        for l in range(len(self.all_moving_objects)):

            self.all_moving_objects[l].update(self.i, self.control_step)
        
        mujoco.mj_step(self.model, self.data, int(self.control_step / self.sim_step))
        self.i += 1

    def get_frame(self):
        rgb = np.empty((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
        depth = np.empty((self.viewport.height, self.viewport.width, 1))

        stamp = self.data.time
        mujoco.mjr_readPixels(rgb, depth, self.viewport, self.con)
        return rgb, stamp
    
    def manage_video_recording(self, i):
        time_since_start = i * self.control_step

        if self.video_intervals is not None and self.vid_rec_cntr < len(self.video_intervals):
            if time_since_start >= self.video_intervals[self.vid_rec_cntr + 1] and self.is_recording and self.is_recording_automatically:
                self.is_recording = False
                self.is_recording_automatically = False
                self.vid_rec_cntr += 2
                self.reset_title()
                self.save_video_background()

            elif time_since_start >= self.video_intervals[self.vid_rec_cntr] and time_since_start < self.video_intervals[self.vid_rec_cntr + 1]:
                if not self.is_recording_automatically:
                    self.is_recording = True
                    self.is_recording_automatically = True
                    self.append_title(" (Recording automatically...)")


    def calc_actual_control_step(self):
        self.tc = time.time()
        time_elapsed =  self.tc - self.prev_tc
        self.prev_tc = self.tc
        return time_elapsed / self.n_controlstep_sum
    
    def calc_actual_graphics_step(self):
        self.tg = time.time()
        time_elapsed = self.tg - self.prev_tg
        self.prev_tg = self.tg
        return time_elapsed / self.n_graphicstep_sum
    
    def pause(self):
        if self._with_graphics:
            if not self._is_paused:
                self.t_lastpaused = time.time()
                self._is_paused = True


    def unpause(self):
        if self._with_graphics:
            if self._is_paused:
                self.pause_time += time.time() - self.t_lastpaused
                self._is_paused = False

    
    def is_paused(self):
        if self._with_graphics:
            return self._is_paused
    

    def set_vehicle_names(self):
        
        if len(self.all_mocap_objects) > 0:
            object_names = MocapObject.get_object_names_motive(self.all_mocap_objects)
            object_labels = MocapObject.get_object_names_in_xml(self.all_mocap_objects)
            self.pause()
            gui = VehicleNameGui(vehicle_labels=object_labels, vehicle_names=object_names)
            gui.show()
            MocapObject.set_object_names_motive(self.all_mocap_objects, gui.vehicle_names)
            self.unpause()
    
    def should_close(self, end_time=float("inf")):
        """
        end_time: in seconds
        """

        if self._with_graphics:
            return self.glfw_window_should_close() or self.time >= end_time
        else:
            return self.time >= end_time
        
    
    def reset_data(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.i = 0
        self.pause_time = 0.0
        self._first_loop = True
        self.frame_counter = 0

    def close(self):
        
        glfw.terminate()