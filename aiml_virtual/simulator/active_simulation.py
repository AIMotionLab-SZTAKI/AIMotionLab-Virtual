import mujoco
import glfw
from aiml_virtual.util import mujoco_helper
from aiml_virtual.util.util import sync
from aiml_virtual.simulator.mujoco_display import Display, INIT_WWIDTH, INIT_WHEIGHT
from aiml_virtual.object.moving_object import MocapObject
from aiml_virtual.object.object_parser import parseMovingObjects, parseMocapObjects
import os
if os.name == 'nt':
    import win_precise_time as time
else:
    import time


class ActiveSimulator(Display):

    def __init__(self, xml_file_name, video_intervals, control_step, graphics_step,
                 virt_parsers: list = [parseMovingObjects], mocap_parsers: list = [parseMocapObjects], connect_to_optitrack=False, window_size=[INIT_WWIDTH, INIT_WHEIGHT]):

        super().__init__(xml_file_name, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack, window_size)
        self.video_intervals = ActiveSimulator.__check_video_intervals(video_intervals)

        #self.sim_step = sim_step
        self.control_step = control_step
        self.graphics_step = graphics_step
        self.is_recording_automatically = False

        # To obtain inertia matrix
        self.start_time = 0.0
        self.vid_rec_cntr = 0

        self.i = 0
        self.frame_counter = 0

        self.n_controlstep_sum = int(1 / control_step)
        self.actual_controlstep = self.control_step
        self.n_graphicstep_sum = int(1 / graphics_step)
        self.actual_graphicstep = self.graphics_step

        fc_target = 1.0 / control_step
        self.control_freq_warning_limit = fc_target - (0.05 * fc_target)
    
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

        if self.i == 0:
            self.start_time = time.time()
            self.prev_tc = time.time()
            self.prev_tg = time.time()
        
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
            

        for l in range(len(self.all_moving_objects)):

            self.all_moving_objects[l].update(self.i, self.control_step)
        
        if self.i % (self.graphics_step / self.control_step) == 0:

            self.viewport = mujoco.MjrRect(0, 0, 0, 0)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, pert=None, cam=self.activeCam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=self.scn)
            mujoco.mjr_render(self.viewport, self.scn, self.con)
            
            self.frame_counter += 1
            if self.frame_counter % self.n_graphicstep_sum == 0:
                self.actual_graphicstep = self.calc_actual_graphics_step()  
        
            fc = 1.0 / self.actual_controlstep
            fg = 1.0 / self.actual_graphicstep
            fst = "Control: {:10.3f} Hz\nGraphics: {:10.3f} Hz".format(fc, fg)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, self.viewport, fst, None, self.con)

            if fc < self.control_freq_warning_limit:
                mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, "Control frequency below target", self.con, 1.0, .4, 1.0, 0.1, 0.1)

            if self.is_recording:
                 
                self.append_frame_to_list()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        sync(self.i, self.start_time, self.control_step)

        if self.i % self.n_controlstep_sum == 0:
            self.actual_controlstep = self.calc_actual_control_step()
        
        
        if not self.is_paused:
            mujoco.mj_step(self.model, self.data, int(self.control_step / self.sim_step))
            self.i += 1
        
        return self.data

    def update_(self):
        if self.i == 0:
            self.start_time = time.time()
        

        for l in range(len(self.all_moving_objects)):

            self.all_moving_objects[l].update(self.i, self.control_step)
        
        mujoco.mj_step(self.model, self.data, int(self.control_step / self.sim_step))
        self.i += 1

    
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