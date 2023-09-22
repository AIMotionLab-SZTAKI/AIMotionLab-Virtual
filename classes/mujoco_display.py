import math
import os
import sys

import time
import numpy as np
import mujoco
import glfw
import numpy as np
import time
import threading
from util import mujoco_helper
from gui.vehicle_name_gui import VehicleNameGui
import scipy.signal
from util.mujoco_helper import LiveLFilter
from classes.moving_object import MocapObject, MovingObject
import ffmpeg
import motioncapture

MAX_GEOM = 200
INIT_WWIDTH = 1536
INIT_WHEIGHT = 864

OPTITRACK_IP = "192.168.2.141"

class Display:
    """ Base class for passive and active simulation
    """

    def __init__(self, xml_file_name, graphics_step, virt_parsers: list = None, mocap_parsers: list = None, connect_to_optitrack=False):
        #print(f'Working directory:  {os.getcwd()}\n')
        
        self.is_paused = False

        self.virt_parsers = virt_parsers
        self.mocap_parsers = mocap_parsers

        self.key_b_callback = None
        self.key_d_callback = None
        self.key_l_callback = None
        self.key_o_callback = None
        self.key_t_callback = None
        self.key_v_callback = None
        self.key_delete_callback = None
        self.key_left_callback = None
        self.key_left_release_callback = None
        self.key_right_callback = None
        self.key_right_release_callback = None
        self.key_up_callback = None
        self.key_up_release_callback = None
        self.key_down_callback = None
        self.key_down_release_callback = None

        self.connect_to_optitrack = connect_to_optitrack

        self.mouse_left_btn_down = False
        self.mouse_right_btn_down = False
        self.prev_x, self.prev_y = 0.0, 0.0
        self.followed_vehicle_idx = 0
        self.title0 = "Scene"
        self.is_recording = False
        self.image_list = []
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.video_save_folder = os.path.join(self.abs_path, "..", "video_capture")
        self.video_file_name_base = "output"

        self.graphics_step = graphics_step

        fps = 1.0 / self.graphics_step

        self.init_glfw()
        self.init_cams()
        self.load_model(xml_file_name)

        
        # Connect to optitrack
        if connect_to_optitrack:
            self.mc = motioncapture.MotionCaptureOptitrack(OPTITRACK_IP)
            print("[Display] Connected to Optitrack")

        self.t1 = time.time()


        self.pert = mujoco.MjvPerturb()
        self.opt = mujoco.MjvOption()
        #self.scn = mujoco.MjvScene(self.model, maxgeom=MAX_GEOM)
        #self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        self.viewport = mujoco.MjrRect(0, 0, 0, 0)
        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)

        self.scroll_distance_step = 0.2

        #print(self.data.qpos.size)
        
    def init_glfw(self):
        # Initialize the library
        if not glfw.init():
            print("Could not initialize glfw...")
            return

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(INIT_WWIDTH, INIT_WHEIGHT, self.title0, None, None)
        if not self.window:
            print("Could not create glfw window...")
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)
        # setup mouse callbacks
        glfw.set_scroll_callback(self.window, self.zoom)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_key_callback(self.window, self.key_callback)
    
    def init_cams(self):

        self.cam = mujoco.MjvCamera()
        self.camOnBoard = mujoco.MjvCamera()
        self.activeCam = self.cam
        
        self.cam.azimuth, self.cam.elevation = 180, -20
        self.cam.lookat, self.cam.distance = [0, 0, .5], 5
        self.camOnBoard.azimuth, self.camOnBoard.elevation = 180, -30
        self.camOnBoard.lookat, self.camOnBoard.distance = [0, 0, 0], 1

        # set up low-pass filters for the camera that follows the vehicles
        fs = 1 / self.graphics_step  # sampling rate, Hz
        cutoff = 4
        self.b, self.a = scipy.signal.iirfilter(4, Wn=cutoff, fs=fs, btype="low", ftype="butter")
        self.azim_filter_sin = LiveLFilter(self.b, self.a)
        self.azim_filter_cos = LiveLFilter(self.b, self.a)
        fs = 1 / self.graphics_step  # sampling rate, Hz
        cutoff = 0.5
        self.b, self.a = scipy.signal.iirfilter(4, Wn=cutoff, fs=fs, btype="low", ftype="butter")
        self.elev_filter_sin = LiveLFilter(self.b, self.a)
        self.elev_filter_cos = LiveLFilter(self.b, self.a)

        self.onBoard_elev_offset = 30

    def set_cam_position(self, azimuth: float, elevation: float, lookat: list, distance: float):
        """
        Sets the position of the main camera
        azimuth: rotation about the vertical axis (yaw) in degrees
        elevation: rotation about the horizontal axis perpendicular to the view direction (pitch) in degrees
        lookat: 3D point for the camera to look at (a list of 3 floats)
        distance: the distance from the lookat point
        """
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.lookat = lookat
        self.cam.distance = distance

    
    def load_model(self, xml_file_name):
        self.xmlFileName = xml_file_name

        self.model = mujoco.MjModel.from_xml_path(self.xmlFileName)
        self.data = mujoco.MjData(self.model)

        self.gravity = self.model.opt.gravity

        self.scn = mujoco.MjvScene(self.model, maxgeom=MAX_GEOM)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        self.sim_step = self.model.opt.timestep

        self.all_virt_vehicles = []
        self.all_real_vehicles = []


        if self.virt_parsers is not None:

            for i in range(len(self.virt_parsers)):
                self.all_virt_vehicles += self.virt_parsers[i](self.data, self.model)
        
        if self.mocap_parsers is not None:

            for i in range(len(self.mocap_parsers)):
                self.all_real_vehicles += self.mocap_parsers[i](self.data, self.model)

        print()
        print(str(len(self.all_virt_vehicles)) + " virtual object(s) found in xml.")
        print()
        print(str(len(self.all_real_vehicles)) + " mocap objects(s) found in xml.")
        print("______________________________")
        
        self.all_vehicles = self.all_virt_vehicles + self.all_real_vehicles
        
        # to obtain inertia matrix
        mujoco.mj_step(self.model, self.data)
    

    def get_MovingObject_by_name_in_xml(self, name) -> MovingObject:

        for i in range(len(self.all_virt_vehicles)):

            vehicle = self.all_virt_vehicles[i]
            if name == vehicle.name_in_xml:

                return vehicle

        return None

    def get_MocapObject_by_name_in_xml(self, name):

        for i in range(len(self.all_real_vehicles)):

            vehicle = self.all_real_vehicles[i]
            if name == vehicle.name_in_xml:
                return vehicle
        
        return None
    
    def reload_model(self, xml_file_name, vehicle_names_in_motive = None):
        
        self.load_model(xml_file_name)

        if vehicle_names_in_motive is not None:
            for i in range(len(self.all_real_vehicles)):
                self.all_real_vehicles[i].name_in_motive = vehicle_names_in_motive[i]
        

    def glfw_window_should_close(self):
        return glfw.window_should_close(self.window)

    def set_key_b_callback(self, callback_function):
        if callable(callback_function):
            self.key_b_callback = callback_function

    def set_key_d_callback(self, callback_function):
        if callable(callback_function):
            self.key_d_callback = callback_function

    def set_key_l_callback(self, callback_function):
        if callable(callback_function):
            self.key_l_callback = callback_function
    
    def set_key_o_callback(self, callback_function):
        if callable(callback_function):
            self.key_o_callback = callback_function

    def set_key_t_callback(self, callback_function):
        if callable(callback_function):
            self.key_t_callback = callback_function

    def set_key_v_callback(self, callback_function):
        if callable(callback_function):
            self.key_v_callback = callback_function
    
    def set_key_delete_callback(self, callback_function):
        if callable(callback_function):
            self.key_delete_callback = callback_function


    def set_key_left_callback(self, callback_function):
        if callable(callback_function):
            self.key_left_callback = callback_function
    def set_key_right_callback(self, callback_function):
        if callable(callback_function):
            self.key_right_callback = callback_function
    def set_key_up_callback(self, callback_function):
        if callable(callback_function):
            self.key_up_callback = callback_function
    def set_key_down_callback(self, callback_function):
        if callable(callback_function):
            self.key_down_callback = callback_function

    def set_key_left_release_callback(self, callback_function):
        if callable(callback_function):
            self.key_left_release_callback = callback_function
    def set_key_right_release_callback(self, callback_function):
        if callable(callback_function):
            self.key_right_release_callback = callback_function
    def set_key_up_release_callback(self, callback_function):
        if callable(callback_function):
            self.key_up_release_callback = callback_function
    def set_key_down_release_callback(self, callback_function):
        if callable(callback_function):
            self.key_down_release_callback = callback_function

    def mouse_button_callback(self, window, button, action, mods):

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.prev_x, self.prev_y = glfw.get_cursor_pos(window)
            self.mouse_left_btn_down = True

        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self.mouse_left_btn_down = False

        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            self.prev_x, self.prev_y = glfw.get_cursor_pos(window)
            self.mouse_right_btn_down = True

        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
            self.mouse_right_btn_down = False

    def mouse_move_callback(self, window, xpos, ypos):
        if self.activeCam != self.cam:
            return

        if self.mouse_left_btn_down:
            """
            Rotate camera about the lookat point
            """
            dx, dy = self.calc_dxdy(window)
            scale = 0.1
            self.cam.azimuth -= dx * scale
            self.cam.elevation -= dy * scale

        if self.mouse_right_btn_down:
            """
            Move the point that the camera is looking at
            """
            dx, dy = self.calc_dxdy(window)
            scale = 0.005
            angle = math.radians(self.cam.azimuth + 90)
            dx3d = math.cos(angle) * dx * scale
            dy3d = math.sin(angle) * dx * scale

            self.cam.lookat[0] += dx3d
            self.cam.lookat[1] += dy3d

            # vertical axis is Z in 3D, so 3rd element in the lookat array
            self.cam.lookat[2] += dy * scale

    def calc_dxdy(self, window):
        """
        Calculate cursor displacement
        """
        x, y = glfw.get_cursor_pos(window)
        dx = x - self.prev_x
        dy = y - self.prev_y
        self.prev_x, self.prev_y = x, y

        return dx, dy

    def zoom(self, window, x, y):
        """
        Change distance between camera and lookat point by mouse wheel
        """
        self.activeCam.distance -= self.scroll_distance_step * y

    def key_callback(self, window, key, scancode, action, mods):
        """
        Switch camera on TAB press
        Switch among vehicles on SPACE press if camera is set to follow vehicle
        """
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            self.change_cam()
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            if self.activeCam == self.camOnBoard:
                if self.followed_vehicle_idx + 1 == len(self.all_vehicles):
                    self.followed_vehicle_idx = 0
                else:
                    self.followed_vehicle_idx += 1
                self.azim_filter = LiveLFilter(self.b, self.a)
                self.elev_filter = LiveLFilter(self.b, self.a)
                d = self.all_vehicles[self.followed_vehicle_idx]
                mujoco_helper.update_onboard_cam(d.get_qpos(), self.camOnBoard)
        
        if key == glfw.KEY_B and action == glfw.RELEASE:
            """
            Pass on this event
            """
            if self.key_b_callback:
                self.key_b_callback()

        if key == glfw.KEY_D and action == glfw.RELEASE:
            """
            Pass on this event
            """
            if self.key_d_callback:
                self.key_d_callback()

        if key == glfw.KEY_R and action == glfw.RELEASE:
            """
            Start recording
            """
            if not self.is_recording:
                
                self.append_title(" (Recording)")
                self.is_recording = True
            else:
                self.reset_title()
                self.is_recording = False
                #self.flush_video_process()
                self.save_video_background()

        if key == glfw.KEY_C and action == glfw.RELEASE:
            self.connect_to_Optitrack()

        if key == glfw.KEY_N and action == glfw.RELEASE:
            self.set_vehicle_names()

        if key == glfw.KEY_L and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_l_callback:
                self.key_l_callback()
        
        if key == glfw.KEY_O and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_o_callback:
                self.key_o_callback()
        
        if key == glfw.KEY_P and action == glfw.RELEASE:
            """
            pause simulation
            """
            self.is_paused = not self.is_paused
        
        if key == glfw.KEY_T and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_t_callback:
                self.key_t_callback()
        
        
        if key == glfw.KEY_V and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_v_callback:
                self.key_v_callback()
        
        if key == glfw.KEY_DELETE and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_delete_callback:
                self.key_delete_callback()
        
        if key == glfw.KEY_LEFT and action == glfw.PRESS:
            """
            pass on this event
            """
            if self.key_left_callback:
                self.key_left_callback()
        if key == glfw.KEY_RIGHT and action == glfw.PRESS:
            """
            pass on this event
            """
            if self.key_right_callback:
                self.key_right_callback()
        if key == glfw.KEY_UP and action == glfw.PRESS:
            """
            pass on this event
            """
            if self.key_up_callback:
                self.key_up_callback()
        if key == glfw.KEY_DOWN and action == glfw.PRESS:
            """
            pass on this event
            """
            if self.key_down_callback:
                self.key_down_callback()
        

        if key == glfw.KEY_LEFT and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_left_release_callback:
                self.key_left_release_callback()
        if key == glfw.KEY_RIGHT and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_right_release_callback:
                self.key_right_release_callback()
        if key == glfw.KEY_UP and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_up_release_callback:
                self.key_up_release_callback()
        if key == glfw.KEY_DOWN and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_down_release_callback:
                self.key_down_release_callback()

    
    def set_title(self, window_title):
        self.title0 = window_title
        glfw.set_window_title(self.window, window_title)
    
    def append_title(self, text):
        glfw.set_window_title(self.window, self.title0 + text)
    
    def reset_title(self):
        glfw.set_window_title(self.window, self.title0)


    def connect_to_Optitrack(self):
        self.connect_to_optitrack = True
        self.mc = motioncapture.MotionCaptureOptitrack(OPTITRACK_IP)
        print("[Display] Connected to Optitrack")   


    def change_cam(self):
        """
        Change camera between scene cam and 'on board' cam
        """
        if self.activeCam == self.cam and len(self.all_vehicles) > 0:
            self.activeCam = self.camOnBoard
        else:
            self.activeCam = self.cam
    
    def append_frame_to_list(self):

                 
        # need to create arrays with the exact size!! before passing them to mjr_readPixels()
        rgb = np.empty(self.viewport.width * self.viewport.height * 3, dtype=np.uint8)
        depth = np.zeros(self.viewport.height * self.viewport.width)

        # draw a time stamp on the rendered image
        stamp = str(time.time())
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, self.viewport, stamp, None, self.con)
        
        mujoco.mjr_readPixels(rgb, depth, self.viewport, self.con)
        
        self.image_list.append([stamp, rgb])


    def save_video(self, image_list, width, height):
        """
        Write saved images to hard disk as .mp4
        """
        print("[Display] Saving video...")
        # checking for folder
        if not os.path.exists(self.video_save_folder):
            # then create folder
            os.mkdir(self.video_save_folder)


        time_stamp = time.time()
        filename = os.path.join(self.video_save_folder, self.video_file_name_base + '_' + str(time_stamp) + '.mp4')

        fps = 1.0 / self.graphics_step
        video_process = (
                        ffmpeg
                        .input('pipe:', vsync=0, format='rawvideo', pix_fmt='rgb24', s=f'{self.viewport.width}x{self.viewport.height}', r=f'{fps}')
                        .vflip()
                        .output(filename)
                        .overwrite_output()
                        .run_async(pipe_stdin=True, overwrite_output=True)
                    )

        fps = 1 / self.graphics_step
        #print("fps: " + str(fps))
        self.append_title(" (Saving video...)")
        time_stamp = image_list[0][0].replace('.', '_')

        #out = cv2.VideoWriter(os.path.join(self.video_save_folder, self.video_file_name_base + '_' + time_stamp + '.mp4'),\
        #      cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for i in range(len(image_list)):
            
            rgb = np.reshape(image_list[i][1], (height, width, 3))
            #rgb = np.flip(rgb, 0)
            video_process.stdin.write(rgb.tobytes())
            #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            #out.write(rgb)
        #out.release()
        # Close and flush stdin
        video_process.stdin.close()
        # Wait for sub-process to finish
        video_process.wait()
        print("[Display] Saved video in " + os.path.normpath(self.video_save_folder))
        self.reset_title()

    def save_video_background(self):
        save_vid_thread = threading.Thread(target=self.save_video, args=(self.image_list.copy(), self.viewport.width, self.viewport.height))
        self.image_list = []
        save_vid_thread.start()

    def flush_video_process(self):
        # Close and flush stdin
        self.video_process.stdin.close()
        # Wait for sub-process to finish
        self.video_process.wait()


    def set_vehicle_names(self):
        
        if len(self.all_real_vehicles) > 0:
            object_names = MocapObject.get_object_names_motive(self.all_real_vehicles)
            object_labels = MocapObject.get_object_names_in_xml(self.all_real_vehicles)
            gui = VehicleNameGui(vehicle_labels=object_labels, vehicle_names=object_names)
            gui.show()
            MocapObject.set_object_names_motive(self.all_real_vehicles, gui.vehicle_names)

