import math

import motioncapture
import time
import numpy as np
import mujoco
import glfw
import os
import numpy as np
import time
import threading
from util import mujoco_helper
import cv2
from gui.drone_name_gui import DroneNameGui
import scipy.signal
from util.mujoco_helper import LiveLFilter
from classes.drone import Drone

MAX_GEOM = 200

class Display:
    """ Base class for passive and active simulation
    """

    def __init__(self, xml_file_name, connect_to_optitrack=True):
        print(f'Working directory:  {os.getcwd()}\n')

        self.key_b_callback = None
        self.key_d_callback = None
        self.key_l_callback = None
        self.key_o_callback = None
        self.key_t_callback = None
        self.key_delete_callback = None

        self.connect_to_optitrack = connect_to_optitrack

        self.mouse_left_btn_down = False
        self.mouse_right_btn_down = False
        self.prev_x, self.prev_y = 0.0, 0.0
        self.followed_drone_idx = 0
        self.title = "Scene"
        self.is_recording = False
        self.image_list = []
        self.video_save_folder = os.path.join("..", "video_capture")
        self.video_file_name_base = "output"

        self.graphics_step = 0.04
        
        
        self.init_glfw()
        self.init_cams()
        self.load_model(xml_file_name)

        # Connect to optitrack
        if connect_to_optitrack:
            self.mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")
            print("[Display] Connected to Optitrack")

        self.t1 = time.time()


        self.pert = mujoco.MjvPerturb()
        self.opt = mujoco.MjvOption()
        #self.scn = mujoco.MjvScene(self.model, maxgeom=MAX_GEOM)
        #self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        self.viewport = mujoco.MjrRect(0, 0, 0, 0)
        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)

        self.virtdrones, self.realdrones = Drone.parse_drones(self.data, mujoco_helper.get_joint_name_list(self.model))
        self.drones = self.virtdrones + self.realdrones
        #print(self.data.qpos.size)
        
    def init_glfw(self):
        # Initialize the library
        if not glfw.init():
            print("Could not initialize glfw...")
            return

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1280, 720, self.title, None, None)
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
        self.camFollow = mujoco.MjvCamera()
        self.activeCam = self.cam
        
        self.cam.azimuth, self.cam.elevation = 70, -20
        self.cam.lookat, self.cam.distance = [0, 0, .5], 3
        self.camFollow.azimuth, self.camFollow.elevation = 180, -30
        self.camFollow.lookat, self.camFollow.distance = [0, 0, 0], 1.8

        # set up low-pass filters for the camera that follows the drones
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
    
    def spin_propellers(self, drone):

            if drone.get_qpos()[2] > 0.10:
                drone.spin_propellers(10 * self.graphics_step)
                #drone.print_prop_angles()
            else:
                #drone.spin_propellers(5 * self.graphics_step)
                drone.stop_propellers()
                pass
                

    
    def load_model(self, xml_file_name):
        self.xmlFileName = xml_file_name

        self.model = mujoco.MjModel.from_xml_path(self.xmlFileName)
        self.data = mujoco.MjData(self.model)

        self.scn = mujoco.MjvScene(self.model, maxgeom=MAX_GEOM)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def glfw_window_should_close(self):
        return glfw.window_should_close(self.window)
    
    def reload_model(self, xml_file_name, drone_names_in_motive = None):
        
        self.load_model(xml_file_name)
        self.virtdrones, self.realdrones = Drone.parse_drones(self.data, mujoco_helper.get_joint_name_list(self.model))
        self.drones = self.virtdrones + self.realdrones

        if drone_names_in_motive is not None and len(drone_names_in_motive) > 0:
            i = 0
            for i in range(len(self.realdrones)):
                self.realdrones[i].name_in_motive = drone_names_in_motive[i]
                i += 1


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
    
    def set_key_delete_callback(self, callback_function):
        if callable(callback_function):
            self.key_delete_callback = callback_function


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
        self.activeCam.distance -= 0.2 * y

    def key_callback(self, window, key, scancode, action, mods):
        """
        Switch camera on TAB press
        Switch among drones on SPACE press if camera is set to follow drones
        """
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            self.change_cam()
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            if self.activeCam == self.camFollow:
                if self.followed_drone_idx + 1 == len(self.drones):
                    self.followed_drone_idx = 0
                else:
                    self.followed_drone_idx += 1
                self.azim_filter = LiveLFilter(self.b, self.a)
                self.elev_filter = LiveLFilter(self.b, self.a)
                mujoco_helper.update_follow_cam(self.drones[self.followed_drone_idx].get_qpos(), self.camFollow)
        
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
                glfw.set_window_title(window, self.title + " (Recording)")
                self.is_recording = True
            else:
                glfw.set_window_title(window, self.title)
                self.is_recording = False
                save_vid_thread = threading.Thread(target=self.save_video, args=(self.viewport.width, self.viewport.height))
                
                save_vid_thread.start()

        if key == glfw.KEY_C and action == glfw.RELEASE:
            self.connect_to_Optitrack()

        if key == glfw.KEY_N and action == glfw.RELEASE:
            self.set_drone_names()

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
        
        if key == glfw.KEY_T and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_t_callback:
                self.key_t_callback()
        
        if key == glfw.KEY_DELETE and action == glfw.RELEASE:
            """
            pass on this event
            """
            if self.key_delete_callback:
                self.key_delete_callback()


    def connect_to_Optitrack(self):
        self.connect_to_optitrack = True
        self.mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")
        print("[Display] Connected to Optitrack")


    def change_cam(self):
        """
        Change camera between scene cam and 'on board' cam
        """
        if self.activeCam == self.cam and len(self.drones) > 0:
            self.activeCam = self.camFollow
        else:
            self.activeCam = self.cam
    

    def save_video(self, width, height):
        """
        Write saved images to hard disk as .mp4
        """
        print("[Display] Saving video...")
        # checking for folder
        if not os.path.exists(self.video_save_folder):
            # then create folder
            os.mkdir(self.video_save_folder)

        fps = 1 / self.graphics_step
        #print("fps: " + str(fps))

        glfw.set_window_title(self.window, self.title + " (Saving video...)")
        time_stamp = self.image_list[0][0].replace('.', '_')
        out = cv2.VideoWriter(os.path.join(self.video_save_folder, self.video_file_name_base + '_' + time_stamp + '.mp4'),\
              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for i in range(len(self.image_list)):
            #print(self.image_list[i][0])
            rgb = np.reshape(self.image_list[i][1], (height, width, 3))
            rgb = cv2.cvtColor(np.flip(rgb, 0), cv2.COLOR_BGR2RGB)
            out.write(rgb)
        out.release()
        self.image_list = []
        print("[Display] Saved video in " + os.path.normpath(os.path.join(os.getcwd(), self.video_save_folder)))
        glfw.set_window_title(self.window, self.title)

    def set_drone_names(self):
        
        if len(self.realdrones) > 0:
            drone_names = Drone.get_drone_names_motive(self.realdrones)
            drone_labels = Drone.get_drone_labels(self.realdrones)
            gui = DroneNameGui(drone_labels=drone_labels, drone_names=drone_names)
            gui.show()
            Drone.set_drone_names_motive(self.realdrones, gui.drone_names)

