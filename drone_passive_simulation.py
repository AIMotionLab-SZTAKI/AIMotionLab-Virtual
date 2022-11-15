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
import mujocoHelper
import cv2
from util.util import sync


class PassiveDisplay:

    def __init__(self, xml_file_name, connect_to_optitrack=True):
        print(f'Working directory:  {os.getcwd()}\n')

        self.connect_to_optitrack = connect_to_optitrack

        self.cam = mujoco.MjvCamera()
        self.camFollow = mujoco.MjvCamera()
        self.activeCam = self.cam
        self.mouse_left_btn_down = False
        self.mouse_right_btn_down = False
        self.prev_x, self.prev_y = 0.0, 0.0
        self.followed_drone_ID = 0
        self.DRONE_NUM = 0
        self.title = "Optitrack Scene"
        self.is_recording = False
        self.image_list = []
        self.video_save_folder = "video_capture"
        self.video_file_name_base = "output"

        # Connect to optitrack
        #if connect_to_optitrack:
        #    self.mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")
        #    print("[PassiveDisplay] Connected to Optitrack")

        self.t1 = time.time()

        # Reading model data

        self.xmlFileName = xml_file_name

        self.model = mujoco.MjModel.from_xml_path(self.xmlFileName)

        self.data = mujoco.MjData(self.model)

        self.DRONE_NUM = int(self.data.qpos.size / 7)
        self.droneNames = []
        for i in range(self.DRONE_NUM):
            self.droneNames.append("cf" + str(i + 1))

        # Initialize the library
        if not glfw.init():
            return

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1280, 720, self.title, None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)
        # setup mouse callbacks
        glfw.set_scroll_callback(self.window, self.zoom)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_key_callback(self.window, self.key_callback)

        # initialize visualization data structures
        self.cam.azimuth, self.cam.elevation = 180, -30
        self.cam.lookat, self.cam.distance = [0, 0, 0], 5
        self.camFollow.azimuth, self.camFollow.elevation = 180, -30
        self.camFollow.lookat, self.camFollow.distance = [0, 0, 0], 0.8

        self.pert = mujoco.MjvPerturb()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=50)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        self.viewport = mujoco.MjrRect(0, 0, 0, 0)
        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)

        #print(self.data.qpos.size)

    def set_drone_names(self, *names):
        drone_num = len(names)
        if len(names) > self.DRONE_NUM:
            print("Error: too many (" + str(drone_num) + ") drone names provided. Number of drones in the xml is: " + str(self.DRONE_NUM))
            print('Last ' + str(len(names) - self.DRONE_NUM) + ' drone(s) ignored.')
            drone_num = self.DRONE_NUM

        self.droneNames = []
        for i in range(drone_num):
            self.droneNames.append(names[i])
    

    def set_key_b_callback(self, callback_function):
        self.key_b_callback = callback_function


    def set_key_d_callback(self, callback_function):
        self.key_d_callback = callback_function
    

    def reload_model(self, xml_file_name):
        self.xmlFileName = xml_file_name

        self.model = mujoco.MjModel.from_xml_path(self.xmlFileName)
        self.data = mujoco.MjData(self.model)

        self.scn = mujoco.MjvScene(self.model, maxgeom=50)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        
        self.DRONE_NUM = int(self.data.qpos.size / 7)
        self.droneNames = []
        for i in range(self.DRONE_NUM):
            self.droneNames.append("cf" + str(i + 1))


    def run(self):
        # To obtain inertia matrix
        mujoco.mj_step(self.model, self.data)

        self.timestep = 0.04
        i = 0
        start = time.time()
        while not glfw.window_should_close(self.window):
            
            # getting data from optitrack server
            #if self.connect_to_optitrack:

            #    self.mc.waitForNextFrame()
            #    for name, obj in self.mc.rigidBodies.items():

                    # have to put rotation.w to the front because the order is different
            #        drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

            #        try:
            #            idx = self.droneNames.index(name)
            #        except ValueError:
            #            idx = -1

            #        if idx >= 0:
            #            mujocoHelper.update_drone(self.data, idx, obj.position, drone_orientation)

            if self.activeCam == self.camFollow and self.DRONE_NUM > 0:
                mujocoHelper.update_follow_cam(self.data.qpos, self.followed_drone_ID, self.camFollow)

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
                if self.followed_drone_ID + 1 == self.DRONE_NUM:
                    self.followed_drone_ID = 0
                else:
                    self.followed_drone_ID += 1
        
        if key == glfw.KEY_B and action == glfw.RELEASE:
            """
            Pass on this event
            """
            self.key_b_callback()

        if key == glfw.KEY_D and action == glfw.RELEASE:
            """
            Pass on this event
            """
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
                self.save_video()

        if key == glfw.KEY_C and action == glfw.RELEASE:
            self.connect_to_Optitrack()


    def connect_to_Optitrack(self):
        #self.connect_to_optitrack = True
        #self.mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")
        #print("[PassiveDisplay] Connected to Optitrack")
        pass

    def change_cam(self):
        """
        Change camera between scene cam and 'on board' cam
        """
        if self.activeCam == self.cam and self.DRONE_NUM > 0:
            self.activeCam = self.camFollow
        else:
            self.activeCam = self.cam
    

    def save_video(self):
        """
        Write saved images to hard disk as .mp4
        """
        print("[PassiveDisplay] Saving video...")
        # checking for folder
        if not os.path.exists(self.video_save_folder):
            # then create folder
            os.mkdir(self.video_save_folder)

        fps = 1 / self.timestep

        glfw.set_window_title(self.window, self.title + " (Saving video...)")
        time_stamp = self.image_list[0][0].replace('.', '_')
        out = cv2.VideoWriter(os.path.join(self.video_save_folder, self.video_file_name_base + '_' + time_stamp + '.mp4'),\
              cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.viewport.width, self.viewport.height))
        for i in range(len(self.image_list)):
            #print(self.image_list[i][0])
            rgb = np.reshape(self.image_list[i][1], (self.viewport.height, self.viewport.width, 3))
            rgb = cv2.cvtColor(np.flip(rgb, 0), cv2.COLOR_BGR2RGB)
            out.write(rgb)
        out.release()
        self.image_list = []
        print("[PassiveDisplay] Saved video in " + os.path.join(os.getcwd(), self.video_save_folder))
        glfw.set_window_title(self.window, self.title)



"""
def main():
    display = PassiveDisplay("testEnvironment.xml")
    display.set_drone_names('cf4', 'cf3', 'cf10', 'cf1')
    print(display.droneNames)
    display.run()


if __name__ == '__main__':
    main()
"""