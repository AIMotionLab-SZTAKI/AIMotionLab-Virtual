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


class PassiveDisplay:

    def __init__(self, xml_file_name, connect_to_optitrack=True):

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

        # Connect to optitrack
        #if connect_to_optitrack:
            #self.mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")

        self.t1 = time.time()

        # Reading model data
        print(f'Working directory:  {os.getcwd()}\n')

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



    def run(self):
        # To obtain inertia matrix
        mujoco.mj_step(self.model, self.data)

        while not glfw.window_should_close(self.window):
            # getting data from optitrack server
            #if PassiveDisplay.connect_to_optitrack:

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
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, pert=None, cam=self.activeCam, catmask=mujoco.mjtCatBit.mjCAT_ALL,
                                   scn=self.scn)
            mujoco.mjr_render(viewport, self.scn, self.con)

            glfw.swap_buffers(self.window)
            glfw.poll_events()


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
            self.key_b_callback()

        if key == glfw.KEY_D and action == glfw.RELEASE:
            self.key_d_callback()

        if key == glfw.KEY_R and action == glfw.RELEASE:
            if not self.is_recording:
                glfw.set_window_title(window, self.title + " (Recording)")
                self.is_recording = True
            else:
                glfw.set_window_title(window, self.title)
                self.is_recording = False

            

    def change_cam(self):
        """
        Change camera between scene cam and 'on board' cam
        """
        if self.activeCam == self.cam and self.DRONE_NUM > 0:
            self.activeCam = self.camFollow
        else:
            self.activeCam = self.cam


def main():
    display = PassiveDisplay("testEnvironment.xml")
    display.set_drone_names('cf4', 'cf3', 'cf10', 'cf1')
    print(display.droneNames)
    display.run()


if __name__ == '__main__':
    main()