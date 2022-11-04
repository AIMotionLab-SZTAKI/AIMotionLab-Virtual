import math

import motioncapture
import time
import numpy as np
import mujoco
import glfw
import os
import numpy as np
import time
import reloadScene
import mujocoHelper


class PassiveDisplay:
    cam = mujoco.MjvCamera()
    camFollow = mujoco.MjvCamera()
    activeCam = cam
    mouse_left_btn_down = False
    mouse_right_btn_down = False
    prev_x, prev_y = 0.0, 0.0
    followed_drone_ID = 0
    DRONE_NUM = 0

    def __init__(self, xml_file_name):

        # Connect to optitrack
        self.mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")

        self.t1 = time.time()

        # Reading model data
        print(f'Working directory:  {os.getcwd()}\n')

        self.xmlFileName = xml_file_name

        self.model = mujoco.MjModel.from_xml_path(self.xmlFileName)

        self.data = mujoco.MjData(self.model)

        PassiveDisplay.DRONE_NUM = int(self.data.qpos.size / 7)
        self.droneNames = []
        for i in range(PassiveDisplay.DRONE_NUM):
            self.droneNames.append("cf" + str(i + 1))

        hospitalPos, hospitalQuat, postOfficePos, postOfficeQuat = reloadScene.loadBuildingData("building_positions.txt")
        pole1Pos, pole1Quat, pole2Pos, pole2Quat, pole3Pos, pole3Quat, pole4Pos, pole4Quat = reloadScene.loadPoleData("pole_positions.txt")

        reloadScene.setBuildingData(self.model, hospitalPos, hospitalQuat, "hospital")
        reloadScene.setBuildingData(self.model, postOfficePos, postOfficeQuat, "post_office")

        reloadScene.setBuildingData(self.model, pole1Pos, pole1Quat, "pole1")
        reloadScene.setBuildingData(self.model, pole2Pos, pole2Quat, "pole2")
        reloadScene.setBuildingData(self.model, pole3Pos, pole3Quat, "pole3")
        reloadScene.setBuildingData(self.model, pole3Pos, pole3Quat, "pole4")

        # Initialize the library
        if not glfw.init():
            return

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1280, 720, "Optitrack Scene", None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)
        # setup mouse callbacks
        glfw.set_scroll_callback(self.window, PassiveDisplay.zoom)
        glfw.set_mouse_button_callback(self.window, PassiveDisplay.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, PassiveDisplay.mouse_move_callback)
        glfw.set_key_callback(self.window, PassiveDisplay.key_callback)

        # initialize visualization data structures
        PassiveDisplay.cam.azimuth, PassiveDisplay.cam.elevation = 180, -30
        PassiveDisplay.cam.lookat, PassiveDisplay.cam.distance = [0, 0, 0], 5
        PassiveDisplay.camFollow.azimuth, PassiveDisplay.camFollow.elevation = 180, -30
        PassiveDisplay.camFollow.lookat, PassiveDisplay.camFollow.distance = [0, 0, 0], 1

        self.pert = mujoco.MjvPerturb()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=50)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        #print(self.data.qpos.size)

    def set_drone_names(self, *names):
        drone_num = len(names)
        if len(names) > PassiveDisplay.DRONE_NUM:
            print("Error: too many drones provided. Number of drones in the xml is: " + str(PassiveDisplay.DRONE_NUM))
            print('Last ' + str(len(names) - PassiveDisplay.DRONE_NUM) + ' drone(s) ignored.')
            drone_num = PassiveDisplay.DRONE_NUM

        self.droneNames = []
        for i in range(drone_num):
            self.droneNames.append(names[i])


    def run(self):
        # To obtain inertia matrix
        mujoco.mj_step(self.model, self.data)

        while not glfw.window_should_close(self.window):
            # getting data from optitrack server
            self.mc.waitForNextFrame()
            for name, obj in self.mc.rigidBodies.items():

                # have to put rotation.w to the front because the order is different
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

                try:
                    idx = self.droneNames.index(name)
                except ValueError:
                    idx = -1

                if idx >= 0:
                    mujocoHelper.update_drone(self.data, idx, obj.position, drone_orientation)

            if PassiveDisplay.activeCam == PassiveDisplay.camFollow:
                mujocoHelper.update_follow_cam(self.data.qpos, PassiveDisplay.followed_drone_ID, PassiveDisplay.camFollow)

            mujoco.mj_step(self.model, self.data, 1)
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, pert=None, cam=PassiveDisplay.activeCam, catmask=mujoco.mjtCatBit.mjCAT_ALL,
                                   scn=self.scn)
            mujoco.mjr_render(viewport, self.scn, self.con)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

    @staticmethod
    def mouse_button_callback(window, button, action, mods):

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:

            PassiveDisplay.prev_x, PassiveDisplay.prev_y = glfw.get_cursor_pos(window)
            PassiveDisplay.mouse_left_btn_down = True

        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            PassiveDisplay.mouse_left_btn_down = False

        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            PassiveDisplay.prev_x, PassiveDisplay.prev_y = glfw.get_cursor_pos(window)
            PassiveDisplay.mouse_right_btn_down = True

        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
            PassiveDisplay.mouse_right_btn_down = False

    @staticmethod
    def mouse_move_callback(window, xpos, ypos):
        if PassiveDisplay.activeCam != PassiveDisplay.cam:
            return

        if PassiveDisplay.mouse_left_btn_down:
            """
            Rotate camera about the lookat point
            """
            dx, dy = PassiveDisplay.calc_dxdy(window)
            scale = 0.1
            PassiveDisplay.cam.azimuth -= dx * scale
            PassiveDisplay.cam.elevation -= dy * scale

        if PassiveDisplay.mouse_right_btn_down:
            """
            Move the point that the camera is looking at
            """
            dx, dy = PassiveDisplay.calc_dxdy(window)
            scale = 0.005
            angle = math.radians(PassiveDisplay.cam.azimuth + 90)
            dx3d = math.cos(angle) * dx * scale
            dy3d = math.sin(angle) * dx * scale

            PassiveDisplay.cam.lookat[0] += dx3d
            PassiveDisplay.cam.lookat[1] += dy3d

            # vertical axis is Z in 3D, so 3rd element in the lookat array
            PassiveDisplay.cam.lookat[2] += dy * scale

    @staticmethod
    def calc_dxdy(window):
        """
        Calculate cursor displacement
        """
        x, y = glfw.get_cursor_pos(window)
        dx = x - PassiveDisplay.prev_x
        dy = y - PassiveDisplay.prev_y
        PassiveDisplay.prev_x, PassiveDisplay.prev_y = x, y

        return dx, dy

    @staticmethod
    def zoom(window, x, y):
        """
        Change distance between camera and lookat point by mouse wheel
        """
        PassiveDisplay.activeCam.distance -= 0.2 * y

    @staticmethod
    def key_callback(window, key, scancode, action, mods):
        """
        Switch camera on TAB press
        Switch among drones on SPACE press if camera is set to follow drones
        """
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            PassiveDisplay.change_cam()
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            if PassiveDisplay.activeCam == PassiveDisplay.camFollow:
                if PassiveDisplay.followed_drone_ID + 1 == PassiveDisplay.DRONE_NUM:
                    PassiveDisplay.followed_drone_ID = 0
                else:
                    PassiveDisplay.followed_drone_ID += 1

    @staticmethod
    def change_cam():
        """
        Change camera between scene cam and 'on board' cam
        """
        if PassiveDisplay.activeCam == PassiveDisplay.cam:
            PassiveDisplay.activeCam = PassiveDisplay.camFollow
        else:
            PassiveDisplay.activeCam = PassiveDisplay.cam


def main():
    display = PassiveDisplay("testEnvironment_4drones.xml")
    display.set_drone_names('cf5', 'cf2', 'cf10', 'cf4')
    print(display.droneNames)
    display.run()


if __name__ == '__main__':
    main()