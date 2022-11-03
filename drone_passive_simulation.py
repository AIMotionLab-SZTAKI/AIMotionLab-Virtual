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

MAX_DRONE_NUM = 2

class PassiveDisplay:
    cam = mujoco.MjvCamera()
    camFollow = mujoco.MjvCamera()
    activeCam = cam
    mouse_left_btn_down = False
    mouse_right_btn_down = False
    prev_x, prev_y = 0.0, 0.0
    followed_drone_ID = 0

    def __init__(self):
        # Connect to optitrack
        self.mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")

        self.t1 = time.time()

        # Reading model data
        print(f'Working directory:  {os.getcwd()}\n')

        xmlFileName = "testEnvironment.xml"

        self.model = mujoco.MjModel.from_xml_path(xmlFileName)

        self.data = mujoco.MjData(self.model)

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

    def run(self):
        ## To obtain inertia matrix
        mujoco.mj_step(self.model, self.data)

        while not glfw.window_should_close(self.window):
            # getting data from optitrack server
            self.mc.waitForNextFrame()
            for name, obj in self.mc.rigidBodies.items():

                # have to put rotation.w to the front because the order is different
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

                if name == 'cf1':
                    mujocoHelper.update_drone(self.data, 0, obj.position, drone_orientation)

                if name == 'cf3':
                    mujocoHelper.update_drone(self.data, 1, obj.position, drone_orientation)

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


    def mouse_move_callback(window, xpos, ypos):
        if PassiveDisplay.activeCam != PassiveDisplay.cam:
            return

        if PassiveDisplay.mouse_left_btn_down:
            dx, dy, PassiveDisplay.prev_x, PassiveDisplay.prev_y = PassiveDisplay.calc_dxdy(window)

            PassiveDisplay.cam.azimuth -= dx / 10
            PassiveDisplay.cam.elevation -= dy / 10

        if PassiveDisplay.mouse_right_btn_down:

            dx, dy, PassiveDisplay.prev_x, PassiveDisplay.prev_y = PassiveDisplay.calc_dxdy(window)
            PassiveDisplay.cam.lookat[2] += dy / 100




    def calc_dxdy(window):
        x, y = glfw.get_cursor_pos(window)
        dx = x - PassiveDisplay.prev_x
        dy = y - PassiveDisplay.prev_y

        return dx, dy, x, y



    def zoom(window, x, y):
        PassiveDisplay.activeCam.distance -= 0.2 * y


    def key_callback(window, key, scancode, action, mods):

        if key == glfw.KEY_TAB and action == glfw.PRESS:
            PassiveDisplay.change_cam()
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            if PassiveDisplay.activeCam == PassiveDisplay.camFollow:
                if PassiveDisplay.followed_drone_ID + 1 == MAX_DRONE_NUM:
                    PassiveDisplay.followed_drone_ID = 0
                else:
                    PassiveDisplay.followed_drone_ID += 1


    def change_cam():
        if PassiveDisplay.activeCam == PassiveDisplay.cam:
            PassiveDisplay.activeCam = PassiveDisplay.camFollow
        else:
            PassiveDisplay.activeCam = PassiveDisplay.cam


def main():
    display = PassiveDisplay()
    display.run()


if __name__ == '__main__':
    main()