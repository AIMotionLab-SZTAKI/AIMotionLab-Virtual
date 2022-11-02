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

cam = mujoco.MjvCamera()
mouse_right_btn_down = False
prev_x, prev_y = 0.0, 0.0

def main():
    # Connect to optitrack
    mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")

    t1 = time.time()

    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')

    xmlFileName = "testEnvironment.xml"

    model = mujoco.MjModel.from_xml_path(xmlFileName)

    data = mujoco.MjData(model)

    hospitalPos, hospitalQuat, postOfficePos, postOfficeQuat = reloadScene.loadBuildingData("building_positions.txt")
    pole1Pos, pole1Quat, pole2Pos, pole2Quat, pole3Pos, pole3Quat, pole4Pos, pole4Quat = reloadScene.loadPoleData("pole_positions.txt")

    reloadScene.setBuildingData(model, hospitalPos, hospitalQuat, "hospital")
    reloadScene.setBuildingData(model, postOfficePos, postOfficeQuat, "post_office")

    reloadScene.setBuildingData(model, pole1Pos, pole1Quat, "pole1")
    reloadScene.setBuildingData(model, pole2Pos, pole2Quat, "pole2")
    reloadScene.setBuildingData(model, pole3Pos, pole3Quat, "pole3")
    reloadScene.setBuildingData(model, pole3Pos, pole3Quat, "pole4")

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1280, 720, "Optitrack Scene", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    # setup mouse callbacks
    glfw.set_scroll_callback(window, zoom)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, mouse_move_callback)


    # initialize visualization data structures
    cam.azimuth, cam.elevation = 180, -30
    cam.lookat, cam.distance = [0, 0, 0], 5

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=50)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    #print(data.qpos)

    while not glfw.window_should_close(window):
        mc.waitForNextFrame()
        for name, obj in mc.rigidBodies.items():
            if name == 'cf4':

                pos_message = "{:.5f}".format(obj.position[0]) + "," + "{:.5f}".format(obj.position[1]) + "," + \
                              "{:.5f}".format(obj.position[2])
                t2 = time.time()-t1

                # have to put rotation.w to the front because the order is different
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]
                update_drone(data, 0, obj.position, drone_orientation)

            if name == 'cf1':

                pos_message = "{:.5f}".format(obj.position[0]) + "," + "{:.5f}".format(obj.position[1]) + "," + \
                              "{:.5f}".format(obj.position[2])
                t2 = time.time()-t1

                # have to put rotation.w to the front because the order is different
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]
                mujocoHelper.update_drone(data, 1, obj.position, drone_orientation)


        mujoco.mj_step(model, data, 1)
        viewport = mujoco.MjrRect(0, 0, 0, 0)
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL,
                               scn=scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()
        #print(mouse_right_btn_down)


                #print("--------------------")
                #print("Name:", name)
                #print("Time:", t2, "sec")
                #print("Position:", obj.position)
                #print("Rotation:", obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w)
                #print("Distance:", float(recived))


def mouse_button_callback(window, button, action, mods):
    global mouse_right_btn_down
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        global prev_x, prev_y
        prev_x, prev_y = glfw.get_cursor_pos(window)
        mouse_right_btn_down = True
    elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        mouse_right_btn_down = False
        pass


def mouse_move_callback(window, xpos, ypos):
    if not mouse_right_btn_down:
        return

    global prev_x, prev_y
    x, y = glfw.get_cursor_pos(window)

    dx = x - prev_x
    dy = y - prev_y

    cam.azimuth -= dx / 10
    cam.elevation -= dy / 10

    #print("dx: " + dx)
    #print("dy: " + dy)

    #print("azi: " + cam.azimuth)
    #print("elev: " + cam.elevation)

    prev_x = x
    prev_y = y

    #print(dx, dy)

def zoom(window, x, y):
    cam.distance -= 0.2 * y

if __name__ == '__main__':
    main()