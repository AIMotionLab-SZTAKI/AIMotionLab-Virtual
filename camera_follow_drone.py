import motioncapture
import time
import math
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


    # initialize visualization data structures
    cam.azimuth, cam.elevation = 180, -10
    cam.lookat, cam.distance = [data.qpos[0], data.qpos[1], data.qpos[2]], 1

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=50)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    #print(data.qpos.size)
    mujocoHelper.update_drone(data, 2, [0, 0, 0], [1, 0, 0, 0])

    while not glfw.window_should_close(window):
        mc.waitForNextFrame()
        for name, obj in mc.rigidBodies.items():
            if name == 'cf1':

                pos_message = "{:.5f}".format(obj.position[0]) + "," + "{:.5f}".format(obj.position[1]) + "," + \
                              "{:.5f}".format(obj.position[2])
                t2 = time.time()-t1

                # have to put rotation.w to the front because the order is different
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]
                roll_x, pitch_y, yaw_z = mujocoHelper.euler_from_quaternion(obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z)
                mujocoHelper.update_drone(data, 0, obj.position, drone_orientation)
                cam.lookat = [data.qpos[0], data.qpos[1], data.qpos[2]]
                cam.azimuth = -math.degrees(roll_x)
                cam.elevation = -math.degrees(pitch_y) - 20

            if name == 'cf4':

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





def zoom(window, x, y):
    cam.distance -= 0.02 * y

if __name__ == '__main__':
    main()