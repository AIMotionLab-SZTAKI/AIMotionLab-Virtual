import motioncapture
import time
import numpy as np
import mujoco
import glfw
import os
import numpy as np
import time


def main():
    # Connect to optitrack
    mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")

    t1 = time.time()

    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')

    xmlFileName = "testEnvironment.xml"

    model = mujoco.MjModel.from_xml_path(xmlFileName)

    data = mujoco.MjData(model)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1280, 720, "Reloaded Scene", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 180, -30
    cam.lookat, cam.distance = [0, 0, 0], 4

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=50)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)

    while True and not glfw.window_should_close(window):
        mc.waitForNextFrame()
        for name, obj in mc.rigidBodies.items():
            if name == 'cf4':

                pos_message = "{:.5f}".format(obj.position[0]) + "," + "{:.5f}".format(obj.position[1]) + "," + \
                              "{:.5f}".format(obj.position[2])
                t2 = time.time()-t1

                drone_pos_rot = [obj.position[0], obj.position[1], obj.position[2], obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w]
                data.qpos = drone_pos_rot


        mujoco.mj_step(model, data, 1)
        viewport = mujoco.MjrRect(0, 0, 0, 0)
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL,
                               scn=scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()


                #print("--------------------")
                #print("Name:", name)
                #print("Time:", t2, "sec")
                #print("Position:", obj.position)
                #print("Rotation:", obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w)
                #print("Distance:", float(recived))




if __name__ == '__main__':
    main()