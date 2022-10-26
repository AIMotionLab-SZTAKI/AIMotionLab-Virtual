import motioncapture
import time
import numpy as np
import mujoco
import glfw
import os
import numpy as np
import time

cam = mujoco.MjvCamera()

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
    window = glfw.create_window(1280, 720, "Optitrack Scene", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    glfw.set_scroll_callback(window, zoom)
    glfw.set_mouse_button_callback(window, mouse_button_callback)

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

    while True and not glfw.window_should_close(window):
        mc.waitForNextFrame()
        for name, obj in mc.rigidBodies.items():
            if name == 'cf4':

                pos_message = "{:.5f}".format(obj.position[0]) + "," + "{:.5f}".format(obj.position[1]) + "," + \
                              "{:.5f}".format(obj.position[2])
                t2 = time.time()-t1

                # have to put rotation.w to the front because the order is different
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]
                update_drone(data, 0, obj.position, drone_orientation)

            if name == 'cf7':

                pos_message = "{:.5f}".format(obj.position[0]) + "," + "{:.5f}".format(obj.position[1]) + "," + \
                              "{:.5f}".format(obj.position[2])
                t2 = time.time()-t1

                # have to put rotation.w to the front because the order is different
                drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]
                update_drone(data, 1, obj.position, drone_orientation)


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


def update_drone(data, droneID, position, orientation):
    startIdx = droneID * 7
    data.qpos[startIdx:startIdx + 3] = position
    data.qpos[startIdx + 3:startIdx + 7] = orientation

def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        x, y = glfw.get_cursor_pos(window)
        print(x, y)




def zoom(window, x, y):
    cam.distance -= 0.2 * y

if __name__ == '__main__':
    main()