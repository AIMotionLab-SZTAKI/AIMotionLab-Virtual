from email.mime import image
import mujoco
import glfw
import os
import numpy as np
import time
#import imageio
import cv2

def main():
    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')

    print(type(time.time()))

    xmlFileName = "built_scene.xml"

    model = mujoco.MjModel.from_xml_path(xmlFileName)

    data = mujoco.MjData(model)

    hospitalPos, hospitalQuat, postOfficePos, postOfficeQuat = loadBuildingData("building_positions.txt")
    pole1Pos, pole1Quat, pole2Pos, pole2Quat, pole3Pos, pole3Quat, pole4Pos, pole4Quat = loadPoleData("pole_positions.txt")

    #setBuildingData(model, hospitalPos, hospitalQuat, "hospital")
    #setBuildingData(model, postOfficePos, postOfficeQuat, "post_office")

    #setBuildingData(model, pole1Pos, pole1Quat, "pole1")
    #setBuildingData(model, pole2Pos, pole2Quat, "pole2")
    #setBuildingData(model, pole3Pos, pole3Quat, "pole3")
    #setBuildingData(model, pole3Pos, pole3Quat, "pole4")

    #saveModelAsXml(model, "mod" + xmlFileName)

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
    cam.lookat,  cam.distance  = [0, 0, 0], 3
    
    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=50)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    #mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)

    #print(con.currentBuffer)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)

    idx = 1 * 7 + 2
    #pos_z = data.qpos[idx]

    image_list = []


    while not glfw.window_should_close(window):
        mujoco.mj_step(model, data, 1)
        viewport = mujoco.MjrRect(0,0,0,0)
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

        rgb = np.zeros(viewport.width * viewport.height * 3, dtype=np.uint8)
        depth = np.zeros(viewport.width * viewport.height, dtype=np.float32)

        stamp = str(time.time())
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, stamp, None, con)
        
        mujoco.mjr_readPixels(rgb, depth, viewport, con)

        #rgb = np.reshape(rgb, (viewport.height, viewport.width, 3))
        #print(rgb.shape)
        #imageio.imwrite("image_capture/" + stamp + ".jpg", rgb)
        
        image_list.append([stamp, rgb])

    #image_list.sort(key=stamp_value)
    out = cv2.VideoWriter(os.path.join('image_capture', 'output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (viewport.width, viewport.height))
    for i in range(len(image_list)):
      print(image_list[i][0])
      rgb = np.reshape(image_list[i][1], (viewport.height, viewport.width, 3))
      rgb = cv2.cvtColor(np.flip(rgb, 0), cv2.COLOR_BGR2RGB)
      #imageio.imwrite("image_capture/" + image_list[i][0] + ".jpg", rgb)
      out.write(rgb)
    out.release()


    glfw.terminate()


def stamp_value(input):
  return input[0]

# sets new position and orientation for a building specified by buildingName
def setBuildingData(model, newPosition, newOrientation, buildingName: str):
  building = model.body(buildingName)
  building.pos = newPosition
  building.quat = newOrientation

def saveModelAsXml(model, fileName):
  mujoco.mj_saveLastXML(fileName, model)


def loadBuildingData(fileName):
  hospitalData, postOfficeData = np.loadtxt(fileName, delimiter=",")
  # the two arrays should have 7 elements
  # first three numbers are position, the remaining 4 are orientation as quaternion
  return hospitalData[:3], hospitalData[3:], postOfficeData[:3], postOfficeData[3:]


def loadPoleData(fileName):
  pole1, pole2, pole3, pole4 = np.loadtxt(fileName, delimiter=",")
  # the two arrays should have 7 elements
  # first three numbers are position, the remaining 4 are orientation as quaternion
  return pole1[:3], pole1[3:], pole2[:3], pole2[3:], pole3[:3], pole3[3:], pole4[:3], pole4[3:]

if __name__ == '__main__':
    main()
