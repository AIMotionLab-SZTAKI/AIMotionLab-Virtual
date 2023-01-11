# Classes

## classes/mujoco_display.py

### class Display
  
  Description:
  
  Base class for passive and active simulation. Manages window, cameras and MuJoCo.
  
  ```
  __init__(self, xml_file_name, connect_to_optitrack=True):
  ```
Inputs:
  * xml_file_name: string, file name of the model to be loaded
  * connect_to_optitrack: boolean, whether to connect to Motive or not

Description:
  
Initializes member variables, glfw library, cameras, MuJoCo library, loads the mujoco model from xml, and looks for drones that follow the naming convention in the model. The glfw library is the window manager. The cameras are MuJoCo's MjvCamera class. There is one main camera, and one "on board" camera, that can be switched from vehicle to vehicle. The on board camera is equipped with a low pass filter for the azimuth angle, and one for the elevation angle to get rid of the high frequency judder that comes from Motive.
  
  ```
  init_glfw(self):
  ```
  
Description:
  
Initializes the glfw library. Binds methods to window events such as mouse scroll, mouse button, cursor position and keyboard.
  
  ```
  init_cams(self):
  ```
  
Description:
  
Initializes the main camera and the on board camera. Two pairs of LiveLFilter's (see in util/mujoco_helper.py) low-pass filters are also initialized for the on board camera. One pair for the azimuth angle and one pair for the elevation angle. The reason for requiring two filters for each angle is explained in the description of update_onboard_cam in util/mujoco_helper.py.
  
  ```
  load_model(self, xml_file_name):
  ```
  
Inputs:
  * xml_file_name: string, file name of the model to be loaded
  
Description:
  
Saves the xml file name in a member variable, loads the model, creates MjData from MjModel, and initializes MjvScene, and MjrContext.
  
  ```
  reload_model(self, xml_file_name, drone_names_in_motive = None):
  ```
  
Inputs:
  * xml_file_name: string, file name of the model to be loaded
  * drone_names_in_motive: list of strings, drone names that are being tracked in Motive
  
Description:
  
Calls load_model(), and then checks for drones in the model that follow the naming convention (see naming_convention_in_xml.txt) and creates a Drone or DroneMocap instance for each. Then sets the name_in_motive attribute of the mocap drones based on drone_names_in_motive list.
  
  ```
  glfw_window_should_close(self):
  ```
  
Description:
  
Makes the glfw.window_should_close(self.window) method public.

  ```
  set_key_{key}_callback(self, callback_function):
  ```

Inputs:
  * callback_function: a callable that needs to be called when a key event happens

Description:

Binds a method to key events, to make these events public.

  ```
  mouse_button_callback(self, window, button, action, mods):
  ```
  
Inputs are those that are required by glfw.

Description:

Saves cursor position and button state in member variables when either mouse buttons are pressed.

  ```
  mouse_move_callback(self, window, xpos, ypos):
  ```
  
Inputs are those that are required by glfw.

Description:

If left mouse button is pressed, rotates the camera about the lookat point when mouse is moving. If right button is pressed, moves the lookat point when mouse is moving.

  ```
  calc_dxdy(self, window):
  ```

Inputs:
  * window: a glfw window in which cursor displacement needs to be calculated

Description:

Calculates the cursor displacement since the last known cursor position. Helper function for mouse_move_callback()

  ```
  zoom(self, window, x, y):
  ```

Inputs are those that are required by glfw.

Description:

Is bound to mouse scroll callback. Moves the camera closer or further to/from the lookat point on scrolling.

  ```
  key_callback(self, window, key, scancode, action, mods):
  ```
  
Inputs are those that are required by glfw.

Description:

Manages keyboard events. Some events are processed in this method like TAB press and SPACE press. Other events are made public by calling a callback function like key_b_callback set previously from the outside.

  ```
  set_title(self, window_title):
  ```

Inputs:
  * window_title: new title for the window

Description:

Sets a new title for the window.

  ```
  append_title(self, text):
  ```

Inputs:
  * text: string, whatever needs to be added to the title

Description:

Appends a string to the window title. Like when video is being recorded to let the user know what's going on.

  ```
  reset_title(self):
  ```

Description:

Resets the window title to the original title, or the newest title if the original has been changed.

  ```
  connect_to_Optitrack(self):
  ```

Description:

Tries to connect to Motive server. Unfortunately, if Motive is not streaming, it freezes at the constructor of MotionCaptureOptitrack.

  ```
  change_cam(self):
  ```

Description:

Changes active camera to be on board camera, if it's currently main camera, and vica versa.

  ```
  save_video(self, image_list, width, height):
  ```

Inputs:
  * image_list: the list of frames that have been saved during video recording
  * width: the width of the images
  * height: the height of the images

Description:

Writes the saved frames as mp4 video to hard drive at self.video_save_folder using OpenCV's VideoWriter. If the folder does not exist, it creates it. Since the recorded frames are 1D arrays, the method converts them to the desired shape OpenCV requires.


  ```
  save_video_background(self):
  ```

Description:

Runs save_video() on a different thread with a copy of image_list, and sets image_list to an empty list, so that the next video recording can be started even before the saving process has finished.

  ```
  set_drone_names(self):
  ```

Description:

If there are any mocap drones, it creates a small pop-up window where the user can set the mocap drones' name_in_motive.
  
## classes/active_simulation.py

### class ActiveSimulator(Display)

Description:
  
Child of Display. Handles the simulation the graphics, video recording and will handle data logging in the future.


  ```
  __init__(self, xml_file_name, video_intervals, sim_step, control_step, graphics_step, connect_to_optitrack=True):
  ```
Inputs:
  * xml_file_name: string, file name of the model to be loaded
  * video_intervals: list of floats, with even number of elements. Each pair represents a time interval in seconds when video should be recorded. 0 seconds is when simulation starts. Set it to None if no video needs to be recorded.
  * sim_step: float, in seconds, how frequently physics should be updated
  * control_step: float, in seconds, how frequently control is updated
  * graphics_step: float, in seconds, how frequently graphics is updated
  * connect_to_optitrack: boolean, whether to connect to Motive or not
  
Description:

Saves the inputs for later use, and steps physics once to get inertia matrix.
  
  
  ```
  update(self, i):
  ```
Inputs:
  * i: the loop variable
  
Description:

Should be called in an infinite loop, passing in the incremented loop variable. If connected to Motive, it checks for data coming in based on which it updates the mocap vehicles. It updates simulated vehicles, steps the physics, updates the window displaying the graphics, and manages video recording.
  
  
  ```
  manage_video_recording():
  ```
Description:

It is called on every update. It checks whether frames should be saved based on video_intervals. When the end of an interval is reached, it writes the saved frames as an .mp4 to hard disk on a different thread. The location of the saved video is printed to the terminal.
  
