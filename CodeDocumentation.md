# Classes

## classes/mujoco_display.py

### class Display
  
  Description:
  Base class for passive and active simulation. Manages window, cameras and MuJoCo.
  
  1. Constructor
  ```
  __init__(self, xml_file_name, connect_to_optitrack=True):
  ```
  Inputs:
    * xml_file_name: string, file name of the model to be loaded
    * connect_to_optitrack: boolean, whether to connect to Motive or not

  Description:
  Initializes member variables, glfw library, cameras, MuJoCo library, loads the mujoco model from xml, and looks for drones that follow the naming convention in the model. The glfw library is the window manager. The cameras are MuJoCo's MjvCamera class. There is one main camera, and one "on board" camera, that can be switched from vehicle to vehicle. The on board camera is equipped with a low pass filter for the azimuth angle, and one for the elevation angle to get rid of the high frequency judder that comes from Motive.
  
  2.
  ```
  init_glfw(self):
  ```
  
  Description:
  Initializes the glfw library. Binds methods to window events such as mouse scroll, mouse button, cursor position and keyboard.
  
  3.
  ```
  init_cams(self):
  ```
  
  Description:
  Initializes the main camera and the on board camera. Two pairs of LiveLFilter's (see in util/mujoco_helper.py) low-pass filters are also initialized for the on board camera. One pair for the azimuth angle and one pair for the elevation angle. The reason for requiring two filters for each angle is explained in the description of update_onboard_cam in util/mujoco_helper.py.
  
  4.
  ```
  load_model(self, xml_file_name):
  ```
  
  Inputs:
    * xml_file_name: string, file name of the model to be loaded
  
  Description:
  Saves the xml file name in a member variable, loads the model, creates MjData from MjModel, and initializes MjvScene, and MjrContext.
  
  5.
  ```
  reload_model(self, xml_file_name, drone_names_in_motive = None):
  ```
  
  Inputs:
    * xml_file_name: string, file name of the model to be loaded
    * drone_names_in_motive: list of strings, drone names that are being tracked in Motive
  
  Description:
  Calls load_model(), and then checks for drones in the model that follow the naming convention (see naming_convention_in_xml.txt) and creates a Drone or DroneMocap instance for each. Then sets the name_in_motive attribute of the drones based on drone_names_in_motive list.
  
  6.
  ```
  glfw_window_should_close(self):
  ```
  
  Description:
  Makes the glfw.window_should_close(self.window) method public.
  
  


## classes/active_simulation.py

### class ActiveSimulator(Display)

  Description:
  
  Child of Display. Handles the simulation the graphics, video recording and will handle data logging in the future.

  1. Constructor 
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
  
  2.
  ```
  update(self, i):
  ```
  Inputs:
    * i: the loop variable
  
  Description:
  Should be called in an infinite loop, passing in the loop variable. If connected to Motive, it checks for data coming in based on which it updates the mocap vehicles. It updates simulated vehicles, steps the physics, updates the window displaying the graphics, and manages video recording.
  
  3.
  ```
  manage_video_recording():
  ```
  Description:
  It is called on every update. It checks whether frames should be saved based on video_intervals. When the end of an interval is reached, it writes the saved frames as an .mp4 to hard disk on a different thread. The location of the saved video is printed to the terminal.
  
