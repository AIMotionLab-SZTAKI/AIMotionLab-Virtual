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
  
## classes/drone.py

### class SPIN_DIR(Enum)

Description:

The direction in which a propeller of a drone spins. Clockwise is 1 and counter clockwise is -1 so that the speed of the spin can easily be multiplied by it.

### class Drone

Description:
A virtually simulated drone using the MuJoCo library. It manages the data that corresponds to this drone in the MjData.
  
  ```
  __init__(self, data: mujoco.MjData, name_in_xml, trajectory, controllers, parameters = {"mass" : 0.1}):
  ```
  
Inputs:
  * data: MjData created from the loaded MjModel
  * name_in_xml: the name that follows the naming convention of this drone in the xml
  * trajectory: an instance of any class that is a child class of Trajectory
  * controllers: a dictionary of controllers of that this drone needs
  * parameters: a dictionary of parameters of this drone

Description:

Saves a view of the inputs in member variables. Looks for its free joint and the joint of the propellers in the data input. Saves a view of its qpos and the qpos of each propellers as member variables. Also saves a view of its control from the data input. Makes a copy of the initial propeller angle, so that the spin can be updated such that the physics does not affect it. Saves a view of its body.xquat, free_joint.qvel, and data.sensor that corresponds to this drone. These are necessary to make the methods much more readable, because in MjData, everything is stacked on top of each other, and it's a mess.

  ```
  update(self, i):
  ```
  
Inputs:
  * i: integer, the incremented loop variable

Description:

This needs to be called in every simulation control step. If the drone has a trajectory, it gets evaluated, and it's output is passed to the drone's controllers that compute it's control. Then the new control corresponding to this drone is set in MjData.

  ```
  compute_control(self, input_dict):
  ```

Inputs:
  * input_dict: a dictionary that is the output of the evaluation of the trajectory

Description:

Depending on which controller is needed for the current part of the trajectory, the method calls the corresponding controller's control computing method.

  ```
  spin_propellers(self, angle_step):
  ```
  
Inputs:
  * angle_step: float, by how much the propeller needs to be rotated in each step
 
Description:

It updates the propeller angles that have been copied in the constructor each with a small difference so that it is a bit more visually convincing, and updates the joint angle of each of the four propellers.

  ```
  fake_propeller_spin(self, control_step , speed = 10):
  ```

Inputs:
  * control_step: float, in seconds, how frequently control is updated
  * speed: float, the speed of the spinning

Description:

Spins the drone's propellers if they are above 0.1m, and stops the propellers if they are below.

  ```
  parse_drones(data, joint_names): (static method)
  ```

Inputs:
  * joint_names: a list of names of all joints in the model (can be generated by mujoco_helper.get_joint_name_list())

Returns a list of virtually simulated Drone/DroneHooked instances.

Description:

Looks for drones that follow the naming convention in the list of joints. If one is found it checks whether it's a crazyflie or a bumblebee, and whether it has a hook. Then it creates a corresponding Drone or DroneHooked instance, and places it in a list which it returns after it went through the joint list.

  ```
  find_hook_for_drone(names, drone_name): (static method)
  ```

Inputs:
  * names: list of joint names in the model
  * drone_name: name of the drone that needs a hook

Returns the name of the hook in the xml, or None if it wasn't found.

### class DroneHooked(Drone)

Description:

Child class of Drone. Extends Drone class with a hanging hook for picking up objects. It manages the data that corresponds to this drone in the MjData.

  ```
  __init__(self, data: mujoco.MjData, name_in_xml, hook_name_in_xml, trajectory, controller, parameters = {"mass" : 0.1}):
  ```

Inputs are the same as those of Drone's constructor plus a string called hook_name_in_xml.

Description:

In addition to calling Drone's constructor, it saves the hook_name_in_xml and mines the hook qpos, qvel and rod length from MjData.


  ```
  update(self, i):
  ```

Description:

Same as Drone's update.

  ```
  compute_control(self, input_dict):
  ```

Description:

Very similar to Drone's compute_control(), except that it takes the hook position and velocity into account.

### class DroneMocap

Description:

Not simulated, only mirrors a real drone in the simulation that's being tracked by Optitrack or other motion capture systems. It manages the data that corresponds to this drone in the MjData. MuJoCo provides mocap bodies, but since those bodies cannot have child bodies, to immitate propeller spinning, each propeller needs to be defined as a separate body in the xml. If the xml is generated by util/xml_generator.py, the drones and the propellers will follow the naming convention.

  ```
  __init__(self, model: mujoco.MjModel, data: mujoco.MjData, drone_mocapid, prop_mocapids, name_in_motive, name_in_xml):
  ```

Inputs:
  * model: the loaded MjModel
  * data: the MjData created from MjModel
  * drone_mocapid: the mocap id of this drone that can be used to get its position and orientation from MjData
  * prop_mocapids: same as drone_mocapid, but for propellers, not used anymore
  * name_in_motive: to initialize drone with a name that's being streamed from Motive
  * name_in_xml: the name of the drone in the xml

Description:

Saves views of the inputs, and creates a PropellerMocap instance for each propeller.

  ```
  set_qpos(self, pos, quat):
  ```

Description:

To match the simulated (non-mocap) Drone method name. This drone does not have a qpos in MjData, because it's a mocap body and does not have any joints.

  ```
  parse_mocap_drones(data, model, body_names): (static function)
  ```

Inputs:
  * data: MjData of the whole model
  * model: MjModel of the whole model
  * body_names: a list of body names in the model (can be generated by mujoco_helper.get_body_name_list())

Description:

Does the same as Drone.parse_drones(), except with mocap drones. It needs the model as well to be able to identify body names.

### Class DroneMocapHooked(DroneMocap)

Description:

A mocap drone that has a hook hanging on a rod for picking up objects. Child class of DroneMocap. It manages the data that corresponds to this drone in the MjData. Again, since mocap bodies can't have child bodies in MuJoCo, the hook needs to be a separate body like the propellers. If it were only geoms in the drone's body, it would be glued to it, and couldn't be moving like a pendulum.

  ```
  __init__(self, model: mujoco.MjModel, data: mujoco.MjData, drone_mocapid, prop_mocapids, name_in_motive, name_in_xml, hook_name_in_xml):
  ```

Inputs are the same as DroneMocap plus hook_name_in_xml.

Description:

In addition to DroneMocap's constructor, it saves hook_name_in_xml in a member variable, and searches for the mocapid of the hook in the model. Also saves a view of the hook's position and quaternion from MjData, so that they can be more easily updated later.


  ```
  set_qpos(self, pos, quat):
  ```

Inputs:
  * pos: an array of 3 floats, the position to be set
  * quat: an array of 4 floats, the quaternion to be set

Description:

It not only sets the position and orientation of the drone, but those of the hook too, so that the hook follows the drone. This will need to be changed later when there will be a sensor on the hook of the real drone, so that the hook angle can be changed too.

### class PropellerMocap

  ```
  __init__(self, model, data, name_in_xml, drone_mocap_id, spin_direction = SPIN_DIR.CLOCKWISE):
  ```

Inputs:
  * model: the MjModel in which the drone is
  * data: MjData created from MjModel
  * name_in_xml: the name of the propeller in the xml
  * drone_mocap_id: the mocapid of the drone that owns this propeller
  * spin_direction: the direction in which the propeller needs to be spinned

Description:

Saves its name_in_xml, searches for its mocapid in th model, saves a view of its porition and quaternion, identifies its offset from the center of the drone and saves a view of the owning drone's position and quaternion.

  ```
  update(self):
  ```

Description:

Updates the propeller's position and orientation such that it's center stays the original with respect to the drone even when the propeller is spinning and the drone is moving.

  ```
  spin(self, control_step, spin_speed):
  ```

Inputs:
  * control_step: float, in seconds, how frequently control is updated
  * spin_speed: float, how fast the propeller needs to spin

Description:

Increments or decrements the angle of the propeller by (spin_speed * control_step) on the horizontal plane depending on which direction it's spinning. Then it calls update, to apply the changes.

## classes/passive_display.py

### class PassiveDisplay(Display)

Description:

Child class of Display. Use if no simulated drones are needed, only mocap drones to have a digital twin of reality.

  ```
  __init__(self, xml_file_name, connect_to_optitrack=True):
  ```

Description:

Same as Display's constructor.

  ```
  run(self):
  ```

Description:

Contains an infinite loop that cycles at every graphics_step that's initially 0.04s. In every cycle, it grabs data from the motion capture system, and updates the vehicles' position and orientation, it rotates the propellers to immitate spinning, renders the scene, and manages video recording. If the window is closed while video is being recorded, it writes the saved frames to hard disk as mp4.


## classes/trajectory.py

### class Trajectory

Description:

Base class for drone trajectories. Contains the evaluate method that trajectories must implement. This acts as an interface so that it'd be easy to equip drones with different trajectories. Each drone has a Trajectory member variable whose evaluate() method is called in the drone's update(), and the evaluate() always returns the required format for the drone's controllers.

  ```
  __init__(self, control_step):
  ```

Inputs:
  * control_step: float, in seconds, how frequently control is updated

Description:

Initializes the dictionary that will be the output of each evaluate() call.

  ```
  evaluate(self, i, simtime) -> dict:
  ```

Inputs:
  * i: the loop variable
  * simtime: time that has passed since the simulation started

Description:

The method, that derived classes must implement. It evaluates the trajectory and returns the input parameters (the dictionary initialized in the constructor) for the controllers.

### class TestTrajectory(Trajectory)

Description:

Implements Peter Antal's trajectories for the flip, flying on a Lissajous curve, and carrying 3 payloads. To select from the options, pass FLY, FLIP or HOOK_UP_3_LOADS in the constructor for scenario.

# Scripts

## scripts/build_scene.py

Contains methods for building a scene with either GUI's or automatically from Motive stream. It uses a PassiveDisplay for visual representation, a SceneXmlGenerator to generate the model as xml, and three GUI's for adding buildings, drones, and payloads. For usage, see README.md.

### Methods

  ```
  add_building():
  ```

Description:

Creates and shows a BuildingInputGui pop-up window in which the user can select the kind of building, its position and orientation. It then adds the building to the xml model by the SceneXmlGenerator. Every time the user adds a new building, the xml is saved, and reloaded in MuJoCo, so that the user can see the new model visually. Some buildings can only be added once, like hospital and Sztaki, because there's only one of them in the real test environment too. Multiple poles and landing zones can be added however.


  ```
  add_drone():
  ```

Description:

Creates and shows a DroneInputGui pop-up window in which the user can select the tye of drone, its position and orientation. Then, it adds the desired drone to the xml model via the SceneXmlGenerator, saves the xml, and gets the display to reload it so that the user can see the changes visually.

  ```
  add_load():
  ```

Description:




