# Tutorial

This tutorial demonstrates how to use the classes in the repository.

## util.xml_generator.SceneXmlGenerator
```
from util import xml_generator
```

The simulation process starts with creating an xml model, in which the simulated objects are present, and which the simulator can load. If there is a pre-made model, using this class is not necessary.


The constructor of SceneXmlGenerator expects an xml file name. This file can contain default classes, meshes, textures, materials and geoms that have fixed positions throughout the simulation. In our case, this file is xml_models/scene.xml that contains drone- and building meshes, carpet textures etc. Initially, scene.xml is displayed when scripts/build_scene.py is run. SceneXmlGenerator will "include" this file in the generated xml like this:
```
<include file="../xml_models/scene.xml" />
```
so that everything it contains becomes accessible for the generated xml.

The SceneXmlGenerator class is necessary because MuJoCo does not yet support model creation programmatically. See MuJoCo documentation for details (https://mujoco.readthedocs.io/en/stable/overview.html).

Initialize SceneXmlGenerator like so:

```
xml_path = os.path.join("..", "xml_models")
xml_base_filename = "scene.xml"

scene = xml_generator.SceneXmlGenerator(os.path.join(xml_path, xml_base_filename))
```

To add buildings to the model:

```
# the inputs must be strings in a format that MuJoCo expects
# usually first is position and second is quaternion
# unless the method expects a name like add_pole()
# because multiple of them can be added to the model

scene.add_hospital("1 1 0", "1 0 0 0")
scene.add_sztaki("0 0 0", "1 0 0 0")
scene.add_post_office("0 1 0", "1 0 0 0")
scene.add_airport("1 0 0", "1 0 0 0")
scene.add_pole("pole0", "0.4 0.4 0", "1 0 0 0")
```

To add drones to the model:

```
position = "1 1 1"
orientation = "1 0 0 0"
color = "0.2 0.2 0.85 1.0"

# this will be a simulated (not a mocap), blue crazyflie drone
scene.add_drone(position, orientation, color, True, "crazyflie", False)

position = "1 0 1"
orientation = "1 0 0 0"
color = "0.2 0.85 0.2 1.0"

# this will be a simulated (not a mocap), green bumblebee drone with a hook (because last argument is True)
scene.add_drone(position, orientation, color, True, "bumblebee", True)


position = "0 1 1"
orientation = "1 0 0 0"
color = "0.85 0.2 0.2 1.0"

# this will be a mocap (because 4th argument is False), red bumblebee drone without a hook
scene.add_drone(position, orientation, color, False, "bumblebee", False)
```

To add a payload to the model:

```
# arguments are position, size, mass, quaternion and color
scene.add_load("0 -1 0", ".1 .1 .1", ".15", "1 0 0 0", "0.1 0.1 0.9 1.0")
```

To save the xml to hard disk:

```
save_filename = "built_scene.xml"

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))
```

## classes.active_simulation.ActiveSimulator
```
from classes.active_simulation import ActiveSimulator
```

This class wraps some of the simulation capabilities of MuJoCo and manages other stuff, see CodeDocumentation.md.

The constructor expects an xml file name, time intervals for video recording, time steps for updating control and graphics, and whether or not to connect to Motive server.

To initialize ActiveSimulator:

```
control_step, graphics_step = 0.01, 0.02 # these are in seconds

simulator = ActiveSimulator(os.path.join(xml_path, "built_scene.xml"), None, control_step, graphics_step, False)
```
At this point, a blank window should appear.

To access the list of simulated drones so that a trajectory and controllers can be assigned to each:

```
simulated_drones = simulator.virtdrones

# Controllers are in Peter Antal's repository
from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl
import classes.trajectory as traj

controller = GeomControl(simulator.model, simulator.data)
controllers = {"geom" : controller}

trajectory_flip = traj.TestTrajectory(control_step, traj.FLIP)

simulated_drones[0].set_controllers(controllers)
simulated_drones[0].set_trajectory(trajectory_flip)
```
The drones in this list are in the same order as they have been added to the xml model.

To run the simulator create an infinite loop with a "window should close" condition and call the simulator's update method with the incremented loop variable:

```
while not simulator.glfw_window_should_close():

    simulator.update(i)

    i += 1
```

The update method of each simulated object is called in the simulators update method.

## classes.mujoco_display.Display

```
from classes.mujoco_display import Display
import mujoco
import glfw
```
This is the base class for PassiveDisplay and ActiveSimulator. PassiveDisplay serves as a virtual mirror for real life demos, while ActiveSimulator can simulate virtual objects as well as display motion capture objects.

The responsibilities of Display:
  * wraps most of MuJoCo's required functionality
  * handles window event callbacks
  * initializes cameras and moves them on mouse events
  * loads, reloads MuJoCo models
  * provides methods for video recording

### Usage

Display by itself does not do any simulation or displaying. The child class should implement a method for rendering and simulation that's either called in a loop, or contains the loop. In ActiveSimulation this method is called update() which gets called in an infinite loop in a script, while in PassiveDisplay, it's called run() which contains an infinite loop. The reason for this difference is that the user might want to do extra processing of the data in the case of ActiveSimulator outside of the class.

Steps that need to be done when implementing an update() or run() method in a child class in order to get basic funcionality:

```
mujoco.mj_step(self.model, self.data, number_of_simulation_steps)

self.viewport = mujoco.MjrRect(0, 0, 0, 0)
self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
mujoco.mjv_updateScene(self.model, self.data, self.opt, pert=None, cam=self.activeCam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=self.scn)

mujoco.mjr_render(self.viewport, self.scn, self.con)

glfw.swap_buffers(self.window)
glfw.poll_events()
```

If displaying motion capture objects is needed (at the moment only drones have been implemented):

```
# getting data from optitrack server
if self.connect_to_optitrack:

    self.mc.waitForNextFrame()
    for name, obj in self.mc.rigidBodies.items():

        # have to put rotation.w to the front because the order is different
        drone_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

        drone_to_update = Drone.get_drone_by_name_in_motive(self.drones, name)

        if drone_to_update:
            drone_to_update.set_qpos(obj.position, drone_orientation)
```

## classes.drone.Drone, classes.drone.DroneHooked

```
from classes.drone import Drone, DroneHooked
import mujoco
```

To use these classes, an xml model needs to be created that contains drones that follow the naming convention in naming_convention_in_xml.txt.
The model needs to be loaded, then the parser - which is a static method of Drone - needs to be called. It returns a list of virtually simulated drones, that can contain Drone and DroneHooked instances:

```
model = mujoco.MjModel.from_xml_path(xmlFileName)
data = mujoco.MjData(model)

# each drone is searched in the model by its free joint name,
# therefore a list of joint names is passed to the parser
virtdrones = Drone.parse_drones(data, mujoco_helper.get_joint_name_list(model))
```

When a Drone or DroneHooked instance is created, all its required variables are mined out of the model's MjData (such as qpos, ctrl, qvel, sensor data), and class methods are provided to read or change these atributes in MjData. For example:
```
# in the constructor of Drone
free_joint = self.data.joint(self.name_in_xml)
self.qpos = free_joint.qpos
```
```
    # methods of Drone
    def get_qpos(self):
        return self.qpos
    
    def set_qpos(self, position, orientation):
        """
        orientation should be quaternion
        """
        self.qpos[:7] = np.append(position, orientation)
```

To assign controllers and trajectory to a drone:
```
from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl

import classes.trajectory as traj

controller = GeomControl(model, data, drone_type='large_quad')
controllers = {"geom" : controller}

control_step = 0.01
traj_fly = traj.TestTrajectory(control_step, traj.FLY)


virtdrones[0].set_trajectory(traj_fly)
virtdrones[0].set_controllers(controllers)
```

The trajectory's evaluate() and the controllers compute_control() methods are called in the Drone instance's update(), and the drone sets its newly computed control within its update() method.

## classes.drone.DroneMocap, classes.drone.DroneMocapHooked
```
from classes.drone import DroneMocap, DroneMocapHooked
import mujoco
```
To use these classes, an xml model needs to be created that contains mocap drones that follow the naming convention in naming_convention_in_xml.txt.
The model needs to be loaded, then the parser - which is a static method of DroneMocap - needs to be called. It returns a list of mocap drones, that can contain DroneMocap and DroneMocapHooked instances:

```
model = mujoco.MjModel.from_xml_path(xmlFileName)
data = mujoco.MjData(model)

# each drone is searched in the model by its body name,
# therefore a list of body names is passed to the parser
realdrones = DroneMocap.parse_mocap_drones(data, model, mujoco_helper.get_body_name_list(model))
```

Each mocap drone is made of separate mocap bodies in the MuJoCo model, to be able to have spinning propellers, because mocap bodies cannot have child mocap bodies. There are a mocap body for the drone body, four mocap bodies for the propellers, and optionally a mocap body for the hook.


When a DroneMocap or DroneMocapHooked instance is created, four PropellerMocap instances are created and saved as member variable for the four propellers:
```
# in the constructor of DroneMocap
self.prop1 = PropellerMocap(model, data, name_in_xml + "_prop1", drone_mocapid, SPIN_DIR.COUNTER_CLOCKWISE)
self.prop2 = PropellerMocap(model, data, name_in_xml + "_prop2", drone_mocapid, SPIN_DIR.COUNTER_CLOCKWISE)
self.prop3 = PropellerMocap(model, data, name_in_xml + "_prop3", drone_mocapid, SPIN_DIR.CLOCKWISE)
self.prop4 = PropellerMocap(model, data, name_in_xml + "_prop4", drone_mocapid, SPIN_DIR.CLOCKWISE)
```
DroneMocap provides methods for reading and changing its position and orientation.
