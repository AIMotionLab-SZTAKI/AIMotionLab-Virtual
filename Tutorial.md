# Tutorial

This tutorial describes the concept of the simulator, and provides a demonstration to how the simulator needs to be constructed programmatically.

![class_diagram](https://user-images.githubusercontent.com/8826953/219118911-b0a9fa62-8df8-49c5-86db-a87f5957f0ab.png)

## Concept

The scene of a specific simulation is described in a .xml file which is a MuJoCo model. The model can contain static objects, simulated objects and mocap objects. A static object takes part in the collisions, but it is fixed, so its position stays the same throughout the simulation. Simulated objects can move around in the scene, collide with each other and with static- and mocap objects. Mocap objects are treated like static objects, but their position and orientation can be updated programmatically based on a motion capture system or prerecorded data etc.
The simulator instance keeps a list of "MovingObject" and a list of "MocapObject" instances that are the main characters of the simulation. These are the simulated- and the mocap objects. Each MovingObject has an update() method in which its trajectory and control are calculated. Each MocapObject has an update() method in which its position and orientation can be updated. The simulator calls these update methods repeatedly at equal time intervals. This time interval is called control timestep (simulator.control_step).

### Steps of the simulation

1. Generate a MuJoCo xml model with the objects required for the simulation. The simulated- and mocap objects need to follow a naming convention, so that the parsers can dig them out of the model.
2. For each simuated- and mocap object type, create a parser method (such as Drone.parse), that can recognize the objects in the MuJoCo model based on the naming convention, and generate the python "wrapper" object that corresponds to the MuJoCo model object. The parser method must return a list of objects of a particular kind so that the simulator can add them to its own list of simulated and mocap objects.
3. Pass these parser methods to the simulator instance. The simulator calls each of these parsers, and strings the returned lists of objects together for updating each object later during the simulation.
4. Go through the generated list of simulated python objects, and assign them a controller and/or a trajectory if either is necessary for the simulation.
5. The controller and trajectory classes also need to implement a sort of interface, because their compute_control() and evaluate() methods are called with fixed inputs in the owning object's update method.
6. During the simulation, the simulator goes through its list of simulated objects at each control step, and calls their update() method, that computes their new control based on their trajectory. If motioncapture is enabled, the simulator also goes through its list of mocap objects at each control step, and updates their position and orientation if necessary.

#### If the simulation of a new kind of object is needed

1. Come up with a naming convention for the object and its parts (bodies, joints, geoms, actuators, etc... whatever is needed). (See naming_convention_in_xml.txt for examples)
2. Create a python wrapper class that contains all the needed functionality for simulation, and that implents the MovingObject base class, i.e. has an update method that will be called at every control step to compute new control and do whatever else is necessary.
3. Create a parser method, that can generate the python instances based on the loaded MuJoCo model. Add the parser method to the list that is passed to the simulators constructor so that it can be called inside the simulator. The inputs of the parser method must be MjData and MjModel of the loaded MuJoCo model. It must return a list of the generated instances of the new object type.
4. The simulator will treat this new object type like a MovingObject, therefore it only knows that the object has an update() method.

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
drone0_name = scene.add_drone(position, orientation, color, True, "crazyflie", False)

position = "1 0 1"
orientation = "1 0 0 0"
color = "0.2 0.85 0.2 1.0"

# this will be a simulated (not a mocap), green bumblebee drone with a hook (because last argument is True)
drone1_name = scene.add_drone(position, orientation, color, True, "bumblebee", True)


position = "0 1 1"
orientation = "1 0 0 0"
color = "0.85 0.2 0.2 1.0"

# this will be a mocap (because 4th argument is False), red bumblebee drone without a hook
drone2_name = scene.add_drone(position, orientation, color, False, "bumblebee", False)
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
from classes.drone import Drone, DroneMocap
from classes.car import Car, CarMocap
```

This class wraps some of the simulation capabilities of MuJoCo and manages other stuff, see CodeDocumentation.md.

The constructor expects an xml file name, time intervals for video recording, time steps for updating control and graphics, a list of virtual object parser methods, a list of mocap object parser methods, and whether or not to connect to Motive server.

To initialize ActiveSimulator:

```
control_step, graphics_step = 0.01, 0.02 # these are in seconds

virt_parsers = [Drone.parse, Car.parse]
mocap_parsers = [DroneMocap.parse, CarMocap.parse]

simulator = ActiveSimulator(os.path.join(xml_path, save_filename), None, control_step, graphics_step, virt_parsers, mocap_parsers, False)
```
At this point, a blank window should appear.

To access the list of simulated objects so that a trajectory and controllers can be assigned to each:

```

# Controllers are in Peter Antal's repository
from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl

import classes.trajectory as traj

controller = GeomControl(simulator.model, simulator.data)
controllers = {"geom" : controller}

trajectory_flip = traj.TestTrajectory(control_step, traj.FLIP)

drone = simulator.get_MovingObject_by_name_in_xml(drone0_name)
drone.set_controllers(controllers)
drone.set_trajectory(trajectory_flip)
```
The vehicles in this list are in the same order as they have been added to the xml model.

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

If displaying motion capture objects is needed (at the moment only drones and cars have been implemented):

```
# getting data from optitrack server
if self.connect_to_optitrack:

    self.mc.waitForNextFrame()
    for name, obj in self.mc.rigidBodies.items():

        # have to put rotation.w to the front because the order is different
        # only update real vehicles
        vehicle_orientation = [obj.rotation.w, obj.rotation.x, obj.rotation.y, obj.rotation.z]

        vehicle_to_update = MovingMocapObject.get_object_by_name_in_motive(self.all_real_vehicles, name)

        if vehicle_to_update is not None:
            vehicle_to_update.update(obj.position, vehicle_orientation)
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
virtdrones = Drone.parse(data, model)
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
realdrones = DroneMocap.parse(data, model, model)
```

The propellers are attached to the mocap drone with a joint that has one degree of freedom for spinning. If an angular velocity is set at this joint, the propellers will spin indefinitely, unless some collision happens. Therefore it's worth resetting this veocity from time to time.

DroneMocap provides methods for reading and changing its position and orientation.

## classes.trajectory.Trajectory
```
import classes.trajectory.Trajectory
```
This is the base class or "interface" for trajectory classes. All trajectories must implement the evaluate method, because this is what the drone calls in its update method. The evaluate method must uodate return a dictionary (self.output) that's been initialized in the constuctor of the base class.

The controller_name in the dictionary determines which controller the drone will use for that part of the trajectory.

TestTrajectory in classes/trajectory.py is based on Peter Antal's code, and demonstrates how to use the Trajectory base class.
