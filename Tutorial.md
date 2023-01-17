# Tutorial

This tutorial demonstrates how to use the classes in the repository.

## util.xml_generator.SceneXmlGenerator

The simulation process starts with creating an xml model, in which the simulated objects are present, and which the simulator can load. If there is already a pre-made model, using this class is not necessary.


The constructor of SceneXmlGenerator expects an xml file name. This file can contain default classes, meshes, textures, materials and geoms that have fixed positions throughout the simulation. In our case, this file is xml_models/scene.xml that contains drone- and building meshes, carpet textures etc. SceneXmlGenerator will "include" this file in the generated xml. This class is necessary because MuJoCo does not yet support model creation programmatically. See MuJoCo documentation for details (https://mujoco.readthedocs.io/en/stable/overview.html).

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
scene.add_load("0 0 0", ".1 .1 .1", ".15", "1 0 0 0", "0.1 0.1 0.9 1.0")
```

To save the xml to hard disk:

```
save_filename = "built_scene.xml"

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))
```

## classes.active_simulation.ActiveSimulator

