"""
This script shows how to modify and build a scene from a mostly empty base scene.
"""

import os
import sys
import pathlib

# The lines under here are intended to make sure imports work, by adding parent folders to the path (i.e. the list
# of folders where the interpreter will look for a given package when you try to import it). This is to account for
# differences in what the interpreter identifies as your current working directory when launching these scripts
# from the command line as regular scripts vs with the -m option vs from PyCharm, as well as the script being placed
# in any depth of sub-sub-subfolder.
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())

import aiml_virtual
xml_directory = aiml_virtual.xml_directory
from aiml_virtual import scene, simulator
from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.simulated_object.mocap_object import mocap_object

if __name__ == "__main__":
    # Often we don't just read a scene from a file and roll with it. A common use case is reading the base of a scene
    # from a file, and then adding objects to it in a python script. "scene_base.xml" is the SZTAKI Lagyi 6th floor IDT
    scn = scene.Scene(os.path.join(xml_directory, "scene_base.xml"), save_filename=f"example_scene_2.xml")
    # Before adding objects to the scene, let's discuss what objects we can choose from. (a) signals that a class is
    # abstract, meaning that only its descendants may be initialized.
    # SimulatedObject(a): objects that appear both in the python script and in the simulation. There are two types
    # of SimulatedObject: MocapObjects(a) that get their pose data from a mocap system and DynamicObjects(a) that get
    # their pose from MuJoCo calculations. DynamicObjects may also have controllers and actuators, in which case they
    # are ControlledObject(a).
    # An example of a DynamicObject (an object which is subject to MuJoCo physics) is a payload.
    # This payload is 'passive' in that it has no actuators, as opposed to controlled objects, which we will discuss
    # later.
    payload1 = dynamic_object.BoxPayload()
    # Let's add this payload to the scene! We can give it a starting position and starting orientation.
    scn.add_object(payload1, pos="1 0 2.5", quat="1 0 0 0")
    # There is an important thing to note here. Whenever you modify the scene, it saves its state to an xml file
    # determined by save_filename.
    # An example of a MocapObject would be a *mocap* payload:
    payload2 = mocap_object.MocapPayload()
    # Let's add this payload to the scene as well!
    scn.add_object(payload2, pos="0 0 0.5", quat="0 1 0 0")
    # The key difference between the two payloads is the following: one of them is a dynamic object, subject to
    # gravity, the other is a mocap object (although we currently don't provide it any mocap data yet). This means
    # that the former will drop from the sky when the simulation is launched, whereas the latter will stay in the air.
    # Read here about mocap objects in mujoco: https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies
    sim = simulator.Simulator(scn)
    with sim.launch(fps=20, speed=0.2):  # demonstrate lower fps as well as a slower simulation speed
        sim.visualizer.mjvCamera.distance = 4  # demonstrate positioning the camera at a different distance
        while not sim.display_should_close():
            sim.tick()  # tick steps the simulator, including all its subprocesses