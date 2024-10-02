"""
This script shows how to modify and build a scene from a base.
"""

import os
import sys
import pathlib

# make sure imports work by adding the necessary folders to the path:
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())
xml_directory = os.path.join(project_root.resolve().as_posix(), "xml_models")

from aiml_virtual import scene, simulator
from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.simulated_object.mocap_object import mocap_object

if __name__ == "__main__":
    # Often we don't just read a scene from a file and roll with it. A common use case is reading the base of a scene
    # from a file, and then adding objects to it from a python script. "scene_base.xml" is the SZTAKI Lagyi 6th IDT
    scn = scene.Scene(os.path.join(xml_directory, "scene_base.xml"), save_filename=f"example_scene_2.xml")
    # Before adding objects to the scene, let's discuss what objects we can choose from. (a) signals that a class is
    # abstract, meaning that only its descendants may be initialized.
    # SimulatedObject(a): objects that appear both in the python script and in the simulation. There are two types
    # of SimulatedObject: MocapObjects(a) that get their pose data from a mocap system and DynamicObjects(a) that get
    # their pose from MuJoCo calculations. DynamicObjects may also have controllers and actuators, in which case they
    # are ControlledObject(a).
    # An example of a DynamicObject (an object which is subject to MuJoCo physics but is not actuated) is a payload:
    payload1 = dynamic_object.DynamicPayload()
    # Let's add this payload to the scene!
    scn.add_object(payload1, pos="1 0 1.5", quat="1 0 0 0")
    # There is an important thing to note here. Whenever you modify the scene, it saves its state to an xml.
    # An example of a MocapObject would be a *mocap* payload:
    payload2 = mocap_object.MocapPayload()
    # Let's add this payload to the scene as well!
    scn.add_object(payload2, pos="0 0 0.5", quat="0 1 0 0")
    # The key difference between the two payloads is the following: one of them is a dynamic object, subject to
    # gravity, the other is a mocap object (although we currently don't provide it any mocap data yet). This means
    # that the former will drop from the sky when the simulation is launched, whereas the latter will stay in the air.
    sim = simulator.Simulator(scn)
    with sim.launch(fps=20):  # demonstrate lower fps as well
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses