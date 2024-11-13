"""
Example to show how a mocap skeleton (specifically a hooked mocap drone) can be added to a scene.
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
xml_directory = os.path.join(project_root.resolve().as_posix(), "xml_models")
project_root = project_root.resolve().as_posix()

from aiml_virtual import scene, simulator
from aiml_virtual.simulated_object.mocap_skeleton.mocap_hooked_bumblebee import MocapHookedBumblebee2DOF
from aiml_virtual.mocap import optitrack_mocap_source

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "scene_base.xml"), save_filename="example_scene_9.xml")
    mocap = optitrack_mocap_source.OptitrackMocapSource()
    # One mocap object type that we haven't discussed is a mocap skeleton. Mujoco has strict limitations on what can
    # be a mocap object: they must be top-level bodies (direct children of the world body), and have no joints:
    # https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies
    # What happens if we have an object with joints that we want to display as a mocap object? Some examples include
    # the car with a trailer, and a bumblebee with a hook. To handle these objects, we use a MocapSkeleton.
    # MocapSkeletons are MocapObjects but with modified functionality: they contain several MocapObjects
    # These sub-objects appear individually in the xml file, but only the MocapSkeleton appears in the scene and the
    # simulator. In other words: MocapSkeletons are one object from the POV of the simulator, but multiple objects
    # form the POV mujoco.
    # Currently, it is not possible to identify an existing mocap skeleton in an xml file: they should be
    # created in one of the following ways:
    # 1: read automatically from mocap like so: scn.add_mocap_objects(mocap)
    # 2: instantiate -> add so scene, this is the method we'll use here
    # Note that each MocapSkeleton has a "parent" mocap body. You can view this as the first oject in the chain
    # of attached objects. Its mocap name will match the mocap name of the resulting mocap skeleton. In most mocap
    # skeletons this is fairly obvious: in the chain of a bumblebee and an attached hook, the bumblebee is the parent
    bb: MocapHookedBumblebee2DOF = MocapHookedBumblebee2DOF(mocap, "bb3")
    scn.add_object(bb)

    sim = simulator.Simulator(scn)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses