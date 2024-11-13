"""
As opposed to dynamic cars, currently the trailer and the car are separate mocap objects. This
example is meant to demonstrate that, as well as the usage of the Optitrack system as a mocap source.
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
from aiml_virtual.simulated_object.mocap_object.mocap_object import Hospital, Sztaki, PostOffice
from aiml_virtual.mocap import optitrack_mocap_source

if __name__ == "__main__":
    # So far we've only used a dummy motion capture system. In the IDT, we use optitrack. Let's demonstrate its usage
    scn = scene.Scene(os.path.join(xml_directory, "example_scene_8.xml"), save_filename="example_scene_8.xml")
    # In this scene we loaded there should be 3 mocap objects, which are the 3 'buildings' we use for demos. Please
    # make sure that:
    # - these buildings are now in the frame of the optitrack system
    # - the buildings ar ticked on in the optitrack system
    # - the optitrack system is streaming to the correct IP address (network 2)
    # Whereas the dummy mocap source needed a bit of setup, the optitrack system is plug-and-play on the simulator side.
    # All you need to do is make an instance of it, and the constructor connects to it automatically.
    mocap = optitrack_mocap_source.OptitrackMocapSource()
    # We could add mocap objects the same way we did in example 4, but this time, the xml file already contained the
    # buildings. This is to showcase the following behavior:
    # You can initialize mocap objects without a mocap source and mocap name. The same happens when mocap objects
    # are initialized from a mjcf xml file: their mocap properties are uninitialized. You need to bind them to the
    # mocap source:
    hospital: Hospital = next((obj for obj in scn.simulated_objects if obj.name.startswith("Hospital")), Hospital())
    # from now on, hospital shall check mocap for its pose data, looking for the name "bu11"
    hospital.assign_mocap(mocap, "bu11")

    sztaki: Sztaki = next((obj for obj in scn.simulated_objects if obj.name.startswith("Sztaki")), Sztaki())
    # from now on, sztaki shall check mocap for its pose data, looking for the name "bu11"
    sztaki.assign_mocap(mocap, "bu14")

    post_office: PostOffice = next((obj for obj in scn.simulated_objects if obj.name.startswith("PostOffice")), PostOffice())
    # from now on, post_office shall check mocap for its pose data, looking for the name "bu13"
    post_office.assign_mocap(mocap, "bu13")

    sim = simulator.Simulator(scn)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses