"""
This script demonstrates the usage of airflow samplers.
"""

import os
import sys
import pathlib
import numpy as np

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

from aiml_virtual import scene, simulator, airflow_luts_pressure, airflow_luts_velocity, xml_directory
from aiml_virtual.trajectory import dummy_drone_trajectory
from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import hooked_bumblebee
from aiml_virtual.airflow.airflow_sampler import SimpleAirflowSampler, ComplexAirflowSampler

if __name__ == "__main__":
    # In the aiml_virtual package, airflow simulation works the following way:
    # In the airflow subpackage, the airflow_target.py module defines an AirflowTarget abstract
    # base class, basically an interface. Objects may implement this interface by defining
    # an update method (dynamic objects already do this), and a get_rectangle_data method.
    # In the get_recangle_data method, an airflow target must return an AirFlowData object,
    # which defines all the data necessary to generate airflow forces and torques (for example
    # surface normals, their corresponding surface area, etc.). Based on this, an airflow sampler
    # can calculate the forces that a given drone will exert on the object via airflow. Given
    # the most typical usage of airflow simulation, by default the aiml_virtual package implements
    # the AirflowTarget interface on two classes: TeardropPayload and BoxPayload.

    # Here we create a scene the usual way, and add two bumblebee drones with hooks. These drones will each
    # generate air flow, which will perturb given objects.
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=f"example_scene_10.xml")
    bb1 = hooked_bumblebee.HookedBumblebee2DOF()
    bb1.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([0, 1, 2]))
    scn.add_object(bb1, "0 1 2", "1 0 0 0", "0.5 0.5 0.5 1")
    bb2 = hooked_bumblebee.HookedBumblebee2DOF()
    bb2.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([0, -1, 2]))
    scn.add_object(bb2, "0 -1 2", "1 0 0 0", "0.5 0.5 0.5 1")

    # Let's create one of each payload type, which both implement the AirflowTarget interface/abstract base class
    payload1 = dynamic_object.BoxPayload()
    scn.add_object(payload1, "0 1 1.32")
    payload2 = dynamic_object.TeardropPayload()
    scn.add_object(payload2, "0 -1 1.37")

    # There are two type of airflow samplers: SimpleAirflowSampler and ComplexAirflowSampler.
    # SimpleAirflowSamplers only use air pressure data and only at a given rotor speed. Their constructor needs the
    # name of the file which contains the pressure data at the given rotor speed. These files are found in the
    # airflow_luts_pressure folder (this folder's path is exported by the aiml_virtual package). ComplexAirflowSamplers
    # use both air pressure and air velocity data, and interpolate them at the appropriate rotor speed. Their
    # constructor requires the air pressure and air velocity files' folders, which are exported by the aiml_virtual pkg.
    airflowSampler1 = SimpleAirflowSampler(bb1, os.path.join(airflow_luts_pressure, "openfoam_pressure_2000.txt"))
    airflowSampler2 = ComplexAirflowSampler(bb2, airflow_luts_pressure, airflow_luts_velocity)

    # In order for the airflow to be applied to the AirflowTarget, the sampler shall be added to the target like so:
    payload1.add_airflow_sampler(airflowSampler1)
    payload2.add_airflow_sampler(airflowSampler2)

    sim = simulator.Simulator(scn)

    with sim.launch():
        # Note that calculating airflow forces requires some serious computing, so this simulation may run slow with
        # two payloads.
        while not sim.display_should_close():
            sim.tick()  # tick steps the simulator, including all its subprocesses



