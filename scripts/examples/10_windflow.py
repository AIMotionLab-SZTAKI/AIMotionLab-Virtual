"""
This script shows how dynamic objects work.
"""

import os
import sys
import pathlib
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

# The lines under here are intended to make sure imports work, by adding parent folders to the path (i.e. the list
# of folders where the interpreter will look for a given package when you try to import it). This is to account for
# differences in what the interpreter identifies as your current working directory when launching these scripts
# from the command line as regular scripts vs with the -m option vs from PyCharm, as well as the script being placed
# in any depth of sub-sub-subfolder.
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())

import aiml_virtual

xml_directory = aiml_virtual.xml_directory
from aiml_virtual import scene, simulator
from aiml_virtual.trajectory import dummy_drone_trajectory, skyc_trajectory
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import crazyflie
from pathlib import Path


def create_traj():
    from skyc_utils import skyc_maker, skyc_inspector
    traj = skyc_maker.Trajectory(degree=7)

    traj.set_start(skyc_maker.XYZYaw(0.55, 1.05, 1.75, 0))
    traj.add_goto(skyc_maker.XYZYaw(-1.45, 1.05, 1.75, 0), 5)
    traj.add_goto(skyc_maker.XYZYaw(-1.45, -0.95, 1.75, 0), 5)
    traj.add_goto(skyc_maker.XYZYaw(0.55, -0.95, 1.75, 0), 5)
    traj.add_goto(skyc_maker.XYZYaw(0.55, 1.05, 1.75, 0), 5)

    traj.add_goto(skyc_maker.XYZYaw(-0.45, 1.05, 1.35, 0), 5)
    traj.add_goto(skyc_maker.XYZYaw(-1.45, 0.05, 1.35, 0), 5)
    traj.add_goto(skyc_maker.XYZYaw(-0.45, -0.95, 1.35, 0), 5)
    traj.add_goto(skyc_maker.XYZYaw(0.55, 0.05, 1.35, 0), 5)
    traj.add_goto(skyc_maker.XYZYaw(0.55, 1.05, 1.75, 0), 5)

    skyc_maker.write_skyc([traj])


def update_include_path(xml_path_to_edit: Path, new_include_path: Path):
    tree = ET.parse(xml_path_to_edit)
    root = tree.getroot()

    includes = root.findall("include")

    includes[1].set("file", str(new_include_path))
    tree.write(xml_path_to_edit, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with custom windflow data.")
    parser.add_argument("--xml_path", required=True, help="Path to the static_objects.xml file to include")
    parser.add_argument("--csv_path", required=True, help="Path to the windflow data CSV file")
    args = parser.parse_args()

    static_xml_path = Path(args.xml_path)
    if not static_xml_path.is_absolute():
        static_xml_path = Path.cwd() / static_xml_path
    static_xml_path = static_xml_path.resolve()

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / Path("../windflow") / csv_path
    csv_path = csv_path.resolve()

    scene_xml = Path(os.path.join(xml_directory, "scene_base_with_static_objects.xml"))
    update_include_path(scene_xml, static_xml_path)

    # As mentioned in 2_build_scene.py, we can simulate physics using DynamicObjects. So far we've only seen a dynamic
    # object that had no actuators. Let's change that, and build a scene with dynamic objects based on the empty
    # checkerboard scene base!
    scn = scene.Scene(os.path.join(xml_directory, "scene_base_with_static_objects.xml"),
                      save_filename=f"example_scene_3.xml")

    # The dummy trajectory may have seemed a bit boring, even with the disturbance. A more interesting trajectory type
    # is read from a skyc file. An example skyc file is found under scripts/misc/skyc_example.skyc
    cf = crazyflie.Crazyflie()
    cf.set_windflow_data(csv_path)

    create_traj()
    traj = skyc_trajectory.extract_trajectories(os.path.join(aiml_virtual.pkg_dir, "../scripts/examples", str(
        os.path.splitext(os.path.basename(__file__))[0] + ".skyc")))[0]
    traj.set_start(0)

    cf.trajectory = traj
    scn.add_object(cf, "0.55 1.05 1.75", "1 0 0 0", "0.5 0.5 0.5 1")

    sim = simulator.Simulator(scn)
    with sim.launch():
        while not sim.display_should_close():
            sim.tick()  # tick steps the simulator, including all its subprocesses
