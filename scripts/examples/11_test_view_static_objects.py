"""
This script shows how dynamic objects work.
"""

import os
import sys
import pathlib
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

# Add project paths to sys.path for imports
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())
while "aiml_virtual" not in [f.name for f in project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())

import aiml_virtual

xml_directory = aiml_virtual.xml_directory
from aiml_virtual import scene, simulator


def update_include_path(xml_path_to_edit: Path, new_include_path: Path):
    tree = ET.parse(xml_path_to_edit)
    root = tree.getroot()

    includes = root.findall("include")

    includes[1].set("file", str(new_include_path))
    tree.write(xml_path_to_edit, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the include path in the XML and run simulation.")
    parser.add_argument("--xml_path", required=True, help="Path to the static_objects.xml file to include")
    args = parser.parse_args()

    static_xml_path = Path(args.xml_path)
    if not static_xml_path.is_absolute():
        static_xml_path = Path.cwd() / static_xml_path
    static_xml_path = static_xml_path.resolve()

    scene_xml = Path(os.path.join(xml_directory, "scene_base_with_static_objects.xml"))
    update_include_path(scene_xml, static_xml_path)

    scn = scene.Scene(str(scene_xml), save_filename="example_scene_3.xml")
    sim = simulator.Simulator(scn)
    with sim.launch():
        while not sim.display_should_close():
            sim.tick()
