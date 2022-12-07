from dataclasses import dataclass
import numpy as np
import os
from util import xml_generator
from classes.passive_display import PassiveDisplay
from gui.building_input_gui import BuildingInputGui
from gui.drone_input_gui import DroneInputGui
from gui.cargo_input_gui import CargoInputGui


# open the base on which we'll build
xml_path = os.path.join("..", "xml_models")
xmlBaseFileName = "scene.xml"
save_filename = "built_scene_3loads.xml"

scene = xml_generator.SceneXmlGenerator(os.path.join(xml_path, xmlBaseFileName))
RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"



def main():
    scene.add_drone("1 1 1", "1 0 0 0", RED_COLOR, True, "bumblebee", True)
    scene.add_load("0 0 0", ".1 .1 .1", ".15", "1 0 0 0", "0.1 0.1 0.9 1.0")
    scene.add_load("-.6 .6 0", ".075 .075 .075", ".05", "1 0 0 0", "0.1 0.9 0.1 1.0")
    scene.add_load("-.3 -.6 0", ".075 .075 .1", ".1", "1 0 0 0", "0.9 0.1 0.1 1.0")
    scene.save_xml(os.path.join(xml_path, save_filename))
    print("done")

if __name__ == '__main__':
    main()
