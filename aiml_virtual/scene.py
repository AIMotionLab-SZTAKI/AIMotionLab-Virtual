import os.path
import sys
import xml.etree.ElementTree as ET
from typing import Type
import mujoco
import pathlib

from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.simulated_object.moving_object import moving_object

# importing whole modules is supposed to be good practice to avoid circular imports, but I want easier access, so here
# are some aliases to avoid having to type out module.symbol
MocapObject = mocap_object.MocapObject
MovingObject = moving_object.MovingObject
SimulatedObject = simulated_object.SimulatedObject

XML_FOLDER = os.path.join(pathlib.Path(__file__).parents[1].resolve().as_posix(), "xml_models")
EMTPY_SCENE = os.path.join(pathlib.Path(__file__).parents[1].resolve().as_posix(), "xml_models", "empty_scene.xml")

# TODO: make singleton?
class Scene:
    # Scene contains all the data relating to the mjModel, in a way where it can be exposed to Python
    # It has an ElementTree for the XML representation for the model, for saving later
    # It has a reference to the mjModel itself
    # It has a list of all the model elements that have a python object representing them (SimulatedObjects)
    # It is the responsibility of this class to keep these representations consistent with one another
    def __init__(self, base_scene_filename: str = EMTPY_SCENE, save_filename: str = os.path.join(XML_FOLDER, "Scene.xml")):
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(base_scene_filename)
        self.simulated_objects: list[SimulatedObject] = Scene.parse_simulated_objects(self.model)
        self.bind_to_model()
        self.xml_root = ET.parse(base_scene_filename).getroot()
        self.xml_name = save_filename

    def reload_model(self):
        # Sync the mujoco model to the xml. No need to have a separate save_xml function since this saves it as well.
        # Note that
        tree = ET.ElementTree(self.xml_root)
        if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
            ET.indent(tree, space="\t", level=0)
        tree.write(self.xml_name)
        self.model = mujoco.MjModel.from_xml_path(self.xml_name)
        # we want to preserve the references in self.simulated_objects, therefore, instead of reassigning
        # self.simulated_objects, we only re-bind them to the model (this is the reason why there is a separate bind,
        # instead of the bind being part of the constructor
        self.bind_to_model()

    def bind_to_model(self):
        # in order to preserve references, whenever the model is reloaded, the simulated objects aren't re-initialized,
        # instead, their only their reference to the model is reset whenever the model is reset
        for obj in self.simulated_objects:
            obj.bind_to_model(self.model)

    @staticmethod
    def parse_simulated_objects(model: mujoco.MjModel) -> list[SimulatedObject]:
        # this function is to be used when first loading a model, and we have no idea how many simulated objects are in
        # it, as opposed to reloading a model, when we know how many objects there are, and we're just updating the
        # MjModel instance
        simulated_objects: list[SimulatedObject] = []  # here is where we store the future returns
        for i in range(model.nbody):  # let's check all the bodies in the model
            body = model.body(i)
            for sim_name in SimulatedObject.xml_registry.keys():
                # a body is candidate to be a SimulatedObject, if its parent is the worldbody (meaning that its parent
                # id is 0), and its name contains one of the keys which identify Simulated Objects
                if body.parentid[0] == 0 and sim_name in body.name:
                    # this is the class to be instantiated, identified by the registry of the SimulatedObject class
                    cls: Type[SimulatedObject] = SimulatedObject.xml_registry[sim_name]
                    simulated_objects.append(cls())
        return simulated_objects

    def print_elements(self):
        """
        TODO
        """
        for elem in self.xml_root.iter():
            print(f"tag: {elem.tag}, attrib: {elem.attrib}")

    def add_object(self, obj: SimulatedObject, pos: str, quat: str, color: str) -> None:
        """
        TODO

        It makes sense that adding an object to the scene is done via a method of scene itself. However, for each
        object, this may look different. Most notably, their xml model is obviously different. Therefore, making the
        xml model should be the method of the object. This presents a bit of a pickle: each object must make an xml
        model which shall be appended to worldbody, however, they may also want any number of actuators/sensors, in
        other words, we may want to add to any of our grouping elements. Adding to these is the purview of the scene,
        but the object should decide where to add: this is a conflict of responsibilities

        currently this is resolved as below, but subject to change
        """
        # update XML representation
        xml_dict = obj.create_xml_element(pos, quat, color)
        for child in self.xml_root:
            if child.tag in xml_dict.keys():
                for e in xml_dict[child.tag]:
                    child.append(e)

        # update simulated objects
        self.simulated_objects.append(obj)

        # update model
        self.reload_model()
        


