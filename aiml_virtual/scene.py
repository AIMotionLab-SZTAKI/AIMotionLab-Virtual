"""
This module contains the class Scene.
"""

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


class Scene:
    """
    This class contains all the data relating to the mjModel, in a way where it can be exposed to Python. Its
    responsibility is keeping all the following representations of a scene consistent:

    - ElementTree for the XML representation of the model (also called MJCF)
    - The mjModel as loaded by mujoco.MjModel.from_xml_path
    - A list of all the SimulatedObjects in the scene

    .. note::
        The list of simulated objects must be handled with care: the Scene is not the only entity that may have a
        reference to them. A simple example for this:

        .. code-block:: python

            scene = Scene()
            cf = Crazyflie()
            cf.do_something()  # set some variable or call a method to modify the crazyflie
            scene.add_object(cf)
            scene.do_something_that_requires_reloading_the_xml

        After this code segment, if we re-generated the objects when reloading the model, the outer script's reference
        "cf" would be lost. To this end, SimulatedObjects may exist without a model or data. However, they must be bound
        to the model and the data before running a simulation.
    """
    def __init__(self, base_scene_filename: str = EMTPY_SCENE, save_filename: str = os.path.join(XML_FOLDER, "Scene.xml")):
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(base_scene_filename)  #: the mjModel with C bindings
        self.xml_root: ET.Element = ET.parse(base_scene_filename).getroot()  #: root of the XML tree
        self.xml_name: str = save_filename  #: the name under which we will save the new XML

        self.simulated_objects: list[SimulatedObject] = []  #: the objects with python interface
        for i in range(self.model.nbody):  # let's check all the bodies in the model
            body = self.model.body(i)
            for sim_name in SimulatedObject.xml_registry.keys():
                # a body is candidate to be a SimulatedObject, if its parent is the worldbody (meaning that its parent
                # id is 0), and its name contains one of the keys which identify Simulated Objects
                if body.parentid[0] == 0 and sim_name in body.name:
                    # this is the class to be instantiated, identified by the registry of the SimulatedObject class
                    cls: Type[SimulatedObject] = SimulatedObject.xml_registry[sim_name]
                    self.simulated_objects.append(cls())
        self.bind_to_model()

    def reload_model(self) -> None:
        """
        Syncs the mujoco model to the xml. Must be called when objects are added or removed to any of the
        representations. Also saves the XML file.

        .. note::
            Whenever we reload the model, the references stored in self.simulated_objects become obsolete. We have to
            re-assign them using bind_to_model.
        """
        tree = ET.ElementTree(self.xml_root)
        if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
            ET.indent(tree, space="\t", level=0)
        tree.write(self.xml_name)
        self.model = mujoco.MjModel.from_xml_path(self.xml_name)
        # we want to preserve the references in self.simulated_objects, therefore, instead of reassigning
        # self.simulated_objects, we only re-bind them to the model (this is the reason why there is a separate bind,
        # instead of the bind being part of the constructor
        self.bind_to_model()

    def bind_to_model(self) -> None:
        """
        Re-binds all the objects to the model. In order to preserve references in the script using the Scene, whenever
        the model is reloaded, the simulated objects aren't re-initialized, instead, only their reference to the
        model is reset.
        """

        for obj in self.simulated_objects:
            obj.bind_to_model(self.model)


    def add_object(self, obj: SimulatedObject, pos: str = "0 0 0", quat: str = "1 0 0 0", color: str = "0.5 0.5 0.5 1") \
            -> None:
        """
        Adds an object to the Scene, with the specified position, orientation and base color.

        Args:
            obj (SimulatedObject): The object to add. As it's a SimulatedObject, it handles its own xml representation.
            pos (str): The position of the object, as a string where x y and z are separated by spaces.
            quat (str): The orientation quaternion of the object, as a space-separated string in "w x y z" order.
            color (str): The base color of the object, as a space separated string in "r g b a" order.
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
        


