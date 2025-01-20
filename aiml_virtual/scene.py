"""
This module contains the class Scene.
"""

import os.path
import sys
import time
import xml.etree.ElementTree as ET
from typing import Type, cast, Union, Optional
import mujoco
import pathlib

import aiml_virtual
from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.simulated_object.mocap_skeleton import mocap_skeleton
from aiml_virtual.simulated_object.dynamic_object.controlled_object import controlled_object
from aiml_virtual.mocap import mocap_source
from aiml_virtual.utils import utils_general

# importing whole modules is supposed to be good practice to avoid circular imports, but I want easier access, so here
# are some aliases to avoid having to type out module.symbol
MocapObject = mocap_object.MocapObject
ControlledObject = controlled_object.ControlledObject
SimulatedObject = simulated_object.SimulatedObject
MocapSource = mocap_source.MocapSource
MocapSkeleton = mocap_skeleton.MocapSkeleton
warning = utils_general.warning

XML_FOLDER = aiml_virtual.xml_directory
EMTPY_SCENE = os.path.join(XML_FOLDER, "empty_scene.xml")

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
    def __init__(self, base_scene_filename: str = EMTPY_SCENE, save_filename: str = "Scene.xml"):
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(base_scene_filename)  #: the mjModel with C bindings
        self.xml_root: ET.Element = ET.Element("mujoco", {"model": save_filename}) #: root of the XML tree
        # include the base xml in the scene when we save it
        ET.SubElement(self.xml_root, "include", {"file": base_scene_filename})
        self.xml_name: str = save_filename  #: the name under which we will save the new XML

        self.simulated_objects: list[SimulatedObject] = []  #: the objects with python interface
        for i in range(self.model.nbody):  # let's check all the bodies in the model
            body = self.model.body(i)
            # a body is candidate to be a SimulatedObject, if its parent is the worldbody (meaning that its parent
            # id is 0), and its name contains one of the keys which identify Simulated Objects
            # note that instead of checking parents of the worldbody, we used to check for free joints, but a body
            # may be a valid body in the simulation without having a free joint (mocap objects for example)
            if body.parentid[0] == 0:
                # this is the class to be instantiated, identified by the registry of the SimulatedObject class:
                cls: Optional[Type[SimulatedObject]] = SimulatedObject.name_to_class(body.name)
                if cls is not None:
                    self.simulated_objects.append(cls())
        self.bind_to_model()

    def save_mjcf(self):
        """
        Saves the mjcf (xml) representation of the model to the file identified by xml_root.
        """
        tree = ET.ElementTree(self.xml_root)
        if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
            ET.indent(tree, space="\t", level=0)
        tree.write(self.xml_name)

    def reload_model(self) -> None:
        """
        Syncs the mujoco model to the xml. Must be called when objects are added or removed to any of the
        representations. Also saves the XML file.

        .. note::
            Whenever we reload the model, the references stored in self.simulated_objects become obsolete. We have to
            re-assign them using bind_to_model.
        """
        self.save_mjcf()
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
        Adds an object to the Scene, with the specified position, orientation and base color. The simplest way to modify
        the underlying mujoco model is by modifying the xml file, and reloading it. To this end, adding an object is
        achieved in three steps (which achieve the consistency between the three representations):

        - Adds the reference to the object to the simulated_objects list
        - Adds the object's xml data to the xml representation
        - Saves the xml file and reloads the model from it.

        Args:
            obj (SimulatedObject): The object to add. As it's a SimulatedObject, it handles its own xml representation.
            pos (str): The position of the object, as a string where x y and z are separated by spaces.
            quat (str): The orientation quaternion of the object, as a space-separated string in "w x y z" order.
            color (str): The base color of the object, as a space separated string in "r g b a" order.
        """
        # update XML representation
        if obj in self.simulated_objects or obj.name in [o.name for o in self.simulated_objects]:
            warning(f"Simulated Object {obj.name} is already in the scene!")
        else:
            xml_dict = obj.create_xml_element(pos, quat, color)
            for xml_tag, xml_elements in xml_dict.items():
                # if the xml tag we want to add to already exists, we just need to add our elements to it
                if xml_tag in [child.tag for child in self.xml_root]:
                    for child in self.xml_root:
                        if child.tag == xml_tag: # check which tag matches ours
                            # and add our elements to it
                            for xml_element in xml_elements:
                                child.append(xml_element)
                else: # however, if it doesn't exist, we must create it
                    new_element = ET.SubElement(self.xml_root, xml_tag)
                    for xml_element in xml_elements:
                        new_element.append(xml_element)

            # update simulated objects
            self.simulated_objects.append(obj)

            # update model
            self.reload_model()

    def remove_object(self, obj: Union[str, SimulatedObject]) -> None:
        """
        Analogous to add_object, except it removes a specified object from the scene.

        .. note::
            Only bodies added to the current scene can be removed. That is to say, a base scene (on top of which we
            build the current scene) cannot be modified by removing objects from it.

        Args:
            obj (Union[str, SimulatedObject]): The object to remove, either by reference or by name.
        """
        # Do nothing if the object is not in the scene. Note that this works regardless of whether obj was a string
        # (the name of the target object), or a reference to the object itself.
        if obj not in self.simulated_objects and obj not in [o.name for o in self.simulated_objects]:
            warning(f"Object {obj} not in scene. Available objects: {[o.name for o in self.simulated_objects]}")
            return
        if isinstance(obj, SimulatedObject):
            obj_to_remove = obj
            obj_name = obj.name
        else:
            obj_to_remove = [o for o in self.simulated_objects if o.name == obj][0]
            obj_name = obj
        self.simulated_objects.remove(obj_to_remove)  # python object needs to be removed
        SimulatedObject.instance_count[type(obj_to_remove)] -= 1  # counter needs to be decremented
        # xml element needs to be removed
        worldbody = self.xml_root.findall(".//worldbody")
        if len(worldbody) == 0:
            return
        else:
            worldbody = worldbody[0]
        for body in worldbody:
            if body.attrib.get('name') == obj_name:
                worldbody.remove(body)
        self.reload_model()  # update model to match XML

    def add_mocap_objects(self, mocap: MocapSource, color: str = "0.5 0.5 0.5 1") -> list[MocapObject]:
        """
        Adds every mocap object found in a motion capture source to the model. As opposed to add_object, this method
        actually creates new object instances.

        .. note::
            A mocap object can be added one of two ways: either directly by using add_object, or using this method. A
            mocap object must always have exactly one mocap source associated with it. This means that when this method
            ges called, it's possible that there already are mocap objects in the scene that are using this mocap
            source. These mocap objecs will be removed from the scene.

        Args:
            mocap (MocapSource): The motion capture source to parse for objects.
            color (str): The base color to use for adding the objects.

        Returns:
            list[MocapObject]: A list of references to the newly created objects.
        """
        # Identify objects already associated with this mocap source. Note: isinstance checks for subclasses as well.
        objects_to_remove = [o for o in self.simulated_objects if isinstance(o, MocapObject) and o.source == mocap]
        for object_to_remove in objects_to_remove:
            self.remove_object(object_to_remove)
        while len(mocap.data) == 0:
            time.sleep(0.1)
        frame = mocap.data  # a dictionary where each items corresponds to one eventual mocap object
        ret: list[MocapObject] = []
        for name, (pos, quat) in frame.items():
            pos_str = ' '.join(map(str, pos))  # np.ndarray([x, y, z]) -> "x y z"
            quat_str = ' '.join(map(str, quat))  # np.ndarray([x, y, z, w]) -> "x y z w"
            obj: Optional[MocapObject] = MocapObject.create(mocap, name)
            if obj is not None:
                self.add_object(obj, pos_str, quat_str, color)
                ret.append(obj)
        return ret


