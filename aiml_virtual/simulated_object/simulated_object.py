"""
This module implements the base class for objects in the simulation that we also want to manipluate in python.
"""

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Optional, Type
import mujoco
import numpy as np


class SimulatedObject(ABC):
    """
    Base class for objects in the simulation for which we also want a python representation.

    This class (in addition to being a blueprint for subclasses) keeps track of its subclasses in two ways:

    - Keeps track of what aliases should be checked when parsing a MJCF file for a given subclass.
    - Keeps track of how many instances of the class exist in the runtime.

    This is achieved by utilizing the __init_subclass__ function, which gets called when a subclass is defined. A
    subclass of SimulatedObject will be (for example) ControlledObject, meaning that when the interpeter first
    encounters the class ControlledObject, it calls __init_subclass__ with the argument cls set to ControlledObject.
    Since ControlledObject is a subclass of SimulatedObject, it inherits this function, meaning that if ControlledObject
    in turn has a subclass Drone, then whenever Drone is defined, this function gets called, even though Drone is not a
    direct subclass of SimulatedObject. If ControlledObject also implements its own __init_subclass__, it will overload
    the inherited __init_subclass__, therefore, if we want to call it, then ControlledObject.__init_subclass__ shall
    include a call super().__init_subclass__, that way it calls the parent's respective function in addition to its own.
    SimulatedObject.__init_subclass__ registers the potential aliases of the class to its xml registry, and sets its
    instance count to 0. If a class wants to be a candidate for parsing in an xml file, it shall implement all the
    abstract methods defined in SimulatedObject and in get_identifiers return a list of valid string identifiers.
    If a class need not be target for parsing, it may return None in get_identifiers.

    .. note::
        Only classes which get initialized with __init_subclass__ are candidates for parsing. If the script the user
        is running doesn't import a given class, and neither of its imports result in the given class being imported
        either, the class cannot be parsed into an object. Typical use case for this is reading objects straight
        from an xml (mjcf) or straight from a mocap stream (as opposed to adding them via Scene.add_object). If the
        mocap stream (or the mjcf file) contains a MocapBumblebee, but the script doesn't import the file defining
        MocapBumblebee, then __init_subclass__(MocapBumblebee) doesn't get invoked, and the entry will be skipped
        when parsing. We can avoid this by calling utils_general.import_submodules(__name__) in __init__.py. This
        utility function will recursively import all subpackages under the package which it defines.

    .. note::
        In order to make it possible for the python objects and the Data/Model to exist separately, references to the
        data and the model aren't saved when the object is initialized. This way, we can initialize a simulated object,
        set whatever properties we want for it, and *then* use the bind_to_model/data functions to bind it. This also
        means that when we reload, and the model/data are lost, the object may persist.
    """

    DEFAULT_UPDATE_FREQ: float = 500 #: **classvar** | The default frequency at which each object runs its update function
    xml_registry: dict[str, Type['SimulatedObject']] = {}  #: **classvar** | The registry of xml names associated with each class.
    instance_count: dict[Type['SimulatedObject'], int] = {}  #: **classvar** | The registry tallying the number of instances per class.

    @classmethod
    @abstractmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        """
        Gives a list of identifiers (aliases) for the class to check for aliases when parsing an XML. A None returns
        signals that this class opts out of parsing. This usually also means that it's an abstract class (ABC).

        Returns:
            Optional[list[str]]: The list of aliases for objects belonging to this class, or None if abstract.
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        #  A class descended from SimulatedObject *must* specify whether he is a candidate for parsing. If the user
        #  doesn't want the class to be a candidate for parsing (because the class is abstract, or for any other
        #  reason), the get_identifiers() function shall return None
        identifiers: Optional[str] = cls.get_identifiers()
        if identifiers is None:
            return
        else:
            for identifier in identifiers:
                if identifier in SimulatedObject.xml_registry:
                    raise ValueError(f"identifier {identifier} is already registered to "
                                     f"{SimulatedObject.xml_registry[identifier].__name__}")
                else:
                    SimulatedObject.xml_registry[identifier] = cls
            SimulatedObject.instance_count[cls] = 0

    def __init__(self):
        cls = self.__class__
        self.name = f"{cls.__name__}_{SimulatedObject.instance_count[cls]}"  #: The name parsed from the MJCF file.
        SimulatedObject.instance_count[cls] += 1
        self.model: Optional[mujoco.MjModel] = None  #: The mujoco model in which the object exists.
        self.data: Optional[mujoco.MjData] = None  #: The mujoco data for the simulation.
        self.update_frequency: float = SimulatedObject.DEFAULT_UPDATE_FREQ  #: The frequency with which the update function will run.

    def bind_to_model(self, model: mujoco.MjModel) -> None:
        """
        Saves a reference of the model to the object.

        Args:
            model (mujoco.MjModel): The model as read from a MJCF file.

        .. note::
            Information regarding the object in the model should be retrieved via properties, instead of overwriting
            this function and binding them. That way, if the model is updated, the properties stay up to date.
        """
        self.model = model

    @abstractmethod
    def bind_to_data(self, data: mujoco.MjData) -> None:
        """
        Method for concrete subclasses to save all their references to controller-actuator-etc. This is where we may
        save references to objects that only get initialized alongside MjData.

        Args:
            data (mujoco.MjData): The data of the simulation (as opposed to the *model*).
        """
        pass

    def update(self) -> None:
        """
        The simulator will call this function once every control loop: if the object needs to update control inputs,
        or change its appearance (such as rotating its propellers), it shall be done in this function.
        """
        pass

    @abstractmethod
    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        """
        It is each object's responsibility to know how it ought to look in a MJCF file. This method is how those XML
        elements are generated.

        Args:
            pos (str): The position of the object in the scene, x-y-z separated by spaces. E.g.: "0 1 -1"
            quat (str): The quaternion orientation of the object in the scene, w-x-y-z separated by spaces.
            color (str): The base color of the object in th scene, r-g-b-opacity separated by spaces, scaled 0.0  to 1.0

        Returns:
            dict[str, list[ET.Element]]: A dictionary where the keys are tags of XML elements in the MJCF file, and the
            values are lists of XML elements to be appended as children to those XML elements.
        """
        pass

    @property
    def id(self) -> Optional[int]:
        """
        Property to look up the ID of the body associated with the python object in the mujoco model.
        """
        if self.model is not None:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.name)
        else:
            return None

    @property
    def xpos(self) -> Optional[np.ndarray]:
        """
        Property to look up the global position of the body associated with the python object in the mujoco data.
        """
        if self.data is not None:
            return self.data.body(self.name).xpos
        else:
            return None

    @property
    def xquat(self) -> Optional[np.ndarray]:
        """
        Property to look up the orientation of the body associated with the python object in the mujoco data.
        """
        if self.data is not None:
            return self.data.body(self.name).xquat
        else:
            return None