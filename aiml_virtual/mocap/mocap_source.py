"""
This module contains the base class for mocap sources.
"""

from abc import ABC, abstractmethod
import threading
import copy
import numpy as np


class MocapSource(ABC):
    """
    Abstract base class for mocap sources, which provide data for mocap objects.

    .. note::
        Each MocapSource will have a private _data, which is a dictionary housing its data. This object musts not be
        accessed directly, only through the data property! The reason is that each MocapSource runs an infinite loop
        updating its _data in a thread separate from the main simulation thread. This thread can be started using the
        start_mocap_thread method, and its concrete implementation should be provided in the mocap
        method. In order to avoid concurrent accesses, _data is protected by a lock. When accessed through the data
        property, this lock is used properly, which means the following:

        - _data is only touched whenever the accessing thread has control of the lock.
        - Only a deep copy of _data is available through the data property, to avoid providing a reference through which
          other threads may try to modify it without a lock.

        Whenever one wants to work with the internal data, they shall either access it through data, or take care to
        only access it after acquiring its lock, as well as never returning a direct reference to it.
    """

    # TODO: read this config from file
    config: dict[str, str] = {
        "cf1": "MocapCrazyflie",
        "cf2": "MocapCrazyflie",
        "cf3": "MocapCrazyflie",
        "cf4": "MocapCrazyflie",
        "cf5": "MocapCrazyflie",
        "cf6": "MocapCrazyflie",
        "cf7": "MocapCrazyflie",
        "cf8": "MocapCrazyflie",
        "cf9": "MocapCrazyflie",
        "cf10": "MocapCrazyflie",
        "bb1": "MocapBumblebee",
        "bb2": "MocapBumblebee",
        "AI_Car_01": "MocapCar",
        "RC_Car_01": "MocapCar",
        "RC_Car_02": "MocapCar",
        "RC_Car_03": "MocapCar",
        "obst0": "Pole",
        "obst1": "Pole",
        "obst2": "Pole",
        "bu11": "Hospital",
        "bu13": "PostOffice",
        "bu14": "Sztaki",
        "payload1": "MocapPayload",
        "payload2": "MocapPayload",
        "bb3": "MocapHookedBumblebee2DOF",
        "JoeBush1": "MocapCar",  # TODO: move to skeleton
        "trailer": "MocapTrailer"
    }  #: **classcar** | Contains the recognized rigid bodies. Keys are the optitrack names, values are the class identifiers.

    def __init__(self):
        self.lock: threading.Lock = threading.Lock()  #: The lock used to access the underlying data dictionary.
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {}  #: The underyling data dictionary

    @property
    def data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Property to read the data of the motion capture system. This property provides a safe access to the underyling
        _data:

        - Only accesses it through an appropriate threading lock.
        - Only returns a deep copy of _data, instead of the actual reference to it.

        The keys of data are the names of the objects recognized by the motion capture system, the values are tuples.
        The first element of these tuples is the position, the second is the orientation (in quaternions, in x-y-z-w
        order).
        """
        with self.lock:
            # we don't want the user to get a reference to the underyling data, since they might modify it without
            # the lock: return a deepcopy instead, which reflects the state at this moment in time
            ret = copy.deepcopy(self._data)
        return ret

    @abstractmethod
    def mocap(self) -> None:
        """
        Each concrete subclass must implement this method. In this method, the underyling _data of the motion capture
        system must be updated **in an infinite loop**. For example, in this function frames may be read from the
        optitrack system, and stored in _data.

        .. note::
            When writing _data, make sure to use the threading lock!

        """
        pass

    def start_mocap_thread(self) -> None:
        """
        Dispatches the new thread which will be responsible for updating the underyling data. Usually, this may be
        called from the constructor right away, after setting up the mocap data.
        """
        mocap_thread: threading.Thread = threading.Thread(target=self.mocap)
        mocap_thread.daemon = True
        mocap_thread.start()
