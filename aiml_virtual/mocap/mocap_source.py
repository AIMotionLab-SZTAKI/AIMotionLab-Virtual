# TODO: DOCSTRINGS AND COMMENTS

from abc import ABC, abstractmethod
import threading
import copy

import numpy as np

class MocapSource(ABC):
    def __init__(self):
        self.lock: threading.Lock = threading.Lock()
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:  # TODO: think through the lock here, and everywhere come to think of it
        with self.lock:
            # we don't want the user to get a reference to the underyling data, since they might modify it without
            # the lock: return a deepcopy instead, which reflects the state at this moment in time
            ret = copy.deepcopy(self._data)
        return ret

    @abstractmethod
    def mocap(self) -> None:
        pass
