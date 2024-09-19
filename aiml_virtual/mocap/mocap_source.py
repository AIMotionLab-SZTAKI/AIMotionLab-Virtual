from abc import ABC, abstractmethod
import threading
import copy

import numpy as np
from typing import Optional

class MocapSource(ABC):
    def __init__(self):
        self.lock: threading.Lock = threading.Lock()
        self._data: dict = {}

    @property
    def data(self) -> dict:  # TODO: think through the lock here, and everywhere come to think of it
        with self.lock:
            ret = copy.deepcopy(self._data)
        return ret

    @abstractmethod
    def get_mocap_by_name(self, name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        pass

    @abstractmethod
    def mocap(self) -> None:
        pass
