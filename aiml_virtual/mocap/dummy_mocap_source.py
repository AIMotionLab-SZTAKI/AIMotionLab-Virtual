# TODO: DOCSTRINGS AND COMMENTS
import threading
import platform
import copy
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

import numpy as np

from aiml_virtual.mocap import mocap_source

class DummyMocapSource(mocap_source.MocapSource):

    def __init__(self, fps: float=120):  # TODO: move fps from here to the dummy server to mimic mocap
        super().__init__()
        self.fps: float = fps
        self.start_poses = copy.deepcopy(self._data)
        mocap_thread: threading.Thread = threading.Thread(target=self.mocap)
        mocap_thread.daemon = True
        mocap_thread.start()

    def set_data(self, new_data: dict[str, tuple[np.ndarray, np.ndarray]]):
        with self.lock:
            self._data = copy.deepcopy(new_data)
            self.start_poses = copy.deepcopy(self._data)


    def mocap(self) -> None:
        T = 5
        while True:
            d_pos = np.array([np.cos(2*np.pi*time.time()/T), np.sin(2*np.pi*time.time()/T), 0])
            with self.lock:
                for key in self._data.keys():
                    _, quat = self._data[key]
                    start_pos, _ = self.start_poses[key]
                    self._data[key] = (d_pos + start_pos, quat)
            time.sleep(1/self.fps)





