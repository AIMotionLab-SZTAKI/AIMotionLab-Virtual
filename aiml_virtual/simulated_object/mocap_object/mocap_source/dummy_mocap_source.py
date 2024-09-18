import threading
import socket
import pickle
import platform
from typing import Optional

if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

import numpy as np

from aiml_virtual.simulated_object.mocap_object.mocap_source import mocap_source
from aiml_virtual.utils.dummy_mocap_server import DummyRigidBody

class DummyMocapSource(mocap_source.MocapSource):

    def __init__(self, host: str, port: int, fps: 120):  # TODO: move fps from here to the dummy server to mimic mocap
        super().__init__()
        self.sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host: str = host
        self.port: int = port
        self._data: dict[str, DummyRigidBody] = {}
        self.connect()
        self.fps: float = fps
        mocap_thread: threading.Thread = threading.Thread(target=self.mocap)
        mocap_thread.daemon = True
        mocap_thread.start()

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
        except Exception as e:
            print(e.__repr__())

    @property
    def data(self) -> dict[str, DummyRigidBody]:
        with self.lock:
            return self._data

    def get_mocap_by_name(self, name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        data = self.data
        if name in data:
            return np.array(data[name].position), np.array(data[name].rotation)
        else:
            return None

    def mocap(self) -> None:
        print(f"Entering while loop!")
        while True:
            try:
                # Send a request for data
                self.sock.sendall(b"REQUEST_DATA")
                # Wait for the server's response
                response = self.sock.recv(1024)
                with self.lock:
                    self._data = pickle.loads(response)
                time.sleep(1/self.fps)

            except (ConnectionResetError, ConnectionAbortedError):
                print("Disconnected from server.")
                break



