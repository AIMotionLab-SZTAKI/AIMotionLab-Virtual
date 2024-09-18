import socketserver
import threading
import time
import pickle

from docutils.nodes import target

# todo: types and docstrings

class DummyRigidBody:
    def __init__(self, position: list[float], rotation: list[float]):
        self.position: list[float] = position
        self.rotation: list[float] = rotation

class DummyMocapServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, host: str, port: int, rhc: type[socketserver.BaseRequestHandler],
                 rigidbodies: dict[str, DummyRigidBody], common_speed: list[float],
                 frequency: float = 120):
        self.host: str = host
        self.port: int = port
        self.rigidbodies: dict[str, DummyRigidBody] = rigidbodies
        self.common_speed: list[float] = common_speed
        self.lock: threading.Lock = threading.Lock()
        self.frequency: float = frequency
        super().__init__((host, port), rhc)

        server_thread = threading.Thread(target=self.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print(f"Server started on {self.host}:{self.port}")

        updater_thread = threading.Thread(target=self._update_rigidbodies)
        updater_thread.daemon = True
        updater_thread.start()



    def _update_rigidbodies(self):
            dt = 1/self.frequency
            ds = [v*dt for v in self.common_speed]
            while True:
                time.sleep(1/self.frequency)
                with self.lock:
                    for rigidbody in self.rigidbodies.values():
                        rigidbody.position = [p + dp for p, dp in zip(rigidbody.position, ds)]
                print(f"Updated rigidbodies.")

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server: DummyMocapServer):
        self.lock = server.lock
        self.data = server.rigidbodies
        super().__init__(request, client_address, server)

    def handle(self):
        while True:
            try:
                # Receive the client request (waiting for data request)
                request = self.request.recv(1024).decode('utf-8')

                if request == "REQUEST_DATA":
                    # Wait until the shared_data is updated (via an event or time delay)
                    with self.lock:
                        # Send the updated dictionary
                        self.request.sendall(pickle.dumps(self.data))
                else:
                    # If an unknown message is received, disconnect
                    break
            except ConnectionResetError:
                break

if __name__ == "__main__":
    rigidbodies = {
        "MocapCrazyflie_0": DummyRigidBody([-1, -1, 1], [0, 0, 0, 1]),
        "MocapCrazyflie_1": DummyRigidBody([1, -1, 1], [0, 0, 0, 1]),
        "bb0": DummyRigidBody([-1, 1, 1], [0, 0, 0, 1]),
        "bb1": DummyRigidBody([1, 1, 1], [0, 0, 0, 1])
    }
    server = DummyMocapServer("localhost", 9999, ThreadedTCPRequestHandler, rigidbodies, [0.1, 0, 0], 50)

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.shutdown()
        server.server_close()