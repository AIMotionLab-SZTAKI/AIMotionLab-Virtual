import socket
import pickle  # Use pickle instead of json
import platform
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

from aiml_virtual.utils.dummy_mocap_server import DummyRigidBody


def request_data_from_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server
        sock.connect((host, port))

        while True:
            try:
                # Send a request for data
                sock.sendall(b"REQUEST_DATA")

                # Wait for the server's response
                response = sock.recv(1024)
                data = pickle.loads(response)
                print(f"Received data: {data}")

                # Simulate an infinite loop waiting for data updates

            except (ConnectionResetError, ConnectionAbortedError):
                print("Disconnected from server.")
                break


if __name__ == "__main__":
    host = "localhost"
    port = 9999
    request_data_from_server(host, port)
