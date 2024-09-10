import motioncapture
import threading
import time

from mpmath.rational import mpq_1


def printmocap_data(mocap: motioncapture.MotionCaptureOptitrack):
    while True:
        loop_start = time.time()
        mocap.waitForNextFrame()
        print("_________________________________________________________________")
        for name, obj in mocap.rigidBodies.items():
            # Here we see what the optitrack returns. It has a string name, a list[float] position and a non-iterable
            # quaternion where x-y-z-w are fields. This may be a base for at-home reproduction.
            print(f"Name: {name}\n"
                  f"Position: {obj.position}\n"
                  f"Orientation: x: {obj.rotation.x}, y: {obj.rotation.y}, z: {obj.rotation.z}, w: {obj.rotation.w}")
        time.sleep(loop_start + 1 - time.time())

if __name__ == "__main__":
    mocap = motioncapture.MotionCaptureOptitrack("192.168.2.141")
    mocap_thread = threading.Thread(target=printmocap_data, args=(mocap,))
    mocap_thread.start()
    while True:
        print(f"Main thread doing something...")
        time.sleep(1)
