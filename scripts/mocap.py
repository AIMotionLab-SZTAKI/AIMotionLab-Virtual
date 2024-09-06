import motioncapture

if __name__ == "__main__":
    mocap = motioncapture.MotionCaptureOptitrack("192.168.2.141")
    while True:
        mocap.waitForNextFrame()
        print("_________________________________________________________________")
        for name, obj in mocap.rigidBodies.items():
            # Here we see what the optitrack returns. It has a string name, a list[float] position and a non-iterable
            # quaternion where x-y-z-w are fields. This may be a base for at-home reproduction.
            print(f"Name: {name}\n"
                  f"Position: {obj.position}\n"
                  f"Orientation: x: {obj.rotation.x}, y: {obj.rotation.y}, z: {obj.rotation.z}, w: {obj.rotation.w}")
