import numpy as np
import platform
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time


def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qw, qx, qy, qz: The orientation in quaternion [w,x,y,z] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

class Timer:
    # mimics the behavior of a stopwatch:
    # If you started it, and check its time (using measured_time or __call__), it returns the time since it
    # was started. If you stop it, it returns the time between the start and the stop.
    def __init__(self, initial_time = 0):
        self.time_started: float = initial_time
        self.time_stopped: float = initial_time
        self.running: bool = False

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.time_started = time.time()

    def stop(self) -> None:
        if self.running:
            self.running = False
            self.time_stopped = time.time()

    @property
    def measured_time(self) -> float:
        if self.running:
            return time.time() - self.time_started
        else:
            return self.time_stopped - self.time_started

    def __call__(self, *args, **kwargs):
        return self.measured_time