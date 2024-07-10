import math
import numpy as np
import os
import glob

class WindSampler:
    def __init__(self, wind_data_filename):
        self._cube_size = None
        self._loaded_wind_velocities = None

    def _load_wind_velocities(self):
        tmp = np.loadtxt(wind_data_filename)
        self._cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
        self._loaded_wind_velocities = np.reshape(tmp, (self._cube_size, self._cube_size, self._cube_size))

    def generate_forces(self, drone):
        drone_speed = drone.get_qvel()
        drone_position = drone.get_qpos()
        wind_speed = self._get_wind_speed(drone_position)
        x = wind_speed - drone_speed
        alpha = np.power(10, -2) * np.array([3.3, 2.7, 2.2])
        phi = np.diag(x)
        return np.dot(phi, alpha)

    def _get_wind_speed(self, drone_position):
        """
            we return the corresponding wind_speed from the self._loaded_wind_velocities data
        """
        pass