import math
import numpy as np

class WindSampler:
    def __init__(self, wind_data_filename):
        self._dimensions = np.array([6.1, 5.1, 2.7]) # sidelengths of the rectangular cuboid in meters
        self._resolution = np.array([6100, 5100, 2700]) # the division rate per dimension (mm resolution)

        self._loaded_wind_velocities = None
        self._load_wind_velocities()

    def _load_wind_velocities(self):
        tmp = np.loadtxt(wind_data_filename)
        self._loaded_wind_velocities = np.reshape(tmp, self._resolution)

    def _get_index_from_position(self, drone_position):
        return (drone_position * self._resolution) / self._dimensions

    def generate_forces(self, drone):
        drone_state = drone.get_state_copy()
        drone_velocity = drone_state["vel"]
        drone_position = drone_state["pos"]
        wind_speed = self._get_wind_speed(drone_position)
        x = wind_speed - drone_speed
        alpha = np.power(10, -2) * np.array([3.3, 2.7, 2.2])
        phi = np.diag(x)
        return np.dot(phi, alpha)

    def _get_wind_speed(self, drone_position):
        """
            This works if the drone's and rectangular cuboid's coordinate system align
        """
        return self._load_wind_velocities[self._get_index_from_position(drone_position)]