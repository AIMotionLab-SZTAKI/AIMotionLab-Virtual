import math
import numpy as np
from aiml_virtual.wind_flow.cuboid_array import CuboidArray

class WindSampler:
    def __init__(self, wind_data_filenames):
        self._wind_data = []
        self._load_wind_data(wind_data_filenames)

    def _load_wind_data(self, wind_data_filenames):
        for i in range(len(wind_data_filenames)):
            self._wind_data.append(CuboidArray(wind_data_filenames[i]))

    def generate_forces(self, drone):
        from aiml_virtual.object.drone import Drone
        
        drone_state = drone.get_state_copy()
        drone_velocity = drone_state["vel"]
        drone_position = drone_state["pos"]

        force_sum = np.array([0., 0., 0.])
        for wind_data in self._wind_data:
            wind_velocity = wind_data.get_corresponding_velocity(drone_position)
            x = wind_velocity - drone_velocity
            ALPHA = (10**(-2)) * np.array([3.3, 2.7, 2.2])
            phi = np.diag(x)
            force_sum += np.dot(phi, ALPHA)

        return force_sum