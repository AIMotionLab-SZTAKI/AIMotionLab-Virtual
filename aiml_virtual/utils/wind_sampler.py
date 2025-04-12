import numpy as np

from aiml_virtual.utils.windflow_data import WindflowData

class WindSampler:
    def __init__(self, path_to_wind_data):
        self._windflow_data = WindflowData(path_to_wind_data)

    def get_force(self, drone_position, drone_velocity):
        wind_velocity = self._windflow_data.get_corresponding_velocity(drone_position)
        x = wind_velocity - drone_velocity
        ALPHA = (10 ** (-2)) * np.array([3.3, 2.7, 2.2])
        phi = np.diag(x)
        return np.dot(phi, ALPHA)