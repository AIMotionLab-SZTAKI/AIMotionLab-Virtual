import pandas as pd
import numpy as np
from scipy.spatial import KDTree

class WindflowData:
    def __init__(self, path_to_csv):
        df = pd.read_csv(path_to_csv)
        self._positions = df[["Points:0", "Points:1", "Points:2"]].to_numpy()
        self._velocities = df[["U:0", "U:1", "U:2"]].to_numpy()
        self._NUMBER_OF_NEIGHBORS = 5
        self._tree = KDTree(self._positions)


    def get_corresponding_velocity(self, drone_position):
        distances, indices = self._tree.query(drone_position, self._NUMBER_OF_NEIGHBORS)
        neighbor_velocities = self._velocities[indices]
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        interpolated_value = np.dot(weights, neighbor_velocities)
        return 3.0 * interpolated_value