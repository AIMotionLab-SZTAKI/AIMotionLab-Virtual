import numpy as np
from scipy.spatial import KDTree

class CuboidArray:
    def __init__(self, wind_data_filepath):
        data = np.genfromtxt(wind_data_filepath, delimiter=',')
        unique_pos, indices = np.unique(data[1:, 0:3], axis=0, return_index=True)

        self.position_vectors = unique_pos
        self.velocity_vectors = data[1:, 3:6][indices]
        self.tree = KDTree(self.position_vectors)

    def get(self, drone_position):
        distance, index = self.tree.query(drone_position)
        return self.velocity_vectors[index]