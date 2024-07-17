import numpy as np

# TODO: in the get_corresponding_velocity functions division in the return statement
# should be removed later
class CuboidArray:
    def __init__(self, wind_data_filepath):
        self._RESOLUTION = np.array([56, 125, 91])
        self._vectors = None
        self._velocities = None
        self._x_length  = None
        self._y_length  = None
        self._z_length  = None
        self._bottom_left = None
        self._top_right = None
        
        self._load_wind_data(wind_data_filepath)

    def _load_wind_data(self, wind_data_filepath):
        data = np.genfromtxt(wind_data_filepath, delimiter=',')
        self._vectors = data[1:, 0:3]
        self._velocities = data[1:, 3:6]

        sorted_indices = np.lexsort((self._vectors[:, 0], self._vectors[:, 1], self._vectors[:, 2]))
        self._vectors = self._vectors[sorted_indices]
        self._velocities = self._velocities[sorted_indices]
        self._bottom_left = self._vectors[0]
        self._top_right = self._vectors[-1]
        self._x_length = abs(self._bottom_left[0]) + abs(self._top_right[0])
        self._y_length = abs(self._bottom_left[1]) + abs(self._top_right[1])
        self._z_length = abs(self._bottom_left[2]) + abs(self._top_right[2])
    
        new_dimension = (self._RESOLUTION[2], self._RESOLUTION[1], self._RESOLUTION[0], 3)
        self._vectors = self._vectors.reshape(new_dimension)
        self._velocities = self._velocities.reshape(new_dimension)

    def get_indices_from_position(self, drone_position):
        new_bottom_left_corner = np.array([0., 0., 0.])
        translation = new_bottom_left_corner - self._bottom_left
        translated_position = drone_position + translation
        length_vector = np.array([self._x_length, self._y_length, self._z_length])
        return np.round((translated_position * self._RESOLUTION) / length_vector).astype(int)

    def get_corresponding_vector(self, drone_position):
        indices = self.get_indices_from_position(drone_position)
        return self._vectors[indices[2], indices[1], indices[0]]

    def get_corresponding_velocity(self, drone_position):
        if (drone_position[0] < self._bottom_left[0] or drone_position[1] < self._bottom_left[1] or drone_position[2] < self._bottom_left[2]):
            raise ValueError("Drone out of simulation space!")
        if (drone_position[0] > self._top_right[0] or drone_position[1] > self._top_right[1] or drone_position[2] > self._top_right[2]):
            raise ValueError("Drone out of simulation space!")

        indices = self.get_indices_from_position(drone_position)
        return self._velocities[indices[2], indices[1], indices[0]] / 10000