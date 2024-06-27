import os
import glob
import numpy as np
import math
import bisect

class BoxDictionary:
    def __init__(self, folder_path):
        self._loaded_data = {}
        self._txt_file_paths = glob.glob(folder_path + '/*.txt')
        self._cube_size = None
        self._DIGITS_COUNT_IN_FILENAME = 4

        self._load_raw_files_to_dictionary()
        self._cube_size = self._get_cube_size()
        self._reshape_dictionary()
        
    def _load_raw_files_to_dictionary(self):
        if (len(self._txt_file_paths) == 0):
            raise RuntimeError("no files found at given path!")

        for txt_file_path in self._txt_file_paths:
            number_from_file = self._get_extracted_number_from_file_path(txt_file_path)
            self._loaded_data[number_from_file] = np.loadtxt(txt_file_path)

            is_invalid_number_of_lines_in_file = not self._is_perfect_cube_number(len(self._loaded_data[number_from_file]))
            if (is_invalid_number_of_lines_in_file):
                raise RuntimeError(f"invalid number of lines in {txt_file_path}!")

    def _get_extracted_number_from_file_path(self, file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        if (len(filename) < self._DIGITS_COUNT_IN_FILENAME):
            raise RuntimeError("invalid file name!")
        
        return int(filename[-4:])
        
    def _get_cube_size(self):
        first_key_in_dict = next(iter(self._loaded_data))
        return int(math.pow(self._loaded_data[first_key_in_dict].shape[0] + 1, 1/3))

    def _is_perfect_cube_number(self, n):
        if n <= 0:
            return False
        cube_root = round(n ** (1/3))
        return cube_root ** 3 == n

    def _reshape_dictionary(self):
        first_key_in_dict = next(iter(self._loaded_data))
        dimension = self._get_dimension(self._loaded_data[first_key_in_dict])

        for key in self._loaded_data:
            self._loaded_data[key] = self._loaded_data[key].reshape(dimension)

    def _get_dimension(self, array):
        if (len(array.shape) == 1):
            return (self._cube_size, self._cube_size, self._cube_size)
        return (self._cube_size, self._cube_size, self._cube_size, array.shape[1])

    def get_cube_size(self):
        return self._cube_size

    def get_lower_upper_bounds(self, average_velocity):
        keys = sorted(int(key) for key in self._loaded_data.keys())
        target = int(average_velocity)

        if target <= keys[0]:
            return keys[0], keys[0]
        elif target >= keys[-1]:
            keys[-1], keys[-1]

        pos = bisect.bisect_left(keys, target)
        if (keys[pos] == target):
            return keys[pos], keys[pos]

        less_than_or_equal = keys[0] if pos == 0 else keys[pos - 1]
        greater_than_or_equal = keys[-1] if pos == len(keys) else keys[pos]

        return less_than_or_equal, greater_than_or_equal

    def get_lower_upper_bounds_arrays(self, average_velocity):
        lower_bound, upper_bound = self.get_lower_upper_bounds(average_velocity)
        return self._loaded_data[lower_bound], self._loaded_data[upper_bound]

    @staticmethod
    def get_interpolation_quotients(average, lower, upper):
        distance_lower = abs(average - lower)
        distance_upper = abs(average - upper)

        if distance_lower == 0:
            return 1.0, 0.0
        elif distance_upper == 0:
            return 0.0, 1.0

        if (distance_lower > distance_upper):
            alpha = distance_upper / distance_lower
            return alpha, (1 - alpha)
        else:
            alpha = distance_lower / distance_upper
            return (1 - alpha), alpha
