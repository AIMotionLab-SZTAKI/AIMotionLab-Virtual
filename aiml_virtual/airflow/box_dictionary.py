"""
Lookup and interpolation utilities for voxelized airflow data.

This module defines :class:`BoxDictionary`, a small helper that loads a set of text files
containing voxel grids (pressure or velocity fields) and provides simple lookup and linear
interpolation between grids.

Intended usage
--------------
A folder contains multiple ``.txt`` files. Each file stores a flattened voxel grid sampled
on the same cubic lattice around the drone. The *parameter value* associated with a file
(e.g., rotor-speed proxy / average propeller velocity) is encoded in the last 4 digits of
the filename.

At runtime, an airflow sampler:

- Loads a folder into a :class:`BoxDictionary`
- Queries the two nearest available parameter values via
  :meth:`get_lower_upper_bounds`
- Retrieves the corresponding voxel grids
- Interpolates a voxel grid at the current parameter value via
  :meth:`get_interpolated_array`

File format assumptions
-----------------------
- Each file must contain a number of rows that is a perfect cube (``cube_size**3``).
- Scalar fields are stored as a 1D list of length ``cube_size**3``.
- Vector fields may be stored with multiple columns, yielding an array of shape
  ``(cube_size**3, k)`` before reshaping.
"""
import os
import glob
import numpy as np
import bisect

class BoxDictionary:
    """
    Load voxel grids from a folder and provide bound lookup + linear interpolation.

    The folder is expected to contain ``.txt`` files where the last 4 digits of the filename
    represent an integer key (e.g., 0030, 0125). The content of each file is loaded into a
    numpy array, validated to represent a cube, and reshaped into a 3D (or 4D) grid.

    Attributes:
        _loaded_data (dict[int, np.ndarray]): Mapping from integer key to reshaped voxel grid.
        _cube_size (int): Number of voxels per axis of the cubic grid.
    """
    def __init__(self, folder_path):
        super().__init__()
        self._loaded_data = {}
        self._txt_file_paths = glob.glob(folder_path + '/*.txt')
        self._cube_size = None
        self._DIGITS_COUNT_IN_FILENAME = 4

        self._load_raw_files_to_dictionary()
        self._cube_size = self._get_cube_size()
        self._reshape_dictionary()

    def _load_raw_files_to_dictionary(self):
        """
        Load all ``.txt`` files into the internal dictionary as raw (unreshaped) arrays.

        Each file is keyed by the integer encoded in the last 4 digits of its filename.

        Raises:
            RuntimeError: If no files are found or if the grid shape cannot be inferred.
        """
        if (len(self._txt_file_paths) == 0):
            raise RuntimeError("no files found at given path!")

        for txt_file_path in self._txt_file_paths:
            number_from_file = self._get_extracted_number_from_file_path(txt_file_path)
            self._loaded_data[number_from_file] = np.loadtxt(txt_file_path)

    def _get_extracted_number_from_file_path(self, file_path):
        """
        Extract the integer key from a voxel-grid filename.

        The key is taken from the last 4 digits of the filename stem.

        Args:
            file_path (str): Path to the ``.txt`` file.

        Returns:
            int: The extracted integer key.

        Raises:
            RuntimeError: If the filename stem is shorter than 4 characters.
        """
        filename = os.path.splitext(os.path.basename(file_path))[0]

        if (len(filename) < self._DIGITS_COUNT_IN_FILENAME):
            raise RuntimeError("invalid file name!")

        return int(filename[-4:])

    @staticmethod
    def _infer_grid_shape(n: int) -> tuple[int, int, int]:
        """
        Infer ``(nx, ny, nz)`` from the total voxel count.

        First tries a perfect cube. If that fails, assumes ``nx == ny == 81``
        (the standard LUT xy resolution) and computes ``nz = n / (81 * 81)``.

        Raises:
            RuntimeError: If neither heuristic produces a valid factorisation.
        """
        cube_root = round(n ** (1 / 3))
        if cube_root ** 3 == n:
            return (cube_root, cube_root, cube_root)
        nxy = 81
        if n % (nxy * nxy) == 0:
            return (nxy, nxy, n // (nxy * nxy))
        raise RuntimeError(
            f"Cannot infer grid shape from {n} voxels: not a perfect cube "
            f"and not divisible by {nxy}×{nxy}={nxy*nxy}."
        )

    def _get_cube_size(self):
        """
        Compute the grid shape from the first loaded file and store it.

        Sets ``self._grid_shape = (nx, ny, nz)`` and returns the xy edge length.

        Returns:
            int: ``nx`` (== ``ny``), the number of voxels along the x/y axes.
        """
        first_key_in_dict = next(iter(self._loaded_data))
        n = self._loaded_data[first_key_in_dict].shape[0]
        self._grid_shape = self._infer_grid_shape(n)
        return self._grid_shape[0]

    def _reshape_dictionary(self):
        """
        Reshape each loaded array into its voxel-grid form.

        Scalar fields become ``(nx, ny, nz)``.
        Vector fields become ``(nx, ny, nz, k)``.
        """
        first_key_in_dict = next(iter(self._loaded_data))
        dimension = self._get_dimension(self._loaded_data[first_key_in_dict])

        for key in self._loaded_data:
            self._loaded_data[key] = self._loaded_data[key].reshape(dimension)

    def _get_dimension(self, array):
        """
        Determine the target reshape dimension for a raw loaded array.

        Args:
            array (np.ndarray): Raw array loaded from a file.

        Returns:
            tuple[int, ...]: Target shape for reshaping.
        """
        nx, ny, nz = self._grid_shape
        if (len(array.shape) == 1):
            return (nx, ny, nz)
        return (nx, ny, nz, array.shape[1])

    def get_cube_size(self):
        """
        Get the number of voxels along the x/y axes.

        Returns:
            int: ``nx`` (== ``ny``).
        """
        return self._cube_size

    def get_grid_shape(self) -> tuple[int, int, int]:
        """
        Get the full ``(nx, ny, nz)`` grid shape.

        Returns:
            tuple[int, int, int]: Voxel counts along x, y, and z axes.
        """
        return self._grid_shape

    def get_velocity_range(self) -> tuple[int, int]:
        """Return (min_key, max_key) — the propeller-velocity range covered by this dataset."""
        keys = sorted(self._loaded_data.keys())
        return keys[0], keys[-1]

    def get_lower_upper_bounds(self, average_velocity):
        """
        Find the nearest available keys bracketing a target value.

        This is used to select the two voxel grids between which interpolation should occur.

        Args:
            average_velocity (float): Target value to bracket (typically a rotor-speed proxy).

        Returns:
            tuple[int, int]: (lower_bound, upper_bound) keys from the loaded dataset.

        Notes:
            If the target is outside the available range, both bounds are clamped to the
            nearest endpoint.
        """
        keys = sorted(int(key) for key in self._loaded_data.keys())
        target = int(average_velocity)

        if target <= keys[0]:
            return keys[0], keys[0]
        elif target >= keys[-1]:
            return keys[-1], keys[-1]

        pos = bisect.bisect_left(keys, target)
        if (keys[pos] == target):
            return keys[pos], keys[pos]

        less_than_or_equal = keys[0] if pos == 0 else keys[pos - 1]
        greater_than_or_equal = keys[-1] if pos == len(keys) else keys[pos]

        return less_than_or_equal, greater_than_or_equal

    def get_lower_upper_bounds_arrays(self, lower_bound, upper_bound):
        """
        Retrieve the voxel grids corresponding to two keys.

        Args:
            lower_bound (int): Lower bound key.
            upper_bound (int): Upper bound key.

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower_array, upper_array) voxel grids.
        """
        return self._loaded_data[lower_bound], self._loaded_data[upper_bound]

    def get_interpolated_array(self, average_velocity, upper_array, lower_array, upper_bound, lower_bound):
        """
        Linearly interpolate a voxel grid between two bounds.

        Args:
            average_velocity (float): Target value to interpolate at.
            upper_array (np.ndarray): Voxel grid at ``upper_bound``.
            lower_array (np.ndarray): Voxel grid at ``lower_bound``.
            upper_bound (int): Key associated with ``upper_array``.
            lower_bound (int): Key associated with ``lower_array``.

        Returns:
            np.ndarray: Interpolated voxel grid. If bounds are equal, returns ``lower_array``.
        """
        if lower_bound == upper_bound:
            return lower_array
        else:
            return ((upper_array - lower_array) / (upper_bound - lower_bound)) * (
                        average_velocity - lower_bound) + lower_array