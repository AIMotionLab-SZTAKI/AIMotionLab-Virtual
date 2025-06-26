"""
Module that contains the classes responsible for calculating airflow forces generated
by a drone acting on an airflow target.
"""
import math
import numpy as np
from aiml_virtual.airflow.box_dictionary import BoxDictionary
from aiml_virtual.airflow.airflow_target import AirflowTarget
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.drone import Drone
from aiml_virtual.airflow.utils import *
from abc import ABC, abstractmethod

class AirflowSampler(ABC):
    """
    Abstract base class to define an interface both kind of airflow samplers (simple and complex)
    shall implement.
    """
    def __init__(self, drone: Drone, cube_size):
        super().__init__()
        self.drone: Drone = drone #: The drone which generates the air flow
        self.offset_from_drone_center: np.ndarray = np.array([-cube_size / 200.0,
                                                  -cube_size / 200.0,
                                                  -cube_size / 100.0]) #: Offset of the air pressure dictionary's origo.
        self.index_upper_limit: float = (float(cube_size) - 0.5) / 100 #: Max index of the air pressure dictionary.

    @property
    def drone_position(self) -> np.ndarray:
        """
        Property to grab the drone's position.
        """
        return self.drone.sensors["pos"]

    @property
    def drone_orientation(self) -> np.ndarray:
        """
        Property to grab the drone's orientation.
        """
        return self.drone.sensors["quat"]

    def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the object’s global position and return its orientation quaternion.

        Applies the drone’s orientation quaternion to the object’s body‐frame offset
        vector to yield the world‐frame offset, then adds the drone’s global position
        to get the object’s absolute position. The drone’s orientation quaternion is
        returned unchanged.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - position (np.ndarray): 3-element array [x, y, z] in world coordinates.
                - orientation (np.ndarray): 4-element quaternion [w, x, y, z] of the drone.
        """
        position = qv_mult(self.drone_orientation, self.offset_from_drone_center)
        return position + self.drone_position, self.drone_orientation

    @abstractmethod
    def generate_forces(self, target: AirflowTarget) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the forces acting on the given airflow target.

        Args:
            target (AirflowTarget): The object on which the forces will act.

        Returns:
            tuple[np.ndarray, np.ndarray]: The acting forces (0th) and torques (1st).
        """
        raise NotImplementedError

class SimpleAirflowSampler(AirflowSampler):
    """
    Airflow sampler that disregards the drone's rotor speed change, and supposes that the rotors
    are constantly turning at a given speed.
    """
    def __init__(self, drone: Drone, pressure_file_path: str):
        tmp = np.loadtxt(pressure_file_path)
        cube_size = int(math.pow(tmp.shape[0] + 1, 1 / 3))
        self.pressure_data: np.ndarray = np.reshape(tmp, (cube_size, cube_size, cube_size)) #: 3D pressure map
        super().__init__(drone, cube_size)

    def generate_forces(self, target: AirflowTarget) -> tuple[np.ndarray, np.ndarray]:
        force_sum = np.array([0., 0., 0.])
        torque_sum = np.array([0., 0., 0.])
        selfposition, selforientation = self.get_position_orientation()
        rect_data = target.get_rectangle_data()
        pos, pos_own_frame, normal, area, force_enabled, torque_enabled = (rect_data.pos, rect_data.pos_own_frame,
                                                                           rect_data.normal,rect_data.area,
                                                                           rect_data.force_enabled, rect_data.torque_enabled)
        pos_traffed = pos - selfposition
        pos_traffed = quat_vect_array_mult_passive(selforientation, pos_traffed)
        condition = (pos_traffed[:, 0] >= 0) & (pos_traffed[:, 0] < self.index_upper_limit) & \
                    (pos_traffed[:, 1] >= 0) & (pos_traffed[:, 1] < self.index_upper_limit) & \
                    (pos_traffed[:, 2] >= 0) & (pos_traffed[:, 2] < self.index_upper_limit)
        pos_in_own_frame = pos_own_frame[condition]
        pos_traffed = pos_traffed[condition]
        normal = normal[condition]
        area = area[condition]
        indices = np.rint(pos_traffed * 100).astype(np.int32)
        pressure_values = self.pressure_data[indices[:, 0], indices[:, 1], indices[:, 2]]
        forces = forces_from_pressures(normal, pressure_values, area)
        torques = torque_from_force(pos_in_own_frame, forces)
        for force, torque, f_en, t_en in zip(forces, torques, force_enabled, torque_enabled):
            if f_en:
                force_sum += force
            if t_en:
                torque_sum += torque
        return force_sum, torque_sum

class ComplexAirflowSampler(AirflowSampler):
    """
    Airflow sampler that takes into account the drone's rotor speed changes as well as air velocity.
    """
    def __init__(self, drone: Drone, pressure_folder_path: str, velocity_folder_path: str):
        self.loaded_pressures = BoxDictionary(pressure_folder_path) #: lookup object for pressure
        self.loaded_velocities = BoxDictionary(velocity_folder_path) #: lookup object for velocity
        cube_size = self.loaded_pressures.get_cube_size()
        if self.loaded_velocities.get_cube_size() != cube_size:
            raise RuntimeError("size of look-up tables must match")
        super().__init__(drone, cube_size)

    def generate_forces(self, target: AirflowTarget) -> tuple[np.ndarray, np.ndarray]:
        force_sum = np.array([0., 0., 0.])
        torque_sum = np.array([0., 0., 0.])
        selfposition, selforientation = self.get_position_orientation()
        rect_data = target.get_rectangle_data()
        pos, pos_own_frame, normal, area, force_enabled, torque_enabled = (rect_data.pos, rect_data.pos_own_frame,
                                                                           rect_data.normal,rect_data.area,
                                                                           rect_data.force_enabled, rect_data.torque_enabled)
        pos_traffed = pos - selfposition
        pos_traffed = quat_vect_array_mult_passive(selforientation, pos_traffed)
        condition = (pos_traffed[:, 0] >= 0) & (pos_traffed[:, 0] < self.index_upper_limit) & \
                    (pos_traffed[:, 1] >= 0) & (pos_traffed[:, 1] < self.index_upper_limit) & \
                    (pos_traffed[:, 2] >= 0) & (pos_traffed[:, 2] < self.index_upper_limit)
        pos_in_own_frame = pos_own_frame[condition]
        pos_traffed = pos_traffed[condition]
        normal = normal[condition]
        area = area[condition]
        indices = np.rint(pos_traffed * 100).astype(np.int32)

        abs_average_velocity = np.sum(np.abs(self.drone.prop_vel)) / 4
        lower_bound, upper_bound = self.loaded_pressures.get_lower_upper_bounds(abs_average_velocity)
        lower_pressures, upper_pressures = self.loaded_pressures.get_lower_upper_bounds_arrays(lower_bound,
                                                                                               upper_bound)

        lower_pressure_values = lower_pressures[indices[:, 0], indices[:, 1], indices[:, 2]]
        upper_pressure_values = upper_pressures[indices[:, 0], indices[:, 1], indices[:, 2]]

        pressure_values = self.loaded_pressures.get_interpolated_array(abs_average_velocity, upper_pressure_values,
                                                                       lower_pressure_values, upper_bound,
                                                                       lower_bound)
        forces = forces_from_pressures(normal, pressure_values, area)

        abs_average_velocity = np.sum(np.abs(self.drone.prop_vel)) / 4
        lower_bound, upper_bound = self.loaded_velocities.get_lower_upper_bounds(abs_average_velocity)
        lower_velocities, upper_velocities = self.loaded_velocities.get_lower_upper_bounds_arrays(lower_bound,
                                                                                                  upper_bound)

        lower_velocity_values = lower_velocities[indices[:, 0], indices[:, 1], indices[:, 2]]
        upper_velocity_values = upper_velocities[indices[:, 0], indices[:, 1], indices[:, 2]]
        velocity_values = self.loaded_velocities.get_interpolated_array(abs_average_velocity, upper_velocity_values,
                                                                        lower_velocity_values, upper_bound,
                                                                        lower_bound)
        forces_velocity = forces_from_velocities(normal, velocity_values, area)
        forces += forces_velocity

        torques = torque_from_force(pos_in_own_frame, forces)
        for force, torque, f_en, t_en in zip(forces, torques, force_enabled, torque_enabled):
            if f_en:
                force_sum += force
            if t_en:
                torque_sum += torque
        return force_sum, torque_sum
