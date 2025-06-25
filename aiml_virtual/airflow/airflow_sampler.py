import math
import numpy as np
from aiml_virtual.airflow.box_dictionary import BoxDictionary
from aiml_virtual.airflow.airflow_target import AirflowTarget, AirflowData
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.drone import Drone
from aiml_virtual.simulated_object.dynamic_object.dynamic_object import BoxPayload, TeardropPayload
from aiml_virtual.airflow.utils import *
from abc import ABC, abstractmethod

class AirflowSampler(ABC):
    def __init__(self, drone: Drone, cube_size):
        super().__init__()
        self.drone: Drone = drone
        self.offset_from_drone_center: np.ndarray = np.array([-cube_size / 200.0,
                                                  -cube_size / 200.0,
                                                  -cube_size / 100.0])
        self.index_upper_limit: float = (float(cube_size) - 0.5) / 100

    @property
    def drone_position(self) -> np.ndarray:
        return self.drone.sensors["pos"]

    @property
    def drone_orientation(self) -> np.ndarray:
        return self.drone.sensors["quat"]

    def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        position = qv_mult(self.drone_orientation, self.offset_from_drone_center)
        return position + self.drone_position, self.drone_orientation

    @abstractmethod
    def generate_forces(self, target: AirflowTarget) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class SimpleAirflowSampler(AirflowSampler):
    def __init__(self, drone: Drone, pressure_file_path: str):
        tmp = np.loadtxt(pressure_file_path)
        cube_size = int(math.pow(tmp.shape[0] + 1, 1 / 3))
        self.pressure_data = np.reshape(tmp, (cube_size, cube_size, cube_size))
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
        area = [a for (a, c) in zip(area, condition) if c]
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
    def __init__(self, drone: Drone, pressure_folder_path: str, velocity_folder_path: str):
        self.loaded_pressures = BoxDictionary(pressure_folder_path)
        self.loaded_velocities = BoxDictionary(velocity_folder_path)
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
        area = [a for (a, c) in zip(area, condition) if c]
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


# # TODO: DOCSTRINGS
# class AirflowSampler:
#     def __init__(self,
#                  data_file_name_pressure: str,
#                  owning_drone: Drone,
#                  data_file_name_velocity: str = None,
#                  LOAD_PRESSURE_DICTIONARY=False,
#                  pressure_dictionary_folder_path=None,
#                  LOAD_VELOCITY_DICTIONARY=False,
#                  velocity_dictionary_folder_path=None):
#         super().__init__()
#         self.USE_PRESSURE_DICTIONARY = False
#         self.USE_VELOCITY_DICTIONARY = False
#
#         if LOAD_PRESSURE_DICTIONARY:
#             self.USE_PRESSURE_DICTIONARY = True
#             self.loaded_pressures = BoxDictionary(pressure_dictionary_folder_path)
#             self._cube_size = self.loaded_pressures.get_cube_size()
#         else:
#             tmp = np.loadtxt(data_file_name_pressure)
#             # transform data into 3D array
#             self._cube_size = int(math.pow(tmp.shape[0] + 1, 1 / 3))
#             self.pressure_data = np.reshape(tmp, (self._cube_size, self._cube_size, self._cube_size))
#
#         if LOAD_VELOCITY_DICTIONARY:
#             self.USE_VELOCITY_DICTIONARY = True
#             self.loaded_velocities = BoxDictionary(velocity_dictionary_folder_path)
#             if self.loaded_velocities.get_cube_size() != self._cube_size:
#                 raise RuntimeError("size of look-up tables must match")
#         else:
#             self.use_velocity = False
#             if data_file_name_velocity is not None:
#                 tmp = np.loadtxt(data_file_name_velocity)
#                 if self._cube_size != int(math.pow(tmp.shape[0] + 1, 1 / 3)):
#                     raise RuntimeError("size of look-up tables must match")
#                 self.velocity_data = np.reshape(tmp, (self._cube_size, self._cube_size, self._cube_size, 3))
#                 self.use_velocity = True
#
#         self.drone = owning_drone
#         self.drone_position = owning_drone.sensors["pos"]
#         self.drone_orientation = owning_drone.sensors["quat"]
#         # shifting the cube's middle to the rotor
#         self.offset_from_drone_center = np.array([-self._cube_size / 200.0,
#                                                   -self._cube_size / 200.0,
#                                                   -self._cube_size / 100.0])
#         self.index_upper_limit = (float(self._cube_size) - 0.5) / 100
#
#
#     def get_position_orientation(self):
#
#         position = qv_mult(self.drone_orientation, self.offset_from_drone_center)
#
#         return position + self.drone_position, self.drone_orientation
#
#     def generate_forces(self, target: AirflowTarget) -> tuple[np.ndarray, np.ndarray]:
#         force_sum = np.array([0., 0., 0.])
#         torque_sum = np.array([0., 0., 0.])
#         selfposition, selforientation = self.get_position_orientation()
#         rect_data = target.get_rectangle_data()
#         pos, pos_own_frame, normal, area, force_enabled, torque_enabled = (rect_data.pos, rect_data.pos_own_frame,
#                                                                            rect_data.normal,rect_data.area,
#                                                                            rect_data.force_enabled, rect_data.torque_enabled)
#         pos_traffed = pos - selfposition
#         pos_traffed = quat_vect_array_mult_passive(selforientation, pos_traffed)
#         condition = (pos_traffed[:, 0] >= 0) & (pos_traffed[:, 0] < self.index_upper_limit) & \
#                     (pos_traffed[:, 1] >= 0) & (pos_traffed[:, 1] < self.index_upper_limit) & \
#                     (pos_traffed[:, 2] >= 0) & (pos_traffed[:, 2] < self.index_upper_limit)
#         pos_in_own_frame = pos_own_frame[condition]
#         pos_traffed = pos_traffed[condition]
#         normal = normal[condition]
#         area = [a for (a, c) in zip(area, condition) if c]
#         indices = np.rint(pos_traffed * 100).astype(np.int32)
#
#         if self.USE_PRESSURE_DICTIONARY:
#             abs_average_velocity = np.sum(np.abs(self.drone.prop_vel)) / 4
#             lower_bound, upper_bound = self.loaded_pressures.get_lower_upper_bounds(abs_average_velocity)
#             lower_pressures, upper_pressures = self.loaded_pressures.get_lower_upper_bounds_arrays(lower_bound,
#                                                                                                    upper_bound)
#
#             lower_pressure_values = lower_pressures[indices[:, 0], indices[:, 1], indices[:, 2]]
#             upper_pressure_values = upper_pressures[indices[:, 0], indices[:, 1], indices[:, 2]]
#
#             pressure_values = self.loaded_pressures.get_interpolated_array(abs_average_velocity, upper_pressure_values,
#                                                                            lower_pressure_values, upper_bound,
#                                                                            lower_bound)
#         else:
#             pressure_values = self.pressure_data[indices[:, 0], indices[:, 1], indices[:, 2]]
#         forces = forces_from_pressures(normal, pressure_values, area)
#
#         if self.USE_VELOCITY_DICTIONARY:
#             abs_average_velocity = np.sum(np.abs(self.drone.prop_vel)) / 4
#             lower_bound, upper_bound = self.loaded_velocities.get_lower_upper_bounds(abs_average_velocity)
#             lower_velocities, upper_velocities = self.loaded_velocities.get_lower_upper_bounds_arrays(lower_bound,
#                                                                                                       upper_bound)
#
#             lower_velocity_values = lower_velocities[indices[:, 0], indices[:, 1], indices[:, 2]]
#             upper_velocity_values = upper_velocities[indices[:, 0], indices[:, 1], indices[:, 2]]
#
#             velocity_values = self.loaded_velocities.get_interpolated_array(abs_average_velocity, upper_velocity_values,
#                                                                             lower_velocity_values, upper_bound,
#                                                                             lower_bound)
#         elif self.use_velocity:
#             velocity_values = self.velocity_data[indices[:, 0], indices[:, 1], indices[:, 2]]
#
#         if self.USE_VELOCITY_DICTIONARY or self.use_velocity:
#             forces_velocity = forces_from_velocities(normal, velocity_values, area)
#             forces += forces_velocity
#         torques = torque_from_force(pos_in_own_frame, forces)
#         for force, torque, f_en, t_en in zip(forces, torques, force_enabled, torque_enabled):
#             if f_en:
#                 force_sum += force
#             if t_en:
#                 torque_sum += torque
#         return force_sum, torque_sum