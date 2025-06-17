import math
import numpy as np
from aiml_virtual.airflow.box_dictionary import BoxDictionary
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.drone import Drone
from aiml_virtual.simulated_object.dynamic_object.dynamic_object import BoxPayload, TeardropPayload

def quaternion_multiply(quaternion0, quaternion1):
    """Return multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array((
                    -x1*x0 - y1*y0 - z1*z0 + w1*w0,
                    x1*w0 + y1*z0 - z1*y0 + w1*x0,
                    -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                    x1*y0 - y1*x0 + z1*w0 + w1*z0), dtype=np.float64)

def q_conjugate(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

def qv_mult(q1, v1):
    """For active rotation. If passive rotation is needed, use q1 * q2 * q1^(-1)"""
    q2 = np.append(0.0, v1)
    return quaternion_multiply(q_conjugate(q1), quaternion_multiply(q2, q1))[1:]

def quat_quat_array_multiply(quat, quat_array):

    w0, x0, y0, z0 = quat
    w1, x1, y1, z1 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    return np.stack((-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0), axis=1)

def quat_array_quat_multiply(quat_array, quat):
    w0, x0, y0, z0 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    w1, x1, y1, z1 = quat
    return np.stack((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), axis=1)

def quat_vect_array_mult(q, v_array):
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_quat_array_multiply(q_conjugate(q), quat_array_quat_multiply(q_array, q))[:, 1:]

def quat_vect_array_mult_passive(q, v_array):
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_array_quat_multiply(quat_quat_array_multiply(q, q_array), q_conjugate(q))[:, 1:]

def forces_from_pressures(normal, pressure, area):
    # f = np.array([0., 0., -1.])
    # F = np.dot(-normal, f) * np.outer(pressure, f) * area
    if normal.ndim == 1:
        F = np.outer(pressure, -normal) * area
    else:
        F = np.expand_dims(pressure, axis=1) * (-normal) * np.expand_dims(area, axis=1)
    return F

def forces_from_velocities(normal, velocity, area):
    density = 1.293 #kg/m^3
    if normal.ndim == 1:
        F = velocity * density * area * np.dot(velocity, -normal).reshape(-1, 1)
    else:
        F = velocity * density * np.expand_dims(area, axis=1) * np.sum(velocity * (-normal), axis=1).reshape(-1, 1)
    return F

def torque_from_force(r, force):
    """by Adam Weinhardt"""
    M = np.cross(r, force)
    return M

class AirflowSampler:
    def __init__(self,
                 data_file_name_pressure: str,
                 owning_drone: Drone,
                 data_file_name_velocity: str = None,
                 LOAD_PRESSURE_DICTIONARY=False,
                 pressure_dictionary_folder_path=None,
                 LOAD_VELOCITY_DICTIONARY=False,
                 velocity_dictionary_folder_path=None):

        self.USE_PRESSURE_DICTIONARY = False
        self.USE_VELOCITY_DICTIONARY = False

        if LOAD_PRESSURE_DICTIONARY:
            self.USE_PRESSURE_DICTIONARY = True
            self.loaded_pressures = BoxDictionary(pressure_dictionary_folder_path)
            self._cube_size = self.loaded_pressures.get_cube_size()
        else:
            tmp = np.loadtxt(data_file_name_pressure)
            # transform data into 3D array
            self._cube_size = int(math.pow(tmp.shape[0] + 1, 1 / 3))
            self.pressure_data = np.reshape(tmp, (self._cube_size, self._cube_size, self._cube_size))

        if LOAD_VELOCITY_DICTIONARY:
            self.USE_VELOCITY_DICTIONARY = True
            self.loaded_velocities = BoxDictionary(velocity_dictionary_folder_path)
            if self.loaded_velocities.get_cube_size() != self._cube_size:
                raise RuntimeError("size of look-up tables must match")
        else:
            self.use_velocity = False
            if data_file_name_velocity is not None:
                tmp = np.loadtxt(data_file_name_velocity)
                if self._cube_size != int(math.pow(tmp.shape[0] + 1, 1 / 3)):
                    raise RuntimeError("size of look-up tables must match")
                self.velocity_data = np.reshape(tmp, (self._cube_size, self._cube_size, self._cube_size, 3))
                self.use_velocity = True

        self.drone = owning_drone
        self.drone_position = owning_drone.sensors["pos"]
        self.drone_orientation = owning_drone.sensors["quat"]
        # self.offset_from_drone_center = np.copy(owning_drone.prop1_joint_pos) #?
        self.offset_from_drone_center = np.zeros_like(owning_drone.prop_offset) # ?

        # shifting the cube's middle to the rotor
        self.offset_from_drone_center[0] -= self._cube_size / 200.0
        self.offset_from_drone_center[1] -= self._cube_size / 200.0
        self.offset_from_drone_center[2] -= self._cube_size / 100.0

        self._cube_size_meter = self._cube_size / 100.

        self._payload_offset_z = 0 # in cm
        self._payload_offset_z_meter = self._payload_offset_z / 100 # in meter

        self.index_upper_limit = (float(self._cube_size) - 0.5) / 100

        cs = self._cube_size_meter
        self.vertices = np.array([
            self.offset_from_drone_center,
            self.offset_from_drone_center + np.array([cs, 0, 0]),
            self.offset_from_drone_center + np.array([0, cs, 0]),
            self.offset_from_drone_center + np.array([cs, cs, 0]),
            self.offset_from_drone_center + np.array([0, 0, cs]),
            self.offset_from_drone_center + np.array([cs, 0, cs]),
            self.offset_from_drone_center + np.array([0, cs, cs]),
            self.offset_from_drone_center + np.array([cs, cs, cs])
        ]
        )

    def get_position_orientation(self):

        position = qv_mult(self.drone_orientation, self.offset_from_drone_center)

        return position + self.drone_position, self.drone_orientation

    def generate_forces_opt(self, payload: Payload):
        force_sum = np.array([0., 0., 0.])
        torque_sum = np.array([0., 0., 0.])

        abs_average_velocity = None
        if self.USE_PRESSURE_DICTIONARY:
            prop_velocities = self.drone.prop_vel
            abs_average_velocity = np.sum(np.abs(prop_velocities)) / 4

        if isinstance(payload, BoxPayload):
            pos, pos_in_own_frame, normal, area = payload.get_top_rectangle_data()
            force, torque = self._gen_forces_one_side(pos, pos_in_own_frame, normal, area, abs_average_velocity)
            force_sum += force
            torque_sum += torque

            pos, pos_in_own_frame, normal, area = payload.get_bottom_rectangle_data()
            force, torque = self._gen_forces_one_side(pos, pos_in_own_frame, normal, area, abs_average_velocity)
            force_sum += force
            # torque_sum += torque

            pos_n, pos_p, pos_in_own_frame_n, pos_in_own_frame_p, normal_n, normal_p, area = payload.get_side_xz_rectangle_data()
            force, torque = self._gen_forces_one_side(pos_n, pos_in_own_frame_n, normal_n, area, abs_average_velocity)
            force_sum += force
            torque_sum += torque
            force, torque = self._gen_forces_one_side(pos_p, pos_in_own_frame_p, normal_p, area, abs_average_velocity)
            force_sum += force
            torque_sum += torque

            pos_n, pos_p, pos_in_own_frame_n, pos_in_own_frame_p, normal_n, normal_p, area = payload.get_side_yz_rectangle_data()
            force, torque = self._gen_forces_one_side(pos_n, pos_in_own_frame_n, normal_n, area, abs_average_velocity)
            force_sum += force
            torque_sum += torque
            force, torque = self._gen_forces_one_side(pos_p, pos_in_own_frame_p, normal_p, area, abs_average_velocity)
            force_sum += force
            torque_sum += torque

        elif isinstance(payload, TeardropPayload):
            pos, pos_in_own_frame, normal, area = payload.get_top_data()
            force, torque = self._gen_forces_one_side(pos, pos_in_own_frame, normal, area, abs_average_velocity)
            force_sum += force
            torque_sum += torque

            pos, pos_in_own_frame, normal, area = payload.get_bottom_data()
            force, torque = self._gen_forces_one_side(pos, pos_in_own_frame, normal, area, abs_average_velocity)
            force_sum += force
            torque_sum += torque

        else:
            raise RuntimeError("payload not implemented!")

        return force_sum, torque_sum

    def _gen_forces_one_side(self, pos, pos_own_frame, normal, area, abs_average_velocity):
        selfposition, selforientation = self.get_position_orientation()
        pos_traffed = pos - selfposition
        pos_traffed = quat_vect_array_mult_passive(selforientation, pos_traffed)

        # pos_traffed[:, 2] += self._payload_offset_z_meter
        # pt = np.copy(pos_traffed)

        condition = (pos_traffed[:, 0] >= 0) & (pos_traffed[:, 0] < self.index_upper_limit) & \
                    (pos_traffed[:, 1] >= 0) & (pos_traffed[:, 1] < self.index_upper_limit) & \
                    (pos_traffed[:, 2] >= 0) & (pos_traffed[:, 2] < self.index_upper_limit)

        pos_in_own_frame = pos_own_frame[condition]
        pos_traffed = pos_traffed[condition]

        if normal.ndim > 1:
            normal = normal[condition]
            area = area[condition]

        indices = np.rint(pos_traffed * 100).astype(np.int32)

        if self.USE_PRESSURE_DICTIONARY:
            lower_bound, upper_bound = self.loaded_pressures.get_lower_upper_bounds(abs_average_velocity)
            lower_pressures, upper_pressures = self.loaded_pressures.get_lower_upper_bounds_arrays(lower_bound,
                                                                                                   upper_bound)

            lower_pressure_values = lower_pressures[indices[:, 0], indices[:, 1], indices[:, 2]]
            upper_pressure_values = upper_pressures[indices[:, 0], indices[:, 1], indices[:, 2]]

            pressure_values = self.loaded_pressures.get_interpolated_array(abs_average_velocity, upper_pressure_values,
                                                                           lower_pressure_values, upper_bound,
                                                                           lower_bound)

        else:
            pressure_values = self.pressure_data[indices[:, 0], indices[:, 1], indices[:, 2]]

        forces = forces_from_pressures(normal, pressure_values, area)

        if self.USE_VELOCITY_DICTIONARY:
            lower_bound, upper_bound = self.loaded_velocities.get_lower_upper_bounds(abs_average_velocity)
            lower_velocities, upper_velocities = self.loaded_velocities.get_lower_upper_bounds_arrays(lower_bound,
                                                                                                      upper_bound)

            lower_velocity_values = lower_velocities[indices[:, 0], indices[:, 1], indices[:, 2]]
            upper_velocity_values = upper_velocities[indices[:, 0], indices[:, 1], indices[:, 2]]

            velocity_values = self.loaded_velocities.get_interpolated_array(abs_average_velocity, upper_velocity_values,
                                                                            lower_velocity_values, upper_bound,
                                                                            lower_bound)

        elif self.use_velocity:
            velocity_values = self.velocity_data[indices[:, 0], indices[:, 1], indices[:, 2]]

        if self.USE_VELOCITY_DICTIONARY or self.use_velocity:
            forces_velocity = forces_from_velocities(normal, velocity_values, area)
            forces += forces_velocity

        # if pos_in_own_frame.shape != forces.shape:
        #    print("shapes not equal")

        torques = torque_from_force(pos_in_own_frame, forces)

        force_sum = np.sum(forces, axis=0)
        torque_sum = np.sum(torques, axis=0)
        return force_sum, torque_sum