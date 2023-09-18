import math
import numpy as np

from classes.drone import Drone
from classes.payload import Payload
from util import mujoco_helper


class AirflowSampler:

    def __init__(self, data_file_name_pressure : str, owning_drone : Drone, data_file_name_velocity: str = None):
        
        #tmp = np.loadtxt(mujoco_helper.skipper(data_file_name), delimiter=',', dtype=np.float)
        tmp = np.loadtxt(data_file_name_pressure)
        # transform data into 3D array
        self._cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
        self.pressure_data = np.reshape(tmp, (self._cube_size, self._cube_size, self._cube_size))
        #print(self.pressure_data.shape)
        self.use_velocity = False
        if data_file_name_velocity is not None:
            tmp = np.loadtxt(data_file_name_velocity)
            if self._cube_size != int(math.pow(tmp.shape[0] + 1, 1/3)):
                raise RuntimeError("size of look-up tables must match")
        
            self.velocity_data = np.reshape(tmp, (self._cube_size, self._cube_size, self._cube_size, 3))
            self.use_velocity = True

        self.drone = owning_drone
        self.drone_position = owning_drone.sensor_posimeter
        self.drone_orientation = owning_drone.sensor_orimeter
        #self.offset_from_drone_center = np.copy(owning_drone.prop1_joint_pos)
        self.offset_from_drone_center = np.zeros_like(owning_drone.prop1_joint_pos)
        
        # shifting the cube's middle to the rotor
        self.offset_from_drone_center[0] -= self._cube_size / 200.0
        self.offset_from_drone_center[1] -= self._cube_size / 200.0
        self.offset_from_drone_center[2] -= self._cube_size / 100.0


        self._cube_size_meter = self._cube_size / 100.
        self.set_payload_offset(15)

        self.index_upper_limit = (float(self._cube_size) - 0.5) / 100.
        
        cs = self._cube_size_meter
        # for visualization purposes
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

    def set_payload_offset(self, offset_in_centimeter):
        self._payload_offset_z = offset_in_centimeter
        self._payload_offset_z_meter = self._payload_offset_z / 100.

    def get_transformed_vertices(self):
        self.vertices_world = mujoco_helper.quat_vect_array_mult(self.drone_orientation, self.vertices)
        self.vertices_world += self.drone_position

        return self.vertices_world

    def get_payload_offset_z(self):
        return self._payload_offset_z
    
    def get_payload_offset_z_meter(self):
        return self._payload_offset_z_meter
    
    def get_position_orientation(self):

        position = mujoco_helper.qv_mult(self.drone_orientation, self.offset_from_drone_center)

        return position + self.drone.sensor_posimeter, self.drone_orientation


    def sample_pressure_at_idx(self, i: int, j: int, k: int):

        return self.pressure_data[i, j, k]

    def sample_velocity_at_idx(self, i: int, j: int, k: int):

        return self.velocity_data[i, j, k]
    
#    def generate_forces(self, payload : Payload):
#
#        payload_subdiv_x, payload_subdiv_y = payload.get_top_subdiv()
#        force_sum = np.array([0., 0., 0.])
#        torque_sum = np.array([0., 0., 0.])
#        selfposition, selforientation = self.get_position_orientation()
#        # converting orientation
#        for x in range(payload_subdiv_x):
#            for y in range(payload_subdiv_y):
#
#                pos, pos_in_own_frame, normal, area = payload.get_top_minirectangle_data_at(x, y)
#
#                # transforming them into pressure volume coordinate system
#                pos_traffed = pos - selfposition
#                pos_traffed = mujoco_helper.qv_mult_passive(selforientation, pos_traffed)
#
#                i = int(round(pos_traffed[0] * 100))
#                j = int(round(pos_traffed[1] * 100))
#                k = int(round(pos_traffed[2] * 100))
#
#                #if y % 20 == 0:
#                #    print(k)
#                k += self._payload_offset_z
#
#                if i < self._cube_size and i >= 0 and\
#                   j < self._cube_size and j >= 0 and\
#                   k < self._cube_size and k >= 0:
#                
#                    pressure = self.sample_pressure_at_idx(i, j, k)
#
#                    # calculate forces
#                    force = mujoco_helper.force_from_pressure(normal, pressure, area)
#                    force_sum += force
#                    # calculate torque
#                    torque_sum += mujoco_helper.torque_from_force(pos_in_own_frame, force)
#
#        return force_sum, torque_sum
    
    
    def generate_forces_opt(self, payload : Payload):
        
        force_sum = np.array([0., 0., 0.])
        torque_sum = np.array([0., 0., 0.])

        pos, pos_in_own_frame, normal, area = payload.get_top_minirectangle_data()

        force, torque = self._gen_forces_one_side(pos, pos_in_own_frame, normal, area)
        force_sum += force
        torque_sum += torque

        
        pos_n, pos_p, pos_in_own_frame_n, pos_in_own_frame_p, normal_n, normal_p, area = payload.get_side_xz_minirectangle_data()

        force, torque = self._gen_forces_one_side(pos_n, pos_in_own_frame_n, normal_n, area)
        force_sum += force
        torque_sum += torque
        force, torque = self._gen_forces_one_side(pos_p, pos_in_own_frame_p, normal_p, area)
        force_sum += force
        torque_sum += torque

        
        pos_n, pos_p, pos_in_own_frame_n, pos_in_own_frame_p, normal_n, normal_p, area = payload.get_side_yz_minirectangle_data()

        force, torque = self._gen_forces_one_side(pos_n, pos_in_own_frame_n, normal_n, area)
        force_sum += force
        torque_sum += torque
        force, torque = self._gen_forces_one_side(pos_p, pos_in_own_frame_p, normal_p, area)
        force_sum += force
        torque_sum += torque

        return force_sum, torque_sum


    def _gen_forces_one_side(self, pos, pos_own_frame, normal, area):
        
        selfposition, selforientation = self.get_position_orientation()

        pos_traffed = pos - selfposition
        pos_traffed = mujoco_helper.quat_vect_array_mult_passive(selforientation, pos_traffed)

        pos_traffed[:, 2] += self._payload_offset_z_meter
        
        #pt = np.copy(pos_traffed)

        pos_in_own_frame = pos_own_frame[(pos_traffed[:, 0] >= 0) & (pos_traffed[:, 0] < self.index_upper_limit) & 
                                         (pos_traffed[:, 1] >= 0) & (pos_traffed[:, 1] < self.index_upper_limit) &
                                         (pos_traffed[:, 2] >= 0) & (pos_traffed[:, 2] < self.index_upper_limit)]

        pos_traffed = pos_traffed[(pos_traffed[:, 0] >= 0) & (pos_traffed[:, 0] < self.index_upper_limit) & 
                                  (pos_traffed[:, 1] >= 0) & (pos_traffed[:, 1] < self.index_upper_limit) &
                                  (pos_traffed[:, 2] >= 0) & (pos_traffed[:, 2] < self.index_upper_limit)]
        
        indices = np.rint(pos_traffed * 100).astype(np.int32)

        pressure_values = self.pressure_data[indices[:, 0], indices[:, 1], indices[:, 2]]
        forces = mujoco_helper.forces_from_pressures(normal, pressure_values, area)

        if self.use_velocity:
            velocity_values = self.velocity_data[indices[:, 0], indices[:, 1], indices[:, 2]]
            forces_velocity = mujoco_helper.forces_from_velocities(normal, velocity_values, area)
            forces += forces_velocity


        #if pos_in_own_frame.shape != forces.shape:
        #    print("shapes not equal")

        torques = mujoco_helper.torque_from_force(pos_in_own_frame, forces)

        force_sum = np.sum(forces, axis=0)
        torque_sum = np.sum(torques, axis=0)

        return force_sum, torque_sum