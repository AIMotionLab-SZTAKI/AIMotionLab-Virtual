import math
import numpy as np

from classes.drone import Drone
from classes.payload import Payload
from util import mujoco_helper

def skipper(fname):
    with open(fname) as fin:
        no_comments = (line for line in fin if not line.lstrip().startswith('#'))
        next(no_comments, None) # skip header
        for row in no_comments:
            yield row


class PressureSampler:

    def __init__(self, data_file_name : str, owning_drone : Drone):
        
        tmp = np.loadtxt(skipper(data_file_name), delimiter=',', dtype=np.float)
        # transform data into 3D array
        self.cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
        self.data = np.reshape(tmp[:, 4], (self.cube_size, self.cube_size, self.cube_size))
        print(self.data.shape)

        self.drone = owning_drone
        self.drone_position = owning_drone.sensor_posimeter
        self.drone_orientation = owning_drone.sensor_orimeter
        self.offset_from_drone_center = np.copy(owning_drone.prop1_joint_pos)
        
        # shifting the cube's middle to the rotor
        self.offset_from_drone_center[0] -= self.cube_size / 200.0
        self.offset_from_drone_center[1] -= self.cube_size / 200.0
        self.offset_from_drone_center[2] -= self.cube_size / 100.0
    
    def get_position_orientation(self):

        position = mujoco_helper.qv_mult(self.drone_orientation, self.offset_from_drone_center)

        return position + self.drone.sensor_posimeter, self.drone_orientation


    def sample_at_idx(self, i: int, j: int, k: int):

        return self.data[i, j, k]
    
    def generate_forces(self, payload : Payload):

        payload_subdiv_x, payload_subdiv_y = payload.get_top_subdiv()
        force_sum = np.array([0., 0., 0.])
        torque_sum = np.array([0., 0., 0.])
        selfposition, selforientation = self.get_position_orientation()
        # converting orientation
        for x in range(payload_subdiv_x):
            for y in range(payload_subdiv_y):

                pos, pos_in_own_frame, normal, area = payload.get_minirectangle_data_at(x, y)

                # transforming them into pressure volume coordinate system
                pos_traffed = pos - selfposition
                pos_traffed = mujoco_helper.qv_mult_passive(selforientation, pos_traffed)

                i = int(round(pos_traffed[0] * 100))
                j = int(round(pos_traffed[1] * 100))
                k = int(round(pos_traffed[2] * 100))

                #if y % 20 == 0:
                #    print(k)
                k += 25

                if i < self.cube_size and i >= 0 and\
                   j < self.cube_size and j >= 0 and\
                   k < self.cube_size and k >= 0:
                
                    pressure = self.sample_at_idx(i, j, k)

                    # calculate forces
                    force = mujoco_helper.force_from_pressure(normal, pressure, area)
                    force_sum += force
                    # calculate torque
                    torque_sum += mujoco_helper.torque_from_force(pos_in_own_frame, force)

        return force_sum, torque_sum
    
    
        




