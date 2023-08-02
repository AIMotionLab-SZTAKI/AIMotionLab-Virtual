from classes.moving_object import MovingMocapObject, MovingObject
from util import mujoco_helper
from enum import Enum
import numpy as np


class PAYLOAD_TYPES(Enum):
    Box = "Box"
    Teardrop = "Teardrop"

class PayloadMocap(MovingMocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(name_in_xml, name_in_motive)

        self.data = data
        self.mocapid = mocapid
    
    
    def update(self, pos, quat):

        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat
    
    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])


import mujoco
class Payload(MovingObject):

    def __init__(self, model, data, name_in_xml) -> None:
        super().__init__(model, name_in_xml)

        self.data = data

        # supporting only rectangular objects for now
        self.geom = self.model.geom(name_in_xml)
        
        self.free_joint = self.data.joint(self.name_in_xml)
        self.qfrc_passive = self.free_joint.qfrc_passive
        self.qfrc_applied = self.free_joint.qfrc_applied

        
        self.sensor_posimeter = self.data.sensor(self.name_in_xml + "_posimeter").data
        self.sensor_orimeter = self.data.sensor(self.name_in_xml + "_orimeter").data

        if self.geom.type == mujoco.mjtGeom.mjGEOM_BOX:

            self.size = self.geom.size # this is half size on each axis
            self.top_surface_area = 2 * self.size[0] * 2 * self.size[1]
            self.side_surface_area_xz = 2 * self.size[0] * 2 * self.size[2]
            self.side_surface_area_yz = 2 * self.size[1] * 2 * self.size[2]
            self.set_top_mesh(10, 10)
            self.set_side_mesh(10, 10, 10)
        
        self._airflow_samplers = []

        #self._top_minirectangle_positions_world = np.zeros_like(self._top_minirectangle_positions)
        #self._top_minirectangle_positions_own_frame = np.zeros_like(self._top_minirectangle_positions)
        #self._minirectangle_normals = np.zeros_like(self._top_minirectangle_positions)
        #self._minirectangle_areas = np.zeros_like(len(self._top_minirectangle_positions))

    
    def update(self, i, control_step):
        
        if len(self._airflow_samplers) > 0:
            force = np.array([0.0, 0.0, 0.0])
            torque = np.array([0.0, 0.0, 0.0])
            for airflow_sampler in self._airflow_samplers:
                f, t = airflow_sampler.generate_forces_opt(self)
                force += f
                torque += t
            self.set_force_torque(force / 1.5, torque / 1.5)
    
    def add_airflow_sampler(self, airflow_sampler):
        from classes.airflow_sampler import AirflowSampler
        if isinstance(airflow_sampler, AirflowSampler):
            self._airflow_samplers += [airflow_sampler]
        else:
            raise Exception("the received object is not of type AirflowSampler")
    
    def get_qpos(self):
        return np.append(self.sensor_posimeter, self.sensor_orimeter)
    
    def get_top_subdiv(self):
        return self._top_subdivision_x, self._top_subdivision_y

    def set_top_mesh(self, top_subdivision_x, top_subdivision_y):
        
        if self.geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            self._top_subdivision_x = top_subdivision_x
            self._top_subdivision_y = top_subdivision_y
            self.top_miniractangle_area = self.top_surface_area / (top_subdivision_x * top_subdivision_y)
            self._calc_top_minirectangle_positions()
        else:
            raise Exception("Payload type is not box")

    def set_side_mesh(self, subdivision_x, subdivision_y, subdivision_z):
        if self.geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            self._side_subdivision_x = subdivision_x
            self._side_subdivision_y = subdivision_y
            self._side_subdivision_z = subdivision_z
            self.side_miniractangle_area_xz = self.side_surface_area_xz / (subdivision_x * subdivision_z)
            self.side_miniractangle_area_yz = self.side_surface_area_yz / (subdivision_y * subdivision_z)
            self._calc_side_minirectangle_positions()
        else:
            raise Exception("Payload type is not box")
    
    def _calc_top_minirectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small rectangles """

        self._top_minirectangle_positions = np.zeros((self._top_subdivision_x, self._top_subdivision_y, 3))


        pos_z = self.size[2] # no need to divide by 2, because it's half
        division_size_x = (2 * self.size[0]) / self._top_subdivision_x
        division_size_y = (2 * self.size[1]) / self._top_subdivision_y

        self._top_minirectangle_positions_raw = np.zeros((self._top_subdivision_x * self._top_subdivision_y, 3))

        for i in range(self._top_subdivision_x):
            distance_x = i * division_size_x + (division_size_x / 2.0)
            pos_x = distance_x - self.size[0]

            for j in range(self._top_subdivision_y):
                
                distance_y = j * division_size_y + (division_size_y / 2.0)
                pos_y = distance_y - self.size[1]
                self._top_minirectangle_positions[i, j] = np.array((pos_x, pos_y, pos_z))
                self._top_minirectangle_positions_raw[(i * self._top_subdivision_x) + j] = np.array((pos_x, pos_y, pos_z)) # store the same data in a 1D array

    def _calc_side_minirectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small rectangles on the sides """

        self._side_minirectangle_positions_xz_neg_raw = np.zeros((self._side_subdivision_x * self._side_subdivision_z, 3))
        self._side_minirectangle_positions_xz_pos_raw = np.zeros((self._side_subdivision_x * self._side_subdivision_z, 3))
        self._side_minirectangle_positions_yz_neg_raw = np.zeros((self._side_subdivision_y * self._side_subdivision_z, 3))
        self._side_minirectangle_positions_yz_pos_raw = np.zeros((self._side_subdivision_y * self._side_subdivision_z, 3))

        # xz plane negative and positive side
        pos_y = self.size[1]
        div_size_x = (2 * self.size[0]) / self._side_subdivision_x
        div_size_z = (2 * self.size[2]) / self._side_subdivision_z
        
        for i in range(self._side_subdivision_x):
            distance_x = i * div_size_x + (div_size_x / 2.0)
            pos_x = distance_x - self.size[0]

            for j in range(self._side_subdivision_z):
                
                distance_z = j * div_size_z + (div_size_z / 2.0)
                pos_z = distance_z - self.size[2]

                self._side_minirectangle_positions_xz_neg_raw[(i * self._side_subdivision_z) + j] = np.array((pos_x, -pos_y, pos_z))
                self._side_minirectangle_positions_xz_pos_raw[(i * self._side_subdivision_z) + j] = np.array((pos_x, pos_y, pos_z))
        
        
        # yz plane negative and positive side
        pos_x = self.size[0]
        div_size_y = (2 * self.size[1]) / self._side_subdivision_y
        div_size_z = (2 * self.size[2]) / self._side_subdivision_z
        
        for i in range(self._side_subdivision_y):
            distance_y = i * div_size_y + (div_size_y / 2.0)
            pos_y = distance_y - self.size[1]

            for j in range(self._side_subdivision_z):
                
                distance_z = j * div_size_z + (div_size_z / 2.0)
                pos_z = distance_z - self.size[2]

                self._side_minirectangle_positions_yz_neg_raw[(i * self._side_subdivision_z) + j] = np.array((-pos_x, pos_y, pos_z))
                self._side_minirectangle_positions_yz_pos_raw[(i * self._side_subdivision_z) + j] = np.array((pos_x, pos_y, pos_z))


    def _get_top_position_at(self, i, j):
        """ get the center in world coordinates of a small rectangle on the top of the box """
        return self._top_minirectangle_positions[i, j]
    
    #def get_top_surface_normal(self):

        # rotate (0, 0, 1) vector by rotation quaternion
        #rot_matrix = Rotation.from_quat(self.sensor_orimeter)
        #return rot_matrix.apply(np.array((0, 0, 1)))
        #return np.array((0, 0, 1))
    
    def get_top_minirectangle_data_at(self, i, j):

        """
        returns:
        - position (in world coordinates) of the center of the minirectangle,
        - position in payload frame of the center of the minirectangle
        - normal vector of the surface
        - and area of the surface
        at index i, j
        """

        # rotate position with respect to the center of the box
        position_in_own_frame = mujoco_helper.qv_mult(self.sensor_orimeter, self._get_top_position_at(i, j))
        # add position of the center
        position = self.sensor_posimeter + position_in_own_frame
        normal = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, 0, 1)))
        return position, position_in_own_frame, normal, self.top_miniractangle_area
    
    def get_top_minirectangle_data(self):
        pos_in_own_frame = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._top_minirectangle_positions_raw)
        normal = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, 0, 1)))
        return pos_in_own_frame + self.sensor_posimeter, pos_in_own_frame, normal, self.top_miniractangle_area

    def get_side_xz_minirectangle_data(self):
        pos_in_own_frame_negative = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_minirectangle_positions_xz_neg_raw)
        pos_in_own_frame_positive = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_minirectangle_positions_xz_pos_raw)
        normal_negative = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, -1, 0)))
        normal_positive = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, 1, 0)))

        pos_world_negative = pos_in_own_frame_negative + self.sensor_posimeter
        pos_world_positive = pos_in_own_frame_positive + self.sensor_posimeter

        return pos_world_negative, pos_world_positive, pos_in_own_frame_negative, pos_in_own_frame_positive,\
            normal_negative, normal_positive, self.side_miniractangle_area_xz

    
    def get_side_yz_minirectangle_data(self):
        pos_in_own_frame_negative = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_minirectangle_positions_yz_neg_raw)
        pos_in_own_frame_positive = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_minirectangle_positions_yz_pos_raw)
        normal_negative = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((-1, 0, 0)))
        normal_positive = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((1, 0, 0)))

        pos_world_negative = pos_in_own_frame_negative + self.sensor_posimeter
        pos_world_positive = pos_in_own_frame_positive + self.sensor_posimeter

        return pos_world_negative, pos_world_positive, pos_in_own_frame_negative, pos_in_own_frame_positive,\
            normal_negative, normal_positive, self.side_miniractangle_area_yz

    def set_force_torque(self, force, torque):

        self.qfrc_applied[0] = force[0]
        self.qfrc_applied[1] = force[1]
        self.qfrc_applied[2] = force[2]
        self.qfrc_applied[3] = torque[0]
        self.qfrc_applied[4] = torque[1]
        self.qfrc_applied[5] = torque[2]
