from aiml_virtual.object.moving_object import MocapObject, MovingObject
from aiml_virtual.util import mujoco_helper
import aiml_virtual.object.mesh_utility_functions as mutil
from enum import Enum
from stl import mesh
import numpy as np
import math


class PAYLOAD_TYPES(Enum):
    Box = "Box"
    Teardrop = "Teardrop"


class PayloadMocap(MocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(model, data, mocapid, name_in_xml, name_in_motive)

        self.data = data
        self.mocapid = mocapid
    
    
    def update(self, pos, quat):

        #quat_rot = quat.copy()
        #quat_rot = mujoco_helper.quaternion_multiply(quat, np.array((.71, 0.0, 0.0, .71)))

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

        self.qpos = self.free_joint.qpos
        self.qvel = self.free_joint.qvel

        self.sensor_posimeter = self.data.sensor(self.name_in_xml + "_posimeter").data
        self.sensor_orimeter = self.data.sensor(self.name_in_xml + "_orimeter").data
        self.sensor_velocimeter = self.data.sensor(self.name_in_xml + "_velocimeter").data

        self._airflow_samplers = []

        #self._top_rectangle_positions_world = np.zeros_like(self._top_rectangle_positions)
        #self._top_rectangle_positions_own_frame = np.zeros_like(self._top_rectangle_positions)
        #self._rectangle_normals = np.zeros_like(self._top_rectangle_positions)
        #self._rectangle_areas = np.zeros_like(len(self._top_rectangle_positions))


    def create_surface_mesh(self, surface_division_area: float):
        raise NotImplementedError("[Payload] Subclasses need to implement this method.")
    
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
        from aiml_virtual.airflow import AirflowSampler
        if isinstance(airflow_sampler, AirflowSampler):
            self._airflow_samplers += [airflow_sampler]
        else:
            raise Exception("[Payload] The received object is not of type AirflowSampler.")
    
    def get_qpos(self):
        return np.append(self.sensor_posimeter, self.sensor_orimeter)
 
    def set_force_torque(self, force, torque):

        self.qfrc_applied[0] = force[0]
        self.qfrc_applied[1] = force[1]
        self.qfrc_applied[2] = force[2]
        self.qfrc_applied[3] = torque[0]
        self.qfrc_applied[4] = torque[1]
        self.qfrc_applied[5] = torque[2]


class BoxPayload(Payload):

    def __init__(self, model, data, name_in_xml) -> None:
        super().__init__(model, data, name_in_xml)

        
        if self.geom.type == mujoco.mjtGeom.mjGEOM_BOX:

            self.size = self.geom.size # this is half size on each axis
            self.top_bottom_surface_area = 2 * self.size[0] * 2 * self.size[1]
            self.side_surface_area_xz = 2 * self.size[0] * 2 * self.size[2]
            self.side_surface_area_yz = 2 * self.size[1] * 2 * self.size[2]
            self.create_surface_mesh(0.0001)
        
        else:
            raise Exception("[BoxPayload __init__] Payload is not box shaped.")


    def create_surface_mesh(self, surface_division_area: float):
        """
        inputs:
          * surface_division_area: the area of the small surface squares in m^2
        """
        
        a = surface_division_area

        square_side = math.sqrt(a)

        subdiv_x = int(round(self.size[0] / square_side))
        subdiv_y = int(round(self.size[1] / square_side))
        subdiv_z = int(round(self.size[2] / square_side))

        self._set_top_and_bottom_mesh(subdiv_x, subdiv_y)
        self._set_side_mesh(subdiv_x, subdiv_y, subdiv_z)

    def _set_top_and_bottom_mesh(self, top_bottom_subdivision_x, top_bottom_subdivision_y):
        self._top_bottom_subdivision_x = top_bottom_subdivision_x
        self._top_bottom_subdivision_y = top_bottom_subdivision_y
        self.top_bottom_miniractangle_area = self.top_bottom_surface_area / (top_bottom_subdivision_x * top_bottom_subdivision_y)
        self._calc_top_rectangle_positions()
        self._calc_bottom_rectangle_positions()

    def _set_side_mesh(self, subdivision_x, subdivision_y, subdivision_z):
            
        self._side_subdivision_x = subdivision_x
        self._side_subdivision_y = subdivision_y
        self._side_subdivision_z = subdivision_z
        self.side_miniractangle_area_xz = self.side_surface_area_xz / (subdivision_x * subdivision_z)
        self.side_miniractangle_area_yz = self.side_surface_area_yz / (subdivision_y * subdivision_z)
        self._calc_side_rectangle_positions()
    
    def _get_top_position_at(self, i, j):
        """ get the center in world coordinates of a small rectangle on the top of the box """
        return self._top_rectangle_positions[i, j]
    
    #def get_top_surface_normal(self):

        # rotate (0, 0, 1) vector by rotation quaternion
        #rot_matrix = Rotation.from_quat(self.sensor_orimeter)
        #return rot_matrix.apply(np.array((0, 0, 1)))
        #return np.array((0, 0, 1))
    
    def get_top_rectangle_data_at(self, i, j):

        """
        returns:
        - position (in world coordinates) of the center of the small rectangle,
        - position in payload frame of the center of the small rectangle
        - normal vector of the surface
        - and area of the surface
        at index i, j
        """

        # rotate position with respect to the center of the box
        position_in_own_frame = mujoco_helper.qv_mult(self.sensor_orimeter, self._get_top_position_at(i, j))
        # add position of the center
        position = self.sensor_posimeter + position_in_own_frame
        normal = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, 0, 1)))
        return position, position_in_own_frame, normal, self.top_bottom_miniractangle_area
    
    def get_top_rectangle_data(self):
        pos_in_own_frame = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._top_rectangle_positions_raw)
        normal = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, 0, 1)))
        return pos_in_own_frame + self.sensor_posimeter, pos_in_own_frame, normal, self.top_bottom_miniractangle_area

    def get_bottom_rectangle_data(self):
        pos_in_own_frame = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._bottom_rectangle_positions_raw)
        normal = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, 0, -1)))
        return pos_in_own_frame + self.sensor_posimeter, pos_in_own_frame, normal, self.top_bottom_miniractangle_area

    def get_side_xz_rectangle_data(self):
        pos_in_own_frame_negative = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_rectangle_positions_xz_neg_raw)
        pos_in_own_frame_positive = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_rectangle_positions_xz_pos_raw)
        normal_negative = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, -1, 0)))
        normal_positive = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((0, 1, 0)))

        pos_world_negative = pos_in_own_frame_negative + self.sensor_posimeter
        pos_world_positive = pos_in_own_frame_positive + self.sensor_posimeter

        return pos_world_negative, pos_world_positive, pos_in_own_frame_negative, pos_in_own_frame_positive,\
            normal_negative, normal_positive, self.side_miniractangle_area_xz

    def get_side_yz_rectangle_data(self):
        pos_in_own_frame_negative = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_rectangle_positions_yz_neg_raw)
        pos_in_own_frame_positive = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._side_rectangle_positions_yz_pos_raw)
        normal_negative = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((-1, 0, 0)))
        normal_positive = mujoco_helper.qv_mult(self.sensor_orimeter, np.array((1, 0, 0)))

        pos_world_negative = pos_in_own_frame_negative + self.sensor_posimeter
        pos_world_positive = pos_in_own_frame_positive + self.sensor_posimeter

        return pos_world_negative, pos_world_positive, pos_in_own_frame_negative, pos_in_own_frame_positive,\
            normal_negative, normal_positive, self.side_miniractangle_area_yz
        
    def get_top_subdiv(self):
        return self._top_bottom_subdivision_x, self._top_bottom_subdivision_y

    def _calc_top_rectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small top rectangles """

        self._top_rectangle_positions = np.zeros((self._top_bottom_subdivision_x, self._top_bottom_subdivision_y, 3))


        pos_z = self.size[2] # no need to divide by 2, because it's half
        division_size_x = (2 * self.size[0]) / self._top_bottom_subdivision_x
        division_size_y = (2 * self.size[1]) / self._top_bottom_subdivision_y

        self._top_rectangle_positions_raw = np.zeros((self._top_bottom_subdivision_x * self._top_bottom_subdivision_y, 3))

        for i in range(self._top_bottom_subdivision_x):
            distance_x = i * division_size_x + (division_size_x / 2.0)
            pos_x = distance_x - self.size[0]

            for j in range(self._top_bottom_subdivision_y):
                
                distance_y = j * division_size_y + (division_size_y / 2.0)
                pos_y = distance_y - self.size[1]
                self._top_rectangle_positions[i, j] = np.array((pos_x, pos_y, pos_z))
                self._top_rectangle_positions_raw[(i * self._top_bottom_subdivision_y) + j] = np.array((pos_x, pos_y, pos_z)) # store the same data in a 1D array

    def _calc_bottom_rectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small bottom rectangles """

        self._bottom_rectangle_positions = np.zeros((self._top_bottom_subdivision_x, self._top_bottom_subdivision_y, 3))
        self._bottom_rectangle_positions_raw = np.zeros((self._top_bottom_subdivision_x * self._top_bottom_subdivision_y, 3))

        pos_z_offset = (-1) * self.size[2]

        for i in range(self._top_bottom_subdivision_x):
            for j in range(self._top_bottom_subdivision_y):
                self._bottom_rectangle_positions[i, j] = self._top_rectangle_positions[i, j] + pos_z_offset
                self._bottom_rectangle_positions_raw[(i * self._top_bottom_subdivision_y + j)] = self._top_rectangle_positions_raw[(i * self._top_bottom_subdivision_y) + j] + pos_z_offset

    def _calc_side_rectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small rectangles on the sides """

        self._side_rectangle_positions_xz_neg_raw = np.zeros((self._side_subdivision_x * self._side_subdivision_z, 3))
        self._side_rectangle_positions_xz_pos_raw = np.zeros((self._side_subdivision_x * self._side_subdivision_z, 3))
        self._side_rectangle_positions_yz_neg_raw = np.zeros((self._side_subdivision_y * self._side_subdivision_z, 3))
        self._side_rectangle_positions_yz_pos_raw = np.zeros((self._side_subdivision_y * self._side_subdivision_z, 3))

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

                self._side_rectangle_positions_xz_neg_raw[(i * self._side_subdivision_z) + j] = np.array((pos_x, -pos_y, pos_z))
                self._side_rectangle_positions_xz_pos_raw[(i * self._side_subdivision_z) + j] = np.array((pos_x, pos_y, pos_z))
        
        
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

                self._side_rectangle_positions_yz_neg_raw[(i * self._side_subdivision_z) + j] = np.array((-pos_x, pos_y, pos_z))
                self._side_rectangle_positions_yz_pos_raw[(i * self._side_subdivision_z) + j] = np.array((pos_x, pos_y, pos_z))

# TODO: use passed name_in_xml for the .stl path 
class TeardropPayload(Payload):
    def __init__(self, model, data, name_in_xml) -> None:
        super().__init__(model, data, name_in_xml)
        
        self._MINIMAL_Z = -0.03753486
        self._triangles = None
        self._center_positions = None
        self._normals = None
        self._areas = None

        print(name_in_xml)
        self._init_default_values("./../xml_models/meshes/payload/payload_simplified.stl")
        self._bottom_triangles, self._bottom_center_positions, self._bottom_normals, self._bottom_areas = self._init_bottom_data()
        self._top_triangles, self._top_center_positions, self._top_normals, self._top_areas = self._init_top_data()

    def _init_default_values(self, path):
        meter = 1000.0
        self._loaded_mesh = mesh.Mesh.from_file(path)
        self._loaded_mesh.vectors[:, :, [1, 2]] = self._loaded_mesh.vectors[:, :, [2, 1]]
        self._loaded_mesh.vectors /= meter

        self._triangles = self._loaded_mesh.vectors
        self._center_positions = mutil.get_center_positions(self._triangles)
        self._normals = mutil.get_triangle_normals(self._triangles)
        self._areas = (self._loaded_mesh.areas / (meter ** 2)).flatten()
        
        mutil.set_normals_pointing_outward(self._normals, self._center_positions)

    def get_data(self):
        pos_in_own_frame = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._center_positions)
        normals = mujoco_helper.qv_mult(self.sensor_orimeter, self._normals)
        return pos_in_own_frame + self.sensor_posimeter, pos_in_own_frame, normals, self._areas

    def trans_vec(self, vec):
        return mujoco_helper.qv_mult(self.sensor_orimeter, vec)

    def get_bottom_data(self):
        pos_in_own_frame = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._bottom_center_positions)
        
        normals = np.empty_like(self._bottom_normals)
        for i in range(len(self._bottom_normals)):
            normals[i] = self.trans_vec(self._bottom_normals[i])

        return pos_in_own_frame + self.sensor_posimeter, pos_in_own_frame, normals, self._bottom_areas

    def get_top_data(self):
        pos_in_own_frame = mujoco_helper.quat_vect_array_mult(self.sensor_orimeter, self._top_center_positions)
        normals = np.apply_along_axis(self.trans_vec, axis=1, arr=self._top_normals)
        return pos_in_own_frame + self.sensor_posimeter, pos_in_own_frame, normals, self._top_areas

    def _init_top_data(self):
        treshold = 0.0015
        mask = np.any(abs(self._triangles[:, :, 2] - self._MINIMAL_Z) < treshold, axis=1)
        return self._triangles[~mask], self._center_positions[~mask], self._normals[~mask], self._areas[~mask]

    def _init_bottom_data(self):
        treshold = 0.0015
        mask = np.any(abs(self._triangles[:, :, 2] - self._MINIMAL_Z) < treshold, axis=1)
        return self._triangles[mask], self._center_positions[mask], self._normals[mask], self._areas[mask]

    def create_surface_mesh(self, threshold_in_meters):
        area_treshold = threshold_in_meters / 1000.0
        mask = self._areas > area_treshold

        triangles_to_divide = self._triangles[mask]
        normals_to_divide = self._normals[mask]
        areas_to_divide = self._areas[mask]

        triangles_kept = self._triangles[~mask]
        normals_kept = self._normals[~mask]
        areas_kept = self._areas[~mask]    

        if len(triangles_to_divide) == 0:
            return

        new_triangles = []
        midpoints = self._get_mid_points(triangles_to_divide)
        for i in range(len(triangles_to_divide)):
            new_triangles.extend(np.array([
                    [triangles_to_divide[i][0], midpoints[i][0], midpoints[i][2]],
                    [midpoints[i][0], triangles_to_divide[i][1], midpoints[i][1]],
                    [midpoints[i][1], triangles_to_divide[i][2], midpoints[i][2]],
                    [midpoints[i][0], midpoints[i][1], midpoints[i][2]]
                ])
            )

        new_normals = np.repeat(normals_to_divide, 4, axis=0)
        new_areas = np.repeat(areas_to_divide, 4, axis=0) / 4
        new_triangles = np.array(new_triangles)

        self._triangles = np.concatenate([triangles_kept, np.array(new_triangles)])
        self._normals = np.concatenate([normals_kept, np.array(new_normals)])
        self._areas = np.concatenate([areas_kept, np.array(new_areas)])
        self._center_positions = mutil.get_center_positions(self._triangles)

        self._bottom_triangles, self._bottom_center_positions, self._bottom_normals, self._bottom_areas = self._init_bottom_data()
        self._top_triangles, self._top_center_positions, self._top_normals, self._top_areas = self._init_top_data()

        self.create_surface_mesh(threshold_in_meters)

    def _get_mid_points(self, triangles):
        return (triangles[:, [0, 1, 2], :] + triangles[:, [1, 2, 0], :]) / 2