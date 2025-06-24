"""
This module contains classes for SimulatedObjects that adhere to mujoco's rules of physics (gravity, etc.).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Optional, Union
from xml.etree import ElementTree as ET
from abc import ABC
import mujoco
import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation

from aiml_virtual import xml_directory
from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.airflow.utils import *
from aiml_virtual.airflow.airflow_target import AirflowTarget, AirflowData
if TYPE_CHECKING: # this avoids a circular import issue
    from aiml_virtual.airflow.airflow_sampler import AirflowSampler

class DynamicObject(simulated_object.SimulatedObject, ABC):
    """
    Base class for objects that follow the rules of physics. This includes simple objects without actuators such as a
    dynamic payload (as opposed to a mocap payload), as well as actuated objects such as a drone.
    """

    @property
    def mass(self) -> Union[None, float, np.array]:
        """
        Property to look up the mass of the object in the mjModel.
        """
        if self.model:
            return self.model.body(self.name).mass
        else:
            return None

    @property
    def inertia(self) -> Union[None, np.ndarray]:
        """
        Property to look up the diagonal inertia of the object in the mjModel.
        """
        if self.model:
            return self.model.body(self.name).inertia
        else:
            return None

    def bind_to_data(self, data: mujoco.MjData) -> None:
        self.data = data

class TeardropPayload(DynamicObject, AirflowTarget):
    """
    Class for handling a teardrop shaped dynamic payload that is subject to physics. Not to be confused with a
    mocap payload, which is what we use to track a payload in optitrack.
    """
    def __init__(self):
        super().__init__()

        self._triangles = None
        self._center_positions = None
        self._normals = None
        self._areas = None

        self._MIN_Z = None
        self._MAX_Z = None
        self._TOP_BOTTOM_RATIO = 0.005

        self.sensors: dict[str, np.ndarray] = {}  #: Dictionary of sensor data.
        payload_stl_path = os.path.join(xml_directory, "meshes", "payload", "payload_simplified.stl")
        self._init_default_values(payload_stl_path)
        self._bottom_triangles, self._bottom_center_positions, self._bottom_normals, self._bottom_areas = self._init_bottom_data()
        self._top_triangles, self._top_center_positions, self._top_normals, self._top_areas = self._init_top_data()

    def _init_default_values(self, path):
        meter = 1000.0
        self._loaded_mesh = mesh.Mesh.from_file(path)
        self._loaded_mesh.vectors[:, :, [1, 2]] = self._loaded_mesh.vectors[:, :, [2, 1]]
        self._loaded_mesh.vectors /= meter

        self._triangles = self._loaded_mesh.vectors
        self._center_positions = get_center_positions(self._triangles)
        self._normals = get_triangle_normals(self._triangles)
        self._areas = (self._loaded_mesh.areas / (meter ** 2)).flatten()
        self._MIN_Z = np.min(self._triangles[:, :, 2])
        self._MAX_Z = np.max(self._triangles[:, :, 2])

        set_normals_pointing_outward(self._normals, self._center_positions)

    def _init_top_data(self):
        mesh_height = self._MAX_Z - self._MIN_Z
        mask = np.any(self._triangles[:, :, 2] > self._MIN_Z + (mesh_height) * self._TOP_BOTTOM_RATIO, axis=1)
        return self._triangles[mask], self._center_positions[mask], self._normals[mask], self._areas[mask]

    def _init_bottom_data(self):
        mesh_height = self._MAX_Z - self._MIN_Z
        mask = np.any(self._triangles[:, :, 2] > (self._MIN_Z + (mesh_height) * self._TOP_BOTTOM_RATIO), axis=1)
        return self._triangles[~mask], self._center_positions[~mask], self._normals[~mask], self._areas[~mask]

    def get_top_rectangle_data(self) -> AirflowData:
        pos_in_own_frame = quat_vect_array_mult(self.sensors["quat"], self._top_center_positions)
        normals = quat_vect_array_mult(self.sensors["quat"], self._top_normals)
        n = self._top_center_positions.shape[0]
        return AirflowData(pos_in_own_frame + self.sensors["pos"], pos_in_own_frame, normals,
                           self._top_areas, [True] * n, [True] * n)

    def get_bottom_rectangle_data(self) -> AirflowData:
        pos_in_own_frame = quat_vect_array_mult(self.sensors["quat"], self._bottom_center_positions)
        normals = quat_vect_array_mult(self.sensors["quat"], self._bottom_normals)
        n = self._bottom_center_positions.shape[0]
        return AirflowData(pos_in_own_frame + self.sensors["pos"], pos_in_own_frame, normals,
                           self._bottom_areas, [True] * n, [True] * n)

    def get_rectangle_data(self) -> AirflowData:
        top = self.get_top_rectangle_data()
        bottom = self.get_bottom_rectangle_data()
        pos = np.concatenate([top.pos, bottom.pos], axis=0)
        pos_own_frame = np.concatenate([top.pos_own_frame, bottom.pos_own_frame], axis=0)
        normal = np.concatenate([top.normal, bottom.normal], axis=0)
        area = top.area + bottom.area
        force_enabled = top.force_enabled + bottom.force_enabled
        torque_enabled = top.torque_enabled + bottom.torque_enabled
        return AirflowData(pos, pos_own_frame, normal, area, force_enabled, torque_enabled)

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        load_mass = "0.07"  # I'm pretty sure it's something like 70g
        segment_mass = "0.001"
        black = "0 0 0 1"
        body = ET.Element("body", name=self.name, pos=pos, quat=quat)
        ret = {"worldbody": [body],
               "sensor": []}
        capsule_width = "0.004"
        ET.SubElement(body, "joint", name=self.name, type="free")
        # ET.SubElement(body, "joint", name=self.name, type="slide", axis="0 0 1")
        ET.SubElement(body, "geom", name=self.name, type="mesh", mesh="payload_simplified", pos="0 0 0.0405",
                      rgba="0 0 0 1", euler="1.57 0 0", mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", name=f"{self.name}_bottom", type="box", size=".016 .016 .02",
                      pos="0 0 0.0175", mass=load_mass, rgba="1.0 1.0 1.0 0.0")
        ET.SubElement(body, "geom", type="capsule", pos="0 0 0.075", size=capsule_width+" 0.027", rgba=black,
                      mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos="0.01173 0 0.10565", euler="0 1.12200 0",
                      size=capsule_width+" 0.01562", rgba=black, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos="0.01061 0 0.10439", euler="0 1.17810 0",
                      size=capsule_width+" 0.01378", rgba=black, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos="0.02561 0 0.11939", euler="0 0.39270 0",
                      size=capsule_width+" 0.01378", rgba=black, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos="-0.02561 0 0.14061", euler="0 0.39270 0",
                      size=capsule_width+" 0.005", rgba=black, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos="-0.01061 0 0.15561", euler="0 1.17810 0",
                      size=capsule_width+" 0.01378", rgba=black, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos="0.01061 0 0.15561", euler="0 1.96350 0",
                      size=capsule_width+" 0.01378", rgba=black, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos="0.02561 0 0.14061", euler="0 2.74889 0",
                      size=capsule_width+" 0.008", rgba=black, mass=segment_mass, condim="1")
        ET.SubElement(body, "site", name=f"{self.name}_contact_point", pos="0 0 0.16", type="sphere", size="0.002", rgba=black)
        ET.SubElement(body, "site", name=f"{self.name}_hook_center_point", pos="0 0 0.1275", type="sphere", size="0.001", rgba=black)
        ET.SubElement(body, "site", name=f"{self.name}_origin", pos="0 0 0", type="sphere", size="0.001", rgba=black)

        ret["sensor"].append(ET.Element("framepos", objtype="body", objname=self.name, name=self.name + "_posimeter"))
        ret["sensor"].append(
            ET.Element("framelinvel", objtype="body", objname=self.name, name=self.name + "_velocimeter"))
        ret["sensor"].append(ET.Element("framequat", objtype="body", objname=self.name, name=self.name + "_orimeter"))
        return ret

    def bind_to_data(self, data: mujoco.MjData) -> None:
        if self.model is None:
            raise RuntimeError
        self.data = data
        self.sensors["pos"] = self.data.sensor(self.name + "_posimeter").data
        self.sensors["quat"] = self.data.sensor(self.name + "_orimeter").data
        self.sensors["vel"] = self.data.sensor(self.name + "_velocimeter").data

    def update(self) -> None:
        if self.data is not None:
            user_forces = self.data.body(self.name).xfrc_applied
            force = np.array([0.0, 0.0, 0.0])
            torque = np.array([0.0, 0.0, 0.0])
            for airflow_sampler in self.airflow_samplers:
                f, t = airflow_sampler.generate_forces(self)
                force += f
                torque += t
            user_forces[0] = force[0]
            user_forces[1] = force[1]
            user_forces[2] = force[2]
            user_forces[3] = torque[0]
            user_forces[4] = torque[1]
            user_forces[5] = torque[2]

class BoxPayload(DynamicObject, AirflowTarget):
    """
    Class for handling a box shaped dynamic payload that is subject to physics.
    """

    def __init__(self):
        super().__init__()
        self.size: np.ndarray = np.array([0.05, 0.05, 0.05]) #: Dimensions of the box part of the payload (hook size unaffected).
        self.sensors: dict[str, np.ndarray] = {}  #: Dictionary of sensor data.
        self.top_bottom_surface_area = 2 * self.size[0] * 2 * self.size[1] #: TODO
        self.side_surface_area_xz = 2 * self.size[0] * 2 * self.size[2] #: TODO
        self.side_surface_area_yz = 2 * self.size[1] * 2 * self.size[2] #: TODO
        self.create_surface_mesh(0.0001)

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat)
        ret = {"worldbody" : [body],
               "sensor": []}
        ET.SubElement(body, "joint", name=self.name, type="free")
        box_pos = f"0 0 {self.size[2]}"
        str_size = f"{self.size[0]} {self.size[1]} {self.size[2]}"
        ET.SubElement(body, "geom", name=self.name, type="box", size=str_size, pos=box_pos, mass="0.07", rgba=color)
        segment_mass = "0.001"
        capsule_width = "0.004"
        hook_height = self.size[2] * 2
        ET.SubElement(body, "geom", type="capsule", pos=f"0 0 {0.025+hook_height}", size=capsule_width + " 0.027",
                      rgba=color, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos=f"0.01173 0 {0.05565+hook_height}", euler="0 1.12200 0",
                      size=capsule_width + " 0.01562", rgba=color, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos=f"0.01061 0 {0.05439+hook_height}", euler="0 1.17810 0",
                      size=capsule_width + " 0.01378", rgba=color, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos=f"0.02561 0 {0.06939+hook_height}", euler="0 0.39270 0",
                      size=capsule_width + " 0.01378", rgba=color, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos=f"-0.02561 0 {0.09061+hook_height}", euler="0 0.39270 0",
                      size=capsule_width + " 0.005", rgba=color, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos=f"-0.01061 0 {0.10561+hook_height}", euler="0 1.17810 0",
                      size=capsule_width + " 0.01378", rgba=color, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos=f"0.01061 0 {0.10561+hook_height}", euler="0 1.96350 0",
                      size=capsule_width + " 0.01378", rgba=color, mass=segment_mass, condim="1")
        ET.SubElement(body, "geom", type="capsule", pos=f"0.02561 0 {0.09061+hook_height}", euler="0 2.74889 0",
                      size=capsule_width + " 0.008", rgba=color, mass=segment_mass, condim="1")

        ret["sensor"].append(ET.Element("framepos", objtype="body", objname=self.name, name=self.name+"_posimeter"))
        ret["sensor"].append(ET.Element("framelinvel", objtype="body", objname=self.name, name=self.name + "_velocimeter"))
        ret["sensor"].append(ET.Element("framequat", objtype="body", objname=self.name, name=self.name + "_orimeter"))
        return ret

    def bind_to_data(self, data: mujoco.MjData) -> None:
        if self.model is None:
            raise RuntimeError
        self.data = data
        self.sensors["pos"] = self.data.sensor(self.name + "_posimeter").data
        self.sensors["quat"] = self.data.sensor(self.name + "_orimeter").data
        self.sensors["vel"] = self.data.sensor(self.name + "_velocimeter").data

    def update(self) -> None:
        if self.data is not None:
            user_forces = self.data.body(self.name).xfrc_applied
            force = np.array([0.0, 0.0, 0.0])
            torque = np.array([0.0, 0.0, 0.0])
            for airflow_sampler in self.airflow_samplers:
                f, t = airflow_sampler.generate_forces(self)
                force += f
                torque += t
            user_forces[0] = force[0]
            user_forces[1] = force[1]
            user_forces[2] = force[2]
            user_forces[3] = torque[0]
            user_forces[4] = torque[1]
            user_forces[5] = torque[2]

    def create_surface_mesh(self, surface_division_area: float) -> None:
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
        self.top_bottom_miniractangle_area = self.top_bottom_surface_area / (
                    top_bottom_subdivision_x * top_bottom_subdivision_y)
        self._calc_top_rectangle_positions()
        self._calc_bottom_rectangle_positions()

    def _set_side_mesh(self, subdivision_x, subdivision_y, subdivision_z):
        self._side_subdivision_x = subdivision_x
        self._side_subdivision_y = subdivision_y
        self._side_subdivision_z = subdivision_z
        self.side_miniractangle_area_xz = self.side_surface_area_xz / (subdivision_x * subdivision_z)
        self.side_miniractangle_area_yz = self.side_surface_area_yz / (subdivision_y * subdivision_z)
        self._calc_side_rectangle_positions()

    def _calc_top_rectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small top rectangles """

        self._top_rectangle_positions = np.zeros((self._top_bottom_subdivision_x, self._top_bottom_subdivision_y, 3))

        pos_z = self.size[2]  # no need to divide by 2, because it's half
        division_size_x = (2 * self.size[0]) / self._top_bottom_subdivision_x
        division_size_y = (2 * self.size[1]) / self._top_bottom_subdivision_y

        self._top_rectangle_positions_raw = np.zeros(
            (self._top_bottom_subdivision_x * self._top_bottom_subdivision_y, 3))

        for i in range(self._top_bottom_subdivision_x):
            distance_x = i * division_size_x + (division_size_x / 2.0)
            pos_x = distance_x - self.size[0]

            for j in range(self._top_bottom_subdivision_y):
                distance_y = j * division_size_y + (division_size_y / 2.0)
                pos_y = distance_y - self.size[1]
                self._top_rectangle_positions[i, j] = np.array((pos_x, pos_y, pos_z))
                self._top_rectangle_positions_raw[(i * self._top_bottom_subdivision_y) + j] = np.array(
                    (pos_x, pos_y, pos_z))  # store the same data in a 1D array

    def _calc_bottom_rectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small bottom rectangles """

        self._bottom_rectangle_positions = np.zeros((self._top_bottom_subdivision_x, self._top_bottom_subdivision_y, 3))
        self._bottom_rectangle_positions_raw = np.zeros(
            (self._top_bottom_subdivision_x * self._top_bottom_subdivision_y, 3))

        pos_z_offset = (-1) * self.size[2]

        for i in range(self._top_bottom_subdivision_x):
            for j in range(self._top_bottom_subdivision_y):
                self._bottom_rectangle_positions[i, j] = self._top_rectangle_positions[i, j] + pos_z_offset
                self._bottom_rectangle_positions_raw[(i * self._top_bottom_subdivision_y + j)] = \
                self._top_rectangle_positions_raw[(i * self._top_bottom_subdivision_y) + j] + pos_z_offset

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

                self._side_rectangle_positions_xz_neg_raw[(i * self._side_subdivision_z) + j] = np.array(
                    (pos_x, -pos_y, pos_z))
                self._side_rectangle_positions_xz_pos_raw[(i * self._side_subdivision_z) + j] = np.array(
                    (pos_x, pos_y, pos_z))

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

                self._side_rectangle_positions_yz_neg_raw[(i * self._side_subdivision_z) + j] = np.array(
                    (-pos_x, pos_y, pos_z))
                self._side_rectangle_positions_yz_pos_raw[(i * self._side_subdivision_z) + j] = np.array(
                    (pos_x, pos_y, pos_z))


    def get_top_rectangle_data(self) -> AirflowData:
        pos_in_own_frame = quat_vect_array_mult(self.sensors["quat"], self._top_rectangle_positions_raw)
        normal = qv_mult(self.sensors["quat"], np.array((0, 0, 1)))
        n = self._top_rectangle_positions_raw.shape[0]
        return AirflowData(pos_in_own_frame + self.sensors["pos"], pos_in_own_frame, np.tile(normal, (n, 1)),
                           [self.top_bottom_miniractangle_area] * n, [True] * n, [True] * n)

    def get_bottom_rectangle_data(self) -> AirflowData:
        pos_in_own_frame = quat_vect_array_mult(self.sensors["quat"], self._bottom_rectangle_positions_raw)
        normal = qv_mult(self.sensors["quat"], np.array((0, 0, -1)))
        n = self._bottom_rectangle_positions_raw.shape[0]
        return AirflowData(pos_in_own_frame + self.sensors["pos"], pos_in_own_frame, np.tile(normal, (n, 1)),
                           [self.top_bottom_miniractangle_area] * n, [True] * n, [False] * n)


    def get_side_xz_rectangle_data(self) -> tuple[AirflowData, AirflowData]:
        pos_in_own_frame_negative = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_xz_neg_raw)
        pos_in_own_frame_positive = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_xz_pos_raw)
        n = self._side_rectangle_positions_xz_neg_raw.shape[0]
        m = self._side_rectangle_positions_xz_pos_raw.shape[0]
        normal_negative = qv_mult(self.sensors["quat"], np.array((0, -1, 0)))
        normal_positive = qv_mult(self.sensors["quat"], np.array((0, 1, 0)))

        pos_world_negative = pos_in_own_frame_negative + self.sensors["pos"]
        pos_world_positive = pos_in_own_frame_positive + self.sensors["pos"]
        airflow_data_negative = AirflowData(pos_world_negative, pos_in_own_frame_negative,
                                            np.tile(normal_negative, (n, 1)), [self.side_miniractangle_area_xz]*n,
                                            [True] * n, [True] * n)
        airflow_data_positive = AirflowData(pos_world_positive, pos_in_own_frame_positive,
                                            np.tile(normal_positive, (m, 1)), [self.side_miniractangle_area_xz]*m,
                                            [True] * m, [True] * m)
        return airflow_data_negative, airflow_data_positive

    def get_side_yz_rectangle_data(self) -> tuple[AirflowData, AirflowData]:
        pos_in_own_frame_negative = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_yz_neg_raw)
        pos_in_own_frame_positive = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_yz_pos_raw)
        normal_negative = qv_mult(self.sensors["quat"], np.array((-1, 0, 0)))
        normal_positive = qv_mult(self.sensors["quat"], np.array((1, 0, 0)))
        n = self._side_rectangle_positions_yz_neg_raw.shape[0]
        m = self._side_rectangle_positions_yz_pos_raw.shape[0]
        pos_world_negative = pos_in_own_frame_negative + self.sensors["pos"]
        pos_world_positive = pos_in_own_frame_positive + self.sensors["pos"]

        airflow_data_negative = AirflowData(pos_world_negative, pos_in_own_frame_negative,
                                            np.tile(normal_negative, (n, 1)), [self.side_miniractangle_area_yz] * n,
                                            [True] * n, [True] * n)
        airflow_data_positive = AirflowData(pos_world_positive, pos_in_own_frame_positive,
                                            np.tile(normal_positive, (m, 1)), [self.side_miniractangle_area_yz] * m,
                                            [True] * m, [True] * m)
        return airflow_data_negative, airflow_data_positive


    def get_rectangle_data(self) -> AirflowData:
        top = self.get_top_rectangle_data()
        bottom = self.get_bottom_rectangle_data()
        left, right = self.get_side_xz_rectangle_data()
        back, front = self.get_side_yz_rectangle_data()
        pos = np.concatenate([top.pos, bottom.pos, left.pos, right.pos, back.pos, front.pos], axis=0)
        pos_own_frame = np.concatenate([top.pos_own_frame, bottom.pos_own_frame, left.pos_own_frame,
                                        right.pos_own_frame, back.pos_own_frame, front.pos_own_frame], axis=0)
        normal = np.concatenate([top.normal, bottom.normal, left.normal, right.normal, back.normal,
                                 front.normal], axis=0)
        area = top.area + bottom.area + left.area + right.area + back.area + front.area
        force_enabled = (top.force_enabled + bottom.force_enabled + left.force_enabled + right.force_enabled +
                         back.force_enabled + front.force_enabled)
        torque_enabled = (top.torque_enabled + bottom.torque_enabled + left.torque_enabled + right.torque_enabled +
                         back.torque_enabled + front.torque_enabled)
        return AirflowData(pos, pos_own_frame, normal, area, force_enabled, torque_enabled)

