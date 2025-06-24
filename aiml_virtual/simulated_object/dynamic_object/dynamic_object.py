"""
This module contains classes for SimulatedObjects that adhere to mujoco's rules of physics (gravity, etc.).
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Optional, Union
from xml.etree import ElementTree as ET
from abc import ABC
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.airflow.utils import *
from aiml_virtual.airflow.airflow_target import AirflowTarget
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

class TeardropPayload(DynamicObject):
    """
    Class for handling a teardrop shaped dynamic payload that is subject to physics. Not to be confused with a
    mocap payload, which is what we use to track a payload in optitrack.
    """

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        load_mass = "0.07"  # I'm pretty sure it's something like 70g
        segment_mass = "0.001"
        black = "0 0 0 1"
        body = ET.Element("body", name=self.name, pos=pos, quat=quat)
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

        return {"worldbody": [body]}

    def update(self) -> None:
        pass

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


    def get_top_rectangle_data(self):
        pos_in_own_frame = quat_vect_array_mult(self.sensors["quat"], self._top_rectangle_positions_raw)
        normal = qv_mult(self.sensors["quat"], np.array((0, 0, 1)))
        return pos_in_own_frame + self.sensors["pos"], pos_in_own_frame, normal, self.top_bottom_miniractangle_area

    def get_bottom_rectangle_data(self):
        pos_in_own_frame = quat_vect_array_mult(self.sensors["quat"], self._bottom_rectangle_positions_raw)
        normal = qv_mult(self.sensors["quat"], np.array((0, 0, -1)))
        return pos_in_own_frame + self.sensors["pos"], pos_in_own_frame, normal, self.top_bottom_miniractangle_area

    def get_side_xz_rectangle_data(self):
        pos_in_own_frame_negative = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_xz_neg_raw)
        pos_in_own_frame_positive = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_xz_pos_raw)
        normal_negative = qv_mult(self.sensors["quat"], np.array((0, -1, 0)))
        normal_positive = qv_mult(self.sensors["quat"], np.array((0, 1, 0)))

        pos_world_negative = pos_in_own_frame_negative + self.sensors["pos"]
        pos_world_positive = pos_in_own_frame_positive + self.sensors["pos"]

        return pos_world_negative, pos_world_positive, pos_in_own_frame_negative, pos_in_own_frame_positive,\
            normal_negative, normal_positive, self.side_miniractangle_area_xz

    def get_side_yz_rectangle_data(self):
        pos_in_own_frame_negative = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_yz_neg_raw)
        pos_in_own_frame_positive = quat_vect_array_mult(self.sensors["quat"], self._side_rectangle_positions_yz_pos_raw)
        normal_negative = qv_mult(self.sensors["quat"], np.array((-1, 0, 0)))
        normal_positive = qv_mult(self.sensors["quat"], np.array((1, 0, 0)))

        pos_world_negative = pos_in_own_frame_negative + self.sensors["pos"]
        pos_world_positive = pos_in_own_frame_positive + self.sensors["pos"]

        return pos_world_negative, pos_world_positive, pos_in_own_frame_negative, pos_in_own_frame_positive,\
            normal_negative, normal_positive, self.side_miniractangle_area_yz


    def get_rectangle_data(self):
        sides = []
        pos, pos_in_own_frame, normal, area = self.get_top_rectangle_data()
        sides.append((pos, pos_in_own_frame, normal, area, True, True)) # last 2 arguments: consider forces and torques on this side
        pos, pos_in_own_frame, normal, area = self.get_bottom_rectangle_data()
        sides.append((pos, pos_in_own_frame, normal, area, True, False))
        pos_n, pos_p, pos_in_own_frame_n, pos_in_own_frame_p, normal_n, normal_p, area = self.get_side_xz_rectangle_data()
        sides.append((pos_n, pos_in_own_frame_n, normal_n, area, True, True))
        sides.append((pos_p, pos_in_own_frame_p, normal_p, area, True, True))
        pos_n, pos_p, pos_in_own_frame_n, pos_in_own_frame_p, normal_n, normal_p, area = self.get_side_yz_rectangle_data()
        sides.append((pos_n, pos_in_own_frame_n, normal_n, area, True, True))
        sides.append((pos_p, pos_in_own_frame_p, normal_p, area, True, True))
        return sides

