"""
Module containing the interface definition that an object must satisfy if airflow forces shall act on it.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING: # this avoids a circular import issue
    from aiml_virtual.airflow.airflow_sampler import AirflowSampler

class AirflowData:
    """
    Class that encapsulates all the data required by an airflow sampler to calculate the airflow
    forces acting on an object. Also checks if the dimensions are correct. Contains lists or numpy
    arrays of the same length, each one's each element corresponding to a small surface on the object.
    """
    def __init__(self, pos: np.ndarray,
                 pos_own_frame: np.ndarray, # shaped n, 3
                 normal: np.ndarray, # shaped n, 3
                 area: np.ndarray, # shaped n, 1
                 force_enabled: list[bool], # length n
                 torque_enabled: list[bool]): # length n
        l = pos.shape[0]
        assert pos.shape == (l, 3)
        assert pos_own_frame.shape == (l,3)
        assert normal.shape == (l, 3)
        assert len(area) == l
        assert len(force_enabled) == l
        assert len(torque_enabled) == l
        self.pos = pos #: The position of a small surface rectangle.
        self.pos_own_frame = pos_own_frame #: The position of a small surface rectangle relative to the object.
        self.normal = normal #: The normal vector of a small surface rectangle.
        self.area = area #: The area of a surface rectangle.
        self.force_enabled = force_enabled #: Whether force component shall be considered on the surface rectangle.
        self.torque_enabled = torque_enabled #: Whether torque component shall be considered on the surface rectangle.


class AirflowTarget(ABC):
    """
    Abstract Base Class / Interface that a class must implement to be a target for airflow calculations.
    By default, both payload classes (in the dynamic_object module) implement this interface.
    """
    def __init__(self):
        super().__init__()
        self.airflow_samplers: list[AirflowSampler] = [] #: The list of samplers acting on this object.

    @abstractmethod
    def update(self) -> None:
        """
        Although AirflowTarget subclasses are usually also simulated objects, and therefore already implement
        update, this is also represented here, to signal that once forces have been calculated, they shall
        be applied to the object in this method.
        """
        raise NotImplementedError()

    def add_airflow_sampler(self, airflow_sampler: AirflowSampler) -> None:
        """
        Appends a sampler to the list of samplers which act on the object.

        Args:
            airflow_sampler (AirflowSampler): The new sampler.
        """
        self.airflow_samplers += [airflow_sampler]

    @abstractmethod
    def get_rectangle_data(self) -> AirflowData:
        """
        This is the method that subclasses must implement: calculate a partitioning of the surface
        of the object into small rectangles to allow an airflow sampler to calculate forces.

        Returns:
            AirflowData: The data of the surface of the object.
        """
        raise NotImplementedError()