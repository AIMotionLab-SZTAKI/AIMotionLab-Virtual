from __future__ import annotations
from typing import TYPE_CHECKING, Any
from typing_extensions import Literal
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING: # this avoids a circular import issue
    from aiml_virtual.airflow.airflow_sampler import AirflowSampler

class AirflowData:
    pos: np.ndarray
    pos_own_frame: np.ndarray
    normal: np.ndarray
    area: np.ndarray
    force_enabled: list[bool]
    torque_enabled: list[bool]

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
        self.pos = pos
        self.pos_own_frame = pos_own_frame
        self.normal = normal
        self.area = area
        self.force_enabled = force_enabled
        self.torque_enabled = torque_enabled


# TODO: DOCSTRINGS
class AirflowTarget(ABC):
    def __init__(self):
        super().__init__()
        self.airflow_samplers: list[AirflowSampler] = []

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError()

    def add_airflow_sampler(self, airflow_sampler: AirflowSampler) -> None:
        self.airflow_samplers += [airflow_sampler]

    # TODO: TYPE
    @abstractmethod
    def get_rectangle_data(self) -> AirflowData:
        raise NotImplementedError()