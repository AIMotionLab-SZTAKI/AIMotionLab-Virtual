from __future__ import annotations
from typing import TYPE_CHECKING, Any
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING: # this avoids a circular import issue
    from aiml_virtual.airflow.airflow_sampler import AirflowSampler
from dataclasses import dataclass

@dataclass
class AirflowData:
    pos: np.ndarray
    pos_own_frame: np.ndarray
    normal: np.ndarray
    area: list[float]
    force_enabled: list[bool]
    torque_enabled: list[bool]

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