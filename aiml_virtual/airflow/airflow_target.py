from __future__ import annotations
from typing import TYPE_CHECKING, Any
from abc import ABC, abstractmethod
if TYPE_CHECKING: # this avoids a circular import issue
    from aiml_virtual.airflow.airflow_sampler import AirflowSampler

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
    def get_rectangle_data(self) -> list[tuple[Any]]: # TODO: refactor this with proper types
        raise NotImplementedError()