from abc import ABC, abstractmethod
from aiml_virtual.airflow.airflow_sampler import AirflowSampler

class AirflowTarget(ABC):
    def __init__(self):
        self._airflow_samplers: list[AirflowSampler] = []

    @abstractmethod
    def create_surface_mesh(self, surface_division_area: float) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_airflow_sampler(self, airflow_sampler: AirflowSampler) -> None:
        self._airflow_samplers += [airflow_sampler]