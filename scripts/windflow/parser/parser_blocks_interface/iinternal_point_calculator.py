from abc import ABC, abstractmethod


class IInternalPointCalculator:
    @abstractmethod
    def get_internal_point(self, primitives):
        pass
