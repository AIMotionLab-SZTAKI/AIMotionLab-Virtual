from abc import ABC, abstractmethod

class IBoundChecker:
    @abstractmethod
    def check_mesh_bounds(self, primitives):
        pass
