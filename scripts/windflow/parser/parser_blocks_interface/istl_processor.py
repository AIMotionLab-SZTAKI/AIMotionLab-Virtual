from abc import ABC, abstractmethod


class ISTLProcessor:
    @abstractmethod
    def generate_stl(self, primitives, stl_filename):
        pass
