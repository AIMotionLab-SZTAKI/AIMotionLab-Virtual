from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Tuple
import networkx as nx
import math


class Terrain:
    def __init__(self,
                 dimension: np.ndarray,
                 max_height: float,
                 file_path: str,
                 safety_distance: float = 0.0
                 ):
        self.dimension: np.ndarray = dimension # x, y, z_min, z_max [m]
        self.max_height: float = max_height
        self.file_path: str = file_path
        self.safety_distance: float = safety_distance

        self.terrain: list = []

        self.build()

    def build(self) -> None:
        map_img = Image.open(self.file_path)
        map_data = np.asarray(map_img)
        X_map = np.linspace(-self.dimension[0]/2, self.dimension[0]/2, np.shape(map_data)[0])
        Y_map = np.linspace(-self.dimension[1]/2, self.dimension[1]/2, np.shape(map_data)[0])
        Z_map = map_data / (np.amax(map_data) / self.max_height)

        self.terrain = [X_map, -Y_map, Z_map.transpose()]

    def plot(self) -> None:
        ax = plt.gca()
        x = self.terrain[0]
        y = self.terrain[1]
        z = self.terrain[2]
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, z.transpose(), cmap='terrain', alpha=0.5)

    def delte_vertices_inside(self, vertices: np.ndarray, return_deleted: bool = False
                              ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        remove = np.full((len(vertices)), True)
        checked_area = max(abs(self.terrain[0][0] - self.terrain[0][1]), abs(self.terrain[1][0] - self.terrain[1][1])
                           ) * 2 + self.safety_distance
        for i, vertex in enumerate(vertices):
            in_area_X = np.logical_and(self.terrain[0] - checked_area <= vertex[0],
                                       vertex[0] <= self.terrain[0] + checked_area)
            in_area_Y = np.logical_and(self.terrain[1] - checked_area <= vertex[1],
                                       vertex[1] <= self.terrain[1] + checked_area)
            height_of_terrain = self.terrain[2][in_area_X, :][:, in_area_Y]
            if (vertex[2] <= height_of_terrain).any():
                remove[i] = False
        if return_deleted:
            return vertices[remove], vertices[np.logical_not(remove)]
        return vertices[remove]

    def delet_edges_inside(self, graph: nx.Graph) -> nx.Graph:
        res = min(abs(self.terrain[0][0] - self.terrain[0][1]), abs(self.terrain[1][0] - self.terrain[1][1])
                  ) * 2 + self.safety_distance
        for edge in graph.edges():
            length = graph[edge[0]][edge[1]]['weight']
            N = math.ceil(length / res)
            points = np.linspace(graph.nodes.data('pos')[edge[0]], graph.nodes.data('pos')[edge[1]], N)
            for i, point in enumerate(points):
                in_area_X = np.logical_and(self.terrain[0] - res * 2 <= point[0], point[0] <= self.terrain[0] + res * 2)
                in_area_Y = np.logical_and(self.terrain[1] - res * 2 <= point[1], point[1] <= self.terrain[1] + res * 2)
                height_of_terrain = self.terrain[2][in_area_X, :][:, in_area_Y]
                if (point[2] <= height_of_terrain).any():
                    graph.remove_edge(edge[0], edge[1])
                    break

        return graph

