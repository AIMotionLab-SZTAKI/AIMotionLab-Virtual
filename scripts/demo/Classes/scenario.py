from __future__ import annotations

import copy

import numpy as np
import networkx as nx
import math
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import time

from .terrain import Terrain
from .static_obstacle import StaticObstacleS
from .dynamic_obstacle import DynamicObstacle
from .radar import RadarS
from .leader_plane import Leader_plane
from Utils.Utils import c_print, get_edges, plot_fly_zone


def target_layout_set(layout_id: int) -> np.ndarray:
    """
    Define the target points which will be available for the drones to fly to. To add a new set of targets just
    add a new elif statement and define a set of cordinates.

    :param layout_id: int -> the selected target point set.
    :return: array([[x,y,z]...[x,y,z]]) -> the coordinates of the targets
    """
    if layout_id == 0:
        return np.array([])

    elif layout_id == 1:
        # Shape: X_________________________
        #           03 02 01 00|
        #           07       04|
        #           11       08|
        #           15 14 13 12|
        #                                  Y
        inner_xy = 1.1
        outer_xy = 2 * inner_xy
        high_layer = 1.1

        V_fix = [[0, 0, 0.5], [-outer_xy, -outer_xy, high_layer], [-inner_xy, -outer_xy, high_layer],
                 [inner_xy, -outer_xy, high_layer], [outer_xy, -outer_xy, high_layer],
                 [-outer_xy, -inner_xy, high_layer], [outer_xy, -inner_xy, high_layer],
                 [-outer_xy, inner_xy, high_layer], [outer_xy, inner_xy, high_layer],
                 [-outer_xy, outer_xy, high_layer], [-inner_xy, outer_xy, high_layer], [inner_xy, outer_xy, high_layer],
                 [outer_xy, outer_xy, high_layer]]

    elif layout_id == 2:
        N = 160
        V_fix = [[-35, 0, 1], [-30, 0, 1], [-25, 0, 1], [-20, 0, 1], [-15, 0, 1], [-10, 0, 1], [-5, 0, 1], [0, 0, 1],
                 [5, 0, 1], [10, 0, 1], [15, 0, 1], [20, 0, 1], [25, 0, 1], [30, 0, 1], [35, 0, 1]]
        V_fix += np.column_stack((np.expand_dims(np.multiply(np.array([68]), np.random.rand(N)) - 34, axis=1),
                                  np.array([[1.9, 1.5]] * N))).tolist()
        V_fix += np.column_stack(
            (np.multiply(np.array([68]), np.random.rand(N)) - 34, np.array([[-1.9, 1.5]] * N))).tolist()

    elif layout_id == 3:
        V_fix = [[-2, -2, 1], [2, 2, 1]]

    elif layout_id == 4:
        V_fix = [[-4.5, 0, 1], [4.5, 0, 1]]

    elif layout_id == 5:

        X = 4.5

        w1 = np.array([[-4, -3, -2, -1,  0,  1,  2,  3,  4],
                       [-X, -X, -X, -X, -X, -X, -X, -X, -X],
                       [ 2,  2,  2,  2,  2,  2,  2,  2,  2]]).T

        w2 = np.array([[-4, -3, -2, -1,  0,  1,  2,  3, 4],
                       [ X,  X,  X,  X,  X,  X,  X,  X, X],
                       [ 2,  2,  2,  2,  2,  2,  2,  2, 2]]).T

        w3 = np.array([[-X, -X, -X, -X, -X, -X, -X, -X, -X],
                       [-4, -3, -2, -1,  0,  1,  2,  3,  4],
                       [ 2,  2,  2,  2,  2,  2,  2,  2,  2]]).T

        w4 = np.array([[ X,  X,  X,  X,  X,  X,  X,  X, X],
                       [-4, -3, -2, -1,  0,  1,  2,  3, 4],
                       [ 2,  2,  2,  2,  2,  2,  2,  2, 2]]).T

        return np.row_stack((w1, w2, w3, w4))

    elif layout_id == 6:
        V_fix = [[-4.5, 1, 1],  [-4.5, -1, 1], [4.5, -1, 1], [4.5, 1, 1]]

    elif layout_id == 7:
        V_fix = [[1,1,1], [-0.5,-1,0.75], [-1.2,-0.5,0.5], [-1.2, 0.5,0.5]]

    elif layout_id == 8:
        X = 3.5
        H = 1.4

        w1 = np.array([[-2, -1,  0,  1,  2],
                       [-X, -X, -X, -X, -X],
                       [ H,  H,  H,  H,  H]]).T

        w2 = np.array([[-2, -1,  0,  1,  2],
                       [ X,  X,  X,  X,  X],
                       [ H,  H,  H,  H,  H]]).T

        w3 = np.array([[-X, -X, -X, -X, -X],
                       [-2, -1,  0,  1,  2],
                       [ H,  H,  H,  H,  H]]).T

        w4 = np.array([[ X,  X,  X,  X,  X],
                       [-2, -1,  0,  1,  2],
                       [ H,  H,  H,  H,  H]]).T

        w5 = np.array([[X-0.5,  X-0.5, -(X-0.5), -(X-0.5)],
                       [X-0.5, -(X-0.5), X-0.5, -(X-0.5)],
                       [H,  H, H,  H]]).T

        return np.row_stack((w1, w2, w3, w4, w5))

    elif layout_id == 9:
        "AR City demo"
        X = 3
        H = 1.4

        w1 = np.array([[-2, -1,  0,  1,  2],
                       [-X, -X, -X, -X, -X],
                       [ H,  H,  H,  H,  H]]).T

        w2 = np.array([[-2, -1,  0,  1,  2],
                       [ X,  X,  X,  X,  X],
                       [ H,  H,  H,  H,  H]]).T

        w3 = np.array([[-X, -X, -X, -X, -X],
                       [-2, -1,  0,  1,  2],
                       [ H,  H,  H,  H,  H]]).T

        w4 = np.array([[ X,  X,  X,  X,  X],
                       [-2, -1,  0,  1,  2],
                       [ H,  H,  H,  H,  H]]).T

        #w5 = np.array([[X-0.5,  X-0.5, -(X-0.5), -(X-0.5)],
        #               [X-0.5, -(X-0.5), X-0.5, -(X-0.5)],
        #               [H,  H, H,  H]]).T

        return np.row_stack((w1, w2, w3, w4))

    else:
        c_print("The id exedes the number of layouts. Empty fix vertex list was returned", "yellow")
        return np.array([])
    return np.array(V_fix)


class Scenario:
    def __init__(self,
                 dimension: np.ndarray,
                 time_step: float,
                 occupancy_matrix_time_interval: float,
                 vertex_number: int,
                 fix_vertex_layout_id: int,
                 min_vertex_distance: float,
                 max_edge_length: float,
                 point_cloud_density: float,
                 real_dimension: None | np.ndarray = None,
                 use_GPU: bool = False,
                 terrain: Terrain | None = None,
                 static_obstacles: StaticObstacleS | None = None,
                 radars: RadarS | None = None
                 ):
        self.dimension: np.ndarray = dimension # x, y, z_min, z_max [m]
        self.real_dimension = real_dimension
        self.time_step: float = time_step
        self.occupancy_matrix_time_interval = occupancy_matrix_time_interval
        self.vertex_number: int = vertex_number
        self.min_vertex_distance: float = min_vertex_distance
        self.max_edge_legth: float = max_edge_length
        self.point_cloud_density: float = point_cloud_density
        self.use_GPU: bool = use_GPU

        # Environment
        self.terrain: Terrain | None = terrain
        self.static_obstacles: StaticObstacleS | None = static_obstacles
        self.radars: RadarS | None = radars
        self.dynamic_obstacles: list[DynamicObstacle] | None = None # Added after initialization,
        self.leader_plane: None | Leader_plane = None               # because the scenario has to be plotted
                                                                    # for defining the paths of the obstacles

        # Possible targets of the drones
        self.num_of_obstacle_targets = 0
        self.target_positions: np.ndarrray = target_layout_set(fix_vertex_layout_id)
        if self.static_obstacles is not None:
            self.place_targets_above_obstacles()

        # To be filled with self.build()
        self.graph: nx.Graph | None = None
        self.target_vertices: np.ndarray | None = None
        self.extra_target_vertices: np.ndarray | None = None # Unique target set for a second group of drones
        self.point_cloud: np.ndarray | None = None

        self.build()

        # Updated by the drones and the dynamic obstacles to store wich parts of the grap are occupied by them
        self.ocm_times: np.ndarray = np.arange(0, self.occupancy_matrix_time_interval + self.time_step, self.time_step)
        self.occupied_edges: np.ndarray | None = None
        self.occupied_vertices: np.ndarray | None = None

    # ----- GRAPH -----
    def build(self) -> None:
        self.check_target_positions()
        self.target_vertices = np.arange(0, len(self.target_positions), 1)
        self.setup_graph()
        self.generate_point_cloud()

        if self.radars:
            self.graph = self.radars.add_risk(graph=self.graph, point_cloud=self.point_cloud)

        if self.real_dimension is not None:
            self.split_graph()
            self.split_targets()

    def place_targets_above_obstacles(self):
        if self.static_obstacles.target_elevation is not None and len(self.static_obstacles.enclosed_spaces) > 0:
            pos = self.static_obstacles.enclosed_spaces[:, :3] + [0, 0, self.static_obstacles.target_elevation]
            self.num_of_obstacle_targets = len(pos)
            if len(self.target_positions) == 0:
                self.target_positions = pos
            else:
                self.target_positions = np.row_stack((pos, self.target_positions))

    def check_target_positions(self) -> None:
        # --------------------------- OUT OF FLY ZONE ---------------------------
        X_out = np.logical_or(self.target_positions[:, 0] < (-self.dimension[0] / 2),
                              self.target_positions[:, 0] > (self.dimension[0] / 2))
        Y_out = np.logical_or(self.target_positions[:, 1] < (-self.dimension[1] / 2),
                              self.target_positions[:, 1] > (self.dimension[1] / 2))
        Z_out = np.logical_or(self.target_positions[:, 2] < (-self.dimension[2]),
                              self.target_positions[:, 2] > (self.dimension[3]))
        out = np.logical_or(Z_out, np.logical_or(X_out, Y_out))

        if out.any():
            c_print("-----------------------------------------------------------------", "yellow")
            c_print(f"Some tagets are deleted because they were out of fly zone: "
                    f"X:{[-self.dimension[0]/2,self.dimension[0]/2]} "
                    f"Y:{[-self.dimension[1]/2,self.dimension[1]/2]} "
                    f"Z:{[-self.dimension[2],self.dimension[3]]}", "yellow")
            c_print(f"Deleted targets:\n{self.target_positions[out]}", "yellow")
            c_print("-----------------------------------------------------------------", "yellow")
            self.target_positions = self.target_positions[np.logical_not(out)]

        # --------------------------- INSIDE TERRAIN ---------------------------
        if self.terrain:
            self.target_positions, deleted = self.terrain.delte_vertices_inside(vertices=self.target_positions,
                                                                                return_deleted=True)
            if deleted.size != 0:
                c_print("-----------------------------------------------------------------", "yellow")
                c_print(f"Some targets are deleted because they were inside the terrain", "yellow")
                c_print(f"Deleted targets:\n{deleted}", "yellow")
                c_print("-----------------------------------------------------------------", "yellow")

        # --------------------------- INSIDE STATIC OBSTACLES ---------------------------
        if self.static_obstacles:
            self.target_positions, deleted = self.static_obstacles.delete_vertices_inside(vertices=self.target_positions,
                                                                                          return_deleted=True)
            if deleted.size != 0:
                c_print("-----------------------------------------------------------------", "yellow")
                c_print(f"Some targets are deleted because they were inside the obstacles", "yellow")
                c_print(f"Deleted targets:\n{deleted}", "yellow")
                c_print("-----------------------------------------------------------------", "yellow")

    def setup_graph(self) -> None:
        def remove_redundant_vertices(number_of_fix_vertices: int, V: np.ndarray, thres: float) -> np.ndarray:
            """
            Remove vertices which are closer to other vertices than the threshold value. The fix vertices (targets) will be not
             removed.

            :param number_of_fix_vertices: int -> the number of vertices that should not be deleted
            :param V: array([[x,y,z]...[x,y,z]]) -> coordinates of all vertices (including targets)
            :param thres: float -> The minimum distance between the vertices
            :return: vertices: The remaining vertices with the desired minimum distace from each other.
            """
            removal = []
            for k in range(number_of_fix_vertices, len(V) - 2):  # Removing vertices that are too close
                if min(np.linalg.norm(V[k, :] - V[k + 1:, :], axis=1)) < thres:
                    removal.append(k)
                    continue
                if number_of_fix_vertices == 0:
                    continue
                if min(np.linalg.norm(V[k, :] - V[:number_of_fix_vertices], axis=1)) < thres:
                    removal.append(k)
            return np.delete(V, removal, axis=0)

        # --------------------------- VERTICES ---------------------------
        # GENERATE:
        # The generation done whithin 2 steps to increase the chance to an evenly filled space
        graph = nx.Graph()
        random_vertices = np.random.uniform(low=[-self.real_dimension[0]/2, -self.real_dimension[1]/2, self.real_dimension[2]],
                                            high=[self.real_dimension[0]/2, self.real_dimension[1]/2, self.real_dimension[3]],
                                            size=(int(self.vertex_number/2), 3))
        vertices = np.row_stack((self.target_positions, random_vertices))
        vertices = remove_redundant_vertices(number_of_fix_vertices=len(self.target_positions),
                                             V=vertices, thres=self.min_vertex_distance)
        random_vertices = np.random.uniform(low=[-self.dimension[0]/2, -self.dimension[1]/2, self.dimension[2]],
                                            high=[self.dimension[0]/2, self.dimension[1]/2, self.dimension[3]],
                                            size=(int(self.vertex_number/2), 3))
        vertices = np.row_stack((vertices, random_vertices))
        vertices = remove_redundant_vertices(number_of_fix_vertices=len(self.target_positions),
                                             V=vertices, thres=self.min_vertex_distance)
        random_vertices = np.random.uniform(low=[-self.dimension[0]/2, -self.dimension[1]/2, self.dimension[2]],
                                            high=[self.dimension[0]/2, self.dimension[1]/2, self.dimension[3]],
                                            size=(int(self.vertex_number/2), 3))
        vertices = np.row_stack((vertices, random_vertices))
        vertices = remove_redundant_vertices(number_of_fix_vertices=len(self.target_positions),
                                             V=vertices, thres=self.min_vertex_distance)

        if self.terrain:
            vertices = self.terrain.delte_vertices_inside(vertices=vertices)
        if self.static_obstacles:
            vertices = self.static_obstacles.delete_vertices_inside(vertices=vertices)

        # LOAD TO GRAPH:
        for i, position in enumerate(vertices):
            graph.add_node(i, pos=position)

        # --------------------------- BASE EDGES ---------------------------
        # GENERATE:
        tri = Delaunay(vertices)  # Delaunay triangulation of a set of points
        # tri:[A,B,C],[A,B,D] -> Adj_graph:A[B,C,D],B[A,C,D],C[A,B],D[A,B]
        graph_adj = [list() for _ in range(len(vertices))]
        for simplex in tri.simplices:
            for j in range(4):
                a = simplex[j]
                graph_adj[a].extend(simplex)
                graph_adj[a].remove(a)
                graph_adj[a] = list(dict.fromkeys(graph_adj[a])) # olny unique values

        # LOAD TO GRAPH:
        for vertex, neighbours in enumerate(graph_adj):
            for neighbour in neighbours:
                length = np.linalg.norm(graph.nodes.data('pos')[vertex] - graph.nodes.data('pos')[neighbour])
                graph.add_edge(vertex, neighbour, weight=length)

        if self.terrain:
            graph = self.terrain.delet_edges_inside(graph=graph)
        if self.static_obstacles:
            graph = self.static_obstacles.delete_edges_inside(graph=graph)

        # --------------------------- FINE GRAPH ---------------------------
        # The original edges are divided into nearly uniform lengths measured by the maximum edge length
        self.graph = nx.Graph()

        for vertex in graph.nodes.data('pos'):
            self.graph.add_node(vertex[0], pos=vertex[1], detection_risk=0, type=0)

        for edge in graph.edges.data('weight'):
            vertex_1 = graph.nodes.data('pos')[edge[0]]
            vertex_2 = graph.nodes.data('pos')[edge[1]]
            edge_length = edge[2]

            number_of_plus_vertices = math.ceil(edge_length / self.max_edge_legth)+1
            plus_vertices_on_edge = np.linspace(vertex_1, vertex_2, number_of_plus_vertices)[1:-1]

            index_start = len(self.graph.nodes())  # prepare to add more vertices
            index_stop = len(plus_vertices_on_edge)

            # if there is no need for adding an extra vertex to the edge
            if index_stop == 0:
                self.graph.add_edge(edge[0], edge[1], weight=edge_length, detection_risk=0)
                continue
            for i, position in enumerate(plus_vertices_on_edge):
                self.graph.add_node(index_start + i, pos=position, detection_risk=0, type=1)
                if i == 0:
                    self.graph.add_edge(edge[0], index_start + i, weight=np.linalg.norm(vertex_1 - position),
                                        detection_risk=0)
                else:
                    self.graph.add_edge(index_start + i - 1, index_start + i,
                                        weight=np.linalg.norm(plus_vertices_on_edge[i - 1] - position), detection_risk=0)
            self.graph.add_edge(index_start + i, edge[1], weight=np.linalg.norm(vertex_2 - position), detection_risk=0)

    def expand_graph(self, position: np.ndarray, connection_range: float):
        # --------------------------- NEW VERTEX ---------------------------
        new_vertex = len(self.graph.nodes())
        self.graph.add_node(new_vertex, pos=position, detection_risk=0, type=2)

        # --------------------------- NEW EDGES ---------------------------
        vertices = np.array([v for v in self.graph.nodes
                             if self.graph.nodes.data('type')[v] == 0
                             and np.linalg.norm(self.graph.nodes.data('pos')[v]-position) <= connection_range])
                             #and not self.static_obstacles.intersect(self.graph.nodes.data('pos')[v], position,
                             #                                        self.static_obstacles.safe_enclosed_spaces())])
        if len(vertices) < 3:
            c_print("!!! WARNING: Connection range was too small so it had to be increased !!!", "yellow")
            while len(vertices) < 3:
                connection_range = connection_range + self.min_vertex_distance
                vertices = np.array([v for v in self.graph.nodes
                                     if self.graph.nodes.data('type')[v] == 0
                                     and np.linalg.norm(self.graph.nodes.data('pos')[v] - position) <= connection_range])
                                     #and not self.static_obstacles.intersect(self.graph.nodes.data('pos')[v], position,
                                     #                                        self.static_obstacles.safe_enclosed_spaces())])
            c_print(f"             Connection range was set to {connection_range}", "yellow")

        # plot added vertex and edges
        #ax = plt.gca()
        #for v in vertices:
        #    ax.plot(*np.array([position, self.graph.nodes.data('pos')[v]]).T, color="red", alpha=1, lw=0.5)
        #plt.pause(0.000001)

        for v in vertices:
            distance = np.linalg.norm(self.graph.nodes.data('pos')[v]-position)
            number_of_plus_vertices = math.ceil(distance / self.min_vertex_distance)+1
            plus_vertices_on_edge = np.linspace(position, self.graph.nodes.data('pos')[v], number_of_plus_vertices)[1:-1]

            index_start = len(self.graph.nodes())  # prepare to add more vertices
            index_stop = len(plus_vertices_on_edge)

            if index_stop == 0:
                self.graph.add_edge(new_vertex, v, weight=distance, detection_risk=0, index=-1)
                continue
            for i, pos in enumerate(plus_vertices_on_edge):
                self.graph.add_node(index_start + i, pos=pos, detection_risk=0, type=2)
                if i == 0:
                    self.graph.add_edge(new_vertex, index_start + i, weight=np.linalg.norm(pos - position),
                                        detection_risk=0, index=-1)
                else:
                    self.graph.add_edge(index_start + i - 1, index_start + i,
                                        weight=np.linalg.norm(plus_vertices_on_edge[i - 1] - pos), detection_risk=0, index=-1)
            self.graph.add_edge(index_start + i, v, weight=np.linalg.norm(self.graph.nodes.data('pos')[v] - pos), detection_risk=0, index=-1)

        # --------------------------- EXTEND POINT CLOUD ---------------------------
        j = self.point_cloud.shape[0]
        point_number = math.ceil(2 * self.max_edge_legth / self.point_cloud_density)
        self.point_cloud = np.concatenate((self.point_cloud,
                                           np.zeros([len(self.graph.edges())-self.point_cloud.shape[0], point_number,
                                                     3])))
        for edge in self.graph.edges:
            if self.graph[edge[0]][edge[1]]['index'] != -1:
                continue
            self.graph[edge[0]][edge[1]]['index'] = j
            self.point_cloud[j] = np.linspace(self.graph.nodes.data('pos')[edge[0]],
                                              self.graph.nodes.data('pos')[edge[1]], point_number)
            j += 1

    def split_graph(self):
        low = [-self.real_dimension[0] / 2, -self.real_dimension[1] / 2, self.real_dimension[2]],
        high = [self.real_dimension[0] / 2, self.real_dimension[1] / 2, self.real_dimension[3]]
        for n in self.graph:
            pos = self.graph.nodes[n]['pos']
            if np.all(low < pos) and np.all(high > pos):
                self.graph.nodes[n]['Virtual'] = False
            else:
                self.graph.nodes[n]['Virtual'] = True

    def restore_graph_and_OCM(self):
        remove = [vertex[0] for vertex in self.graph.nodes.data('type') if vertex[1] == 2]
        for verex in remove:
            self.graph.remove_node(verex)
        self.point_cloud = self.point_cloud[:len(self.graph.edges()), :, :]

        self.occupied_vertices = self.occupied_vertices[:, :, :len(self.graph.nodes())]
        self.occupied_edges = self.occupied_edges[:, :, :len(self.graph.edges())]

    def generate_point_cloud(self) -> None:
        point_number = math.ceil(2 * self.max_edge_legth / self.point_cloud_density)
        self.point_cloud = np.zeros([len(self.graph.edges()), point_number, 3])
        for i, edge in enumerate(self.graph.edges):
            self.graph[edge[0]][edge[1]]['index'] = i
            self.point_cloud[i] = np.linspace(self.graph.nodes.data('pos')[edge[0]],
                                              self.graph.nodes.data('pos')[edge[1]], point_number)

    def real_target(self, idx):
        if self.static_obstacles is not None:
            if idx < self.static_obstacles.num_of_real:
                return True
        return False

    def split_targets(self):
        real_targets = []
        virtual_targets = []
        for vertex in self.target_vertices:
            if self.real_target(vertex):
                real_targets.append(vertex)
            else:
                virtual_targets.append(vertex)
        self.target_vertices = np.array(real_targets)
        self.extra_target_vertices = np.array(virtual_targets)

    # ----- OCM -----
    def init_OCM_CPU(self, dynamic_obj) -> None:
        """
        - Add the drone to the OCM (in case of the first drone create the OCM)
        - The drone stands at the home position
        """
        t0 = time.time()
        width = dynamic_obj.bounding_box[0]
        height = dynamic_obj.bounding_box[1]

        # -------- VERTICES --------
        vertices = np.array([self.graph.nodes.data('pos')[i] for i in self.graph.nodes])

        inside_x = np.logical_and((dynamic_obj.position[0] - width) < vertices[:, 0],
                                  vertices[:, 0] < (dynamic_obj.position[0] + width))
        inside_y = np.logical_and((dynamic_obj.position[1] - width) < vertices[:, 1],
                                  vertices[:, 1] < (dynamic_obj.position[1] + width))
        inside_z = np.logical_and((dynamic_obj.position[2] - height) < vertices[:, 2],
                                  vertices[:, 2] < (dynamic_obj.position[2] + height))

        occupied_vertices = np.logical_and(np.logical_and(inside_x, inside_y), inside_z)
        occupied_vertices = np.expand_dims(np.full((len(self.ocm_times), len(occupied_vertices)), occupied_vertices),
                                           axis=0)

        if self.occupied_vertices is None:
            self.occupied_vertices = occupied_vertices
        else:
            self.occupied_vertices = np.concatenate((self.occupied_vertices, occupied_vertices), axis=0)

        # -------- EDGES --------
        inside_x = np.logical_and((dynamic_obj.position[0] - width) < self.point_cloud[:, :, 0],
                                  self.point_cloud[:, :, 0] < (dynamic_obj.position[0] + width))
        inside_y = np.logical_and((dynamic_obj.position[1] - width) < self.point_cloud[:, :, 1],
                                  self.point_cloud[:, :, 1] < (dynamic_obj.position[1] + width))
        inside_z = np.logical_and((dynamic_obj.position[2] - height) < self.point_cloud[:, :, 2],
                                  self.point_cloud[:, :, 2] < (dynamic_obj.position[2] + height))
        inside = np.logical_and(np.logical_and(inside_x, inside_y), inside_z)

        occupied_edges = inside.any(axis=1)  # if any point on the edge is inside an obstacle then the edge is occupied
        occupied_edges = np.expand_dims(np.full((len(self.ocm_times), len(occupied_edges)), occupied_edges), axis=0)

        if self.occupied_edges is None:
            self.occupied_edges = occupied_edges
        else:
            self.occupied_edges = np.concatenate((self.occupied_edges, occupied_edges), axis=0)
        dynamic_obj.print_info(f"OCM generation took {time.time()-t0} sec")

    # TODO: implement it
    def init_OCM_GPU(self, dynamic_obj) -> None:
        ...

    def update_OCM_CPU(self, dynamic_obj, t_update: float) -> None:

        positions = dynamic_obj.move(self.ocm_times[self.ocm_times >= t_update])

        width = dynamic_obj.bounding_box[0]
        height = dynamic_obj.bounding_box[1]

        # -------- VERTICES --------
        vertices = np.array([self.graph.nodes.data('pos')[i] for i in self.graph.nodes])
        inside = np.full((len(positions), len(vertices)), True)
        for i, position in enumerate(positions):
            inside_x = np.logical_and((position[0] - width) < vertices[:, 0], vertices[:, 0] < (position[0] + width))
            inside_y = np.logical_and((position[1] - width) < vertices[:, 1], vertices[:, 1] < (position[1] + width))
            inside_z = np.logical_and((position[2] - height) < vertices[:, 2], vertices[:, 2] < (position[2] + height))
            inside[i] = np.logical_and(np.logical_and(inside_x, inside_y), inside_z)
        self.occupied_vertices[dynamic_obj.ID][self.ocm_times >= t_update] = inside

        # -------- EDGES --------
        edges, points = np.shape(self.point_cloud)[:2]
        inside = np.full((len(positions), edges, points), True)
        for i, position in enumerate(positions):
            inside_x = np.logical_and((position[0] - width) < self.point_cloud[:, :, 0],
                                      self.point_cloud[:, :, 0] < (position[0] + width))
            inside_y = np.logical_and((position[1] - width) < self.point_cloud[:, :, 1],
                                      self.point_cloud[:, :, 1] < (position[1] + width))
            inside_z = np.logical_and((position[2] - height) < self.point_cloud[:, :, 2],
                                      self.point_cloud[:, :, 2] < (position[2] + height))
            inside[i] = np.logical_and(np.logical_and(inside_x, inside_y), inside_z)
        self.occupied_edges[dynamic_obj.ID][self.ocm_times >= t_update] = inside.any(axis=2)

    # TODO: implement it
    def update_OCM_GPU(self, dynamic_obj, t_update: np.ndarray) -> None:
        ...

    def step_OCM(self, drones: list, t: float) -> None:
        t = round(math.floor(t / self.time_step) * self.time_step, 5)
        if self.ocm_times[0] < t:
            # SHIFT TIME
            t_prev_last = self.ocm_times[-1]
            ocm_times_copy = copy.deepcopy(self.ocm_times)
            self.ocm_times = np.arange(t, t + self.occupancy_matrix_time_interval + self.time_step, self.time_step)

            t_add = self.ocm_times[t_prev_last < self.ocm_times]
            if len(t_add) == 0:
                "The last time step of OCM is already represents the full planning horizont"
                "But the first time steps of OCMs are passed"
                passed_times = len(ocm_times_copy)-len(self.ocm_times)
                self.occupied_edges = self.occupied_edges[:, passed_times:, :]
                self.occupied_vertices = self.occupied_vertices[:, passed_times:, :]
                return

            # SHIFT MATRICES
            add_matrix = np.full((np.shape(self.occupied_vertices)[0],
                                  len(t_add),
                                  np.shape(self.occupied_vertices)[2]), True)
            self.occupied_vertices = np.concatenate((self.occupied_vertices, add_matrix),
                                                    axis=1)[:, -len(self.ocm_times):, :]
            add_matrix = np.full((np.shape(self.occupied_edges)[0],
                                  len(t_add),
                                  np.shape(self.occupied_edges)[2]), True)
            self.occupied_edges = np.concatenate((self.occupied_edges, add_matrix),
                                                 axis=1)[:, -len(self.ocm_times):, :]

            for drone in drones:
                if drone.trajectory is None:
                    continue
                if self.use_GPU:
                    self.update_OCM_GPU(dynamic_obj=drone, t_update=t_add[0])
                else:
                    self.update_OCM_CPU(dynamic_obj=drone, t_update=t_add[0])
            if self.dynamic_obstacles is not None:
                for obs in self.dynamic_obstacles:
                    if obs.ID is None:
                        continue
                    if self.use_GPU:
                        self.update_OCM_GPU(dynamic_obj=obs, t_update=t_add[0])
                    else:
                        self.update_OCM_CPU(dynamic_obj=obs, t_update=t_add[0])

    def prepare_OCM(self, drone_ID: int, t: float) -> (np.ndarray, np.ndarray):
        remaining_times = self.ocm_times >= t
        self.occupied_vertices = self.occupied_vertices[:, remaining_times]
        self.occupied_edges = self.occupied_edges[:, remaining_times]
        self.ocm_times = self.ocm_times[remaining_times]

        self.occupied_vertices[drone_ID] = False
        self.occupied_edges[drone_ID] = False

        return self.occupied_vertices.any(axis=0), self.occupied_edges.any(axis=0)

    def expand_OCM(self, drones):
        obstacles = drones + self.dynamic_obstacles
        vertices = np.array([self.graph.nodes.data('pos')[i] for i in self.graph.nodes
                             if self.graph.nodes.data('type')[i] == 2])
        extra_edges = len(self.graph.edges()) - self.occupied_edges.shape[2]
        point_cloud = self.point_cloud[-extra_edges:]
        inside_v = np.full((len( obstacles), len(self.ocm_times), len(vertices)), True)
        inside_e = np.full((len( obstacles), len(self.ocm_times), extra_edges, len(self.point_cloud[0])), True)

        for obs in obstacles:
            positions = obs.move(self.ocm_times)
            width = obs.bounding_box[0]
            height = obs.bounding_box[1]

            for i, position in enumerate(positions):
                # -------- VERTICES --------
                inside_x = np.logical_and((position[0] - width) < vertices[:, 0],
                                          vertices[:, 0] < (position[0] + width))
                inside_y = np.logical_and((position[1] - width) < vertices[:, 1],
                                          vertices[:, 1] < (position[1] + width))
                inside_z = np.logical_and((position[2] - height) < vertices[:, 2],
                                          vertices[:, 2] < (position[2] + height))
                inside_v[obs.ID, i] = np.logical_and(np.logical_and(inside_x, inside_y), inside_z)

                # -------- EDGES --------
                inside_x = np.logical_and((position[0] - width) < point_cloud[:, :, 0],
                                          point_cloud[:, :, 0] < (position[0] + width))
                inside_y = np.logical_and((position[1] - width) < point_cloud[:, :, 1],
                                          point_cloud[:, :, 1] < (position[1] + width))
                inside_z = np.logical_and((position[2] - height) < point_cloud[:, :, 2],
                                          point_cloud[:, :, 2] < (position[2] + height))
                inside_e[obs.ID, i] = np.logical_and(np.logical_and(inside_x, inside_y), inside_z)
        self.occupied_vertices = np.concatenate((self.occupied_vertices, inside_v), axis=2)
        self.occupied_edges = np.concatenate((self.occupied_edges, inside_e.any(axis=3)), axis=2)

    # ----- UTILITY -----

    def print_data(self):
        print(f"Vertex number: {len(self.graph.nodes)}")
        print(f"Edge number: {len(self.graph.edges)}")
        print("Number of points in point cloud:", np.size(self.point_cloud))
        sum_L = 0
        max_L = 0
        min_L = 1000000000
        for edge in self.graph.edges.data():
            sum_L = sum_L + edge[2]['weight']
            if max_L < edge[2]['weight']:
                max_L = edge[2]['weight']
            if min_L > edge[2]['weight']:
                min_L = edge[2]['weight']
        print("Average edge length:", sum_L / len(self.graph.edges), "m")
        print("Longest edge is: ", max_L, "m")
        print("Shortest edge is: ", min_L, "m")

    def plot(self,
             plt_real_area: bool = False,
             plt_targets: bool = True,
             plt_graph: bool = False,
             alpha_graph: float = 0.2,
             plt_terrain: bool = True,
             plt_radars: bool = True,
             plt_static_obstacles: bool = True,
             alpha_static_obstacles: float = 1.0,
             plt_dynamic_obstacle_paths: bool = False) -> None:

        plot_fly_zone(self.dimension)

        ax = plt.gca()

        if plt_real_area:
            # Define parameters
            width, length, h1, h2 = self.real_dimension

            # Define vertices
            vertices = np.array([[-width / 2, -length / 2, h1], [width / 2, -length / 2, h1],
                                 [width / 2, length / 2, h1], [-width / 2, length / 2, h1],
                                 [-width / 2, -length / 2, h2], [width / 2, -length / 2, h2],
                                 [width / 2, length / 2, h2], [-width / 2, length / 2, h2]])

            # Define edges
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            edge_lines = np.array([(vertices[i], vertices[j]) for i, j in edges])

            # Plot
            ax.add_collection3d(Line3DCollection(edge_lines, colors='k', alpha=0.3))

        if plt_targets and self.target_positions.size != 0:

            for i, position in enumerate(self.target_positions):
                if self.real_target(i):
                    color = "blue"
                else:
                    color = "black"
                ax.scatter(position[0], position[1], position[2],
                           s=40, alpha=1, c=color, marker=f"${i}$") # marker=f"${i}$"

        if plt_graph:
            edges = get_edges(self.graph)
            for edge in edges:
                ax.plot(*edge.T, color="black", linewidth=0.5, alpha=alpha_graph)
            for node in self.graph.nodes.data():
                if node[1]['Virtual']:
                    color = 'grey'
                else:
                    color = 'black'
                ax.scatter(node[1]['pos'][0], node[1]['pos'][1], node[1]['pos'][2], s=1, alpha=1, c=color)

        if plt_terrain and self.terrain is not None:
            self.terrain.plot()

        if plt_static_obstacles and self.static_obstacles is not None:
            self.static_obstacles.plot(alpha=alpha_static_obstacles)
            self.static_obstacles.plot(alpha_static_obstacles/5, safety_zone=True)

        if plt_radars and self.radars is not None:
            self.radars.plot()

        if plt_dynamic_obstacle_paths:
            print("plot paths") # TODO: implement