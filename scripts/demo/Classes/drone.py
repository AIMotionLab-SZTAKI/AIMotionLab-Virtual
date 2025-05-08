from __future__ import annotations

import numpy as np
from scipy import interpolate
import math
import sys
import heapq
import time
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .scenario import Scenario
from Utils.Utils import c_print, sphere_surface, plot_edge, angle_between_points


class Node:
    def __init__(self,
                 parent, # Node
                 idx: int,
                 speed: float,
                 graph: nx.Graph,
                 time_step: float,
                 target_idx: int,
                 max_speed: float,
                 modified: bool,
                 waiting_time: float = np.inf,
                 greed_factor: float = 1.1, # test with 1 too
                 ):
        self.parent = parent
        self.idx = idx
        self.speed = speed

        self.time_to_reach: float = np.inf
        self.traverse_time: float = np.inf
        self.time_to_goal: float = self.heuristic(graph, target_idx, max_speed)
        self.get_time_to_reach(graph=graph, time_step=time_step, waiting_time=waiting_time)

        self.risk_save = self.risk(graph, waiting_time)

        if parent:
            self.g = parent.g + self.risk_save + self.traverse_time
        else:
            self.g = self.risk_save + self.time_to_reach

        self.cost = self.g + self.time_to_goal*greed_factor
        self.ID = self.set_id(modified)

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.cost < other.cost

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.cost > other.cost

    def get_time_to_reach(self, graph: nx.Graph, time_step, waiting_time) -> None:
        if self.parent is None:
            self.time_to_reach = 0.0
        elif self.idx == self.parent.idx:
            self.time_to_reach = math.ceil((self.parent.time_to_reach + waiting_time)/time_step)*time_step
        else:
            edge_length = graph.edges[self.parent.idx, self.idx]['weight']
            self.traverse_time = (2 * edge_length) / (self.parent.speed + self.speed)
            self.traverse_time = math.ceil(self.traverse_time/time_step)*time_step
            self.time_to_reach = self.parent.time_to_reach + self.traverse_time

    def risk(self, graph: nx.Graph, waiting_time: float) -> float:
        if self.parent is None:
            return graph.nodes[self.idx]['detection_risk']
        elif self.parent.idx == self.idx:
            return graph.nodes[self.idx]['detection_risk'] * waiting_time
        return graph.edges[self.parent.idx, self.idx]['detection_risk'] * self.traverse_time

    def heuristic(self, graph: nx.Graph, target_idx: int, max_speed: float) -> float:
        v0 = graph.nodes.data('pos')[target_idx]
        v1 = graph.nodes.data('pos')[self.idx]
        return math.sqrt((v0[0]-v1[0])**2 + (v0[1]-v1[1])**2 + (v0[2]-v1[2])**2) / max_speed

    def set_id(self, modified: bool) -> str:
        if modified:
            return f"{self.idx}_{self.speed}_{self.time_to_reach}"
        else:
            return f"{self.idx}"


class Drone:
    def __init__(self,
                 ID: int,
                 radius: float,
                 down_wash: float,
                 safety_distance: float,
                 speeds: np.ndarray[float],
                 turnrate_bounds: np.ndarray[int],
                 waiting_time: float,
                 reaction_time: float,
                 color: str = 'blue',
                 mute: bool = False):
        self.ID: int = ID
        self.radius: float = radius
        self.bounding_box = [2*radius+safety_distance, max(down_wash, 2*radius)+safety_distance]
        self.down_wash: float = down_wash
        self.safety_distance: float = safety_distance
        self.speeds: np.ndarray[float] = speeds
        self.turnrate_bounds: np.ndarray[float] = turnrate_bounds
        self.waiting_time: float = waiting_time
        self.reaction_time: float = reaction_time
        self.color: str = color
        self.silent: bool = mute

        self.intermediate_vertex: None | int = None # If the planning cannot finish set it to the best avalible vertex
                                                    # The next planning will sart from here to the original target

        self.virtual = False
        self.lock_home_vertex = False
        self.home_vertex: None | int = None
        self.start_vertex: None | int = None
        self.target_vertex: None | int = None
        self.returned_home: bool = False
        self.landed: bool = False

        self.position: None | np.ndarray = None
        self.trajectory: None | np.ndarray = None

        self.surface = sphere_surface(radius, resolution=5)
        self.my_plot = None
        self.trajectory_plot: list = []

        #self.ocm_diagnostics = np.array([])
        self.succes_rate = [0, 0]
        self.wait_others = 0
        self.traj_gen_time = 0

        self.route = []

    def place(self, scenario: Scenario, vertex: int, lock_home_vertex: bool = False) -> None:
        """
        - Define the home position of the drone.
        - Update the OCM accordingly.
        - Lock down the defined vertex if neccesary so other drones will not use it as target.
        """
        self.home_vertex = vertex
        self.start_vertex = vertex

        if lock_home_vertex:
            self.lock_home_vertex = True
        scenario.target_vertices = np.delete(scenario.target_vertices, scenario.target_vertices == vertex)
        scenario.extra_target_vertices = np.delete(scenario.extra_target_vertices, scenario.extra_target_vertices == vertex)
        self.position = scenario.graph.nodes.data('pos')[vertex]

        scenario.init_OCM_GPU(dynamic_obj=self) if scenario.use_GPU else scenario.init_OCM_CPU(dynamic_obj=self)

    def choose_random_target(self, scenario: Scenario) -> None:
        """
        Choose a new target for the drone and lock it from the others
        """
        if self.target_vertex is not None:
            self.start_vertex = self.target_vertex

        if not self.virtual:
            self.target_vertex = np.random.choice(scenario.target_vertices)
        else:
            self.target_vertex = np.random.choice(scenario.extra_target_vertices)
        #scenario.target_vertices = np.delete(scenario.target_vertices, scenario.target_vertices == self.target_vertex)

        #if not (self.lock_home_vertex and (self.home_vertex == self.start_vertex)):
        #    scenario.target_vertices = np.append(scenario.target_vertices, self.start_vertex)

    def go_to_new_target(self, start_time: float, scenario: Scenario, drones: list, go_home: bool = False):
        scenario.step_OCM(drones=drones, t=start_time)
        OCM_v, OCM_e = scenario.prepare_OCM(drone_ID=self.ID, t=start_time)

        if go_home:
            if self.target_vertex is None:
                sys.exit(f"The demo time was not enough for drone(s): {self.ID}... to fly out.")
            self.start_vertex = self.target_vertex
            self.target_vertex = self.home_vertex

        else:
            self.choose_random_target(scenario)
        try:
            self.print_info(f"Go from v{self.start_vertex} = {scenario.graph.nodes.data('pos')[self.start_vertex]} "
                            f"to v{self.target_vertex} = {scenario.graph.nodes.data('pos')[self.target_vertex]}")
        except KeyError:
            print("debug")

        t0 = time.time()
        final_node = self.A_star(scenario, OCM_e, OCM_v)
        if final_node.idx == self.home_vertex:
            self.returned_home = True
        self.print_info(f"Suces_rte = {self.succes_rate}")
        route = self.get_route(final_node, scenario)
        self.traj_gen_time = time.time() - t0
        self.print_info(f"A* done under {self.traj_gen_time} sec")
        self.fit_trajectory(route, start_time)

        if scenario.use_GPU:
            scenario.update_OCM_GPU(dynamic_obj=self, t_update=scenario.ocm_times[0])
        else:
            scenario.update_OCM_CPU(dynamic_obj=self, t_update=scenario.ocm_times[0])

        self.print_info(f"Trajectory start: {self.trajectory_start_time()} sec")
        self.print_info(f"Trajectory end: {self.trajectory_final_time()} sec")
        self.print_info(f"Trajectory duration: {self.trajectory_final_time() - self.trajectory_start_time()} sec")

    def avoid_collision(self,  scenario: Scenario, t: float, drones: list):
        t0 = time.time()
        # ----- PREPARATION -----
        scenario.expand_graph(position=self.move(t + self.reaction_time), connection_range=1.5) # TODO: hazard
        scenario.expand_OCM(drones=drones)
        speed = np.linalg.norm(interpolate.splev(t + self.reaction_time, self.trajectory, der=1))

        # ----- PATH PLANNING -----
        OCM_v, OCM_e = scenario.prepare_OCM(drone_ID=self.ID, t=t)
        origin_vertex = self.start_vertex
        self.start_vertex = len(scenario.graph.nodes())-1
        final_node = self.A_star(scenario, OCM_e, OCM_v, init_speed=speed)
        route = self.get_route(final_node, scenario)
        self.fit_trajectory(route, t + self.reaction_time)

        # ----- RESTORATION -----
        self.start_vertex = origin_vertex
        scenario.restore_graph_and_OCM()

        if scenario.use_GPU:
            scenario.update_OCM_GPU(dynamic_obj=self, t_update=scenario.ocm_times[0])
        else:
            scenario.update_OCM_CPU(dynamic_obj=self, t_update=scenario.ocm_times[0])

        self.print_info(f"Emrgency planning took {time.time()-t0} sec")

    def A_star(self, scenario: Scenario, OCM_e: np.ndarray, OCM_v: np.ndarray | None = None, init_speed: float = 0,
               modified: bool = True) -> Node | None:
        # ----- PRE CHECKS -----
        if self.start_vertex == self.target_vertex:
            self.print_info(f"Stayed in place at vertex: {self.start_vertex}, "
                            f"coordinates: {scenario.graph.nodes.data('pos')[self.start_vertex]}")
            sys.exit("The Start and the Target cannot be the same vertex")

        if modified and 0 in self.speeds and OCM_v is None:
            sys.exit("For a modified search, enter as input the occupied vertices OCM_v")

        self.succes_rate[0] += 1

        # ----- DEBUG: EDGE PLOT -----
        search_time = 2
        re_plan_delay = 2
        plot_search = False
        #if self.start_vertex == 1327:
        #    plot_search = True
        #    search_time = 10

        if plot_search:
            ax = plt.gca()
            edge_plots = []
            edge_plots.extend(plot_edge(ax=ax, graph=scenario.graph, v1=self.start_vertex, v2=self.target_vertex, color='g'))

        # ----- INIT -----
        time_step = scenario.time_step # TODO: can be other
        max_speed = self.speeds.max(initial=0)
        start_node = Node(parent=None, idx=self.start_vertex, speed=init_speed,
                          time_step=time_step,
                          max_speed=max_speed,
                          graph=scenario.graph,
                          target_idx=self.target_vertex,
                          waiting_time=self.waiting_time,
                          modified=modified)
        if init_speed == 0:
            start_node.time_to_reach = 1 #search_time
        open_list = [start_node]
        closed_set = set()
        hovering_possible = 0 in self.speeds

        safety_node = Node(parent=start_node, idx=self.start_vertex, speed=init_speed,
                           time_step=time_step,
                           max_speed=max_speed,
                           graph=scenario.graph,
                           target_idx=self.target_vertex,
                           waiting_time=re_plan_delay,
                           modified=modified)

        t_search_0 = time.time()

        # ----- SEARCH -----
        while len(open_list) > 0:
            if search_time < (time.time()-t_search_0):
                break
            current_node = heapq.heappop(open_list)

            safety_time = search_time+2
            st = self.waiting_time
            self.waiting_time = safety_time
            if current_node.idx == self.target_vertex and \
                    not self.collisin_on_vertex(OCM_v=OCM_v, node=current_node, scenario=scenario):
                if plot_search:
                    # noinspection PyUnboundLocalVariable
                    for edge in edge_plots:
                        edge.remove()
                self.succes_rate[1] += 1
                if not (self.lock_home_vertex and (self.home_vertex == self.start_vertex)):
                    if not self.virtual:
                        scenario.target_vertices = np.append(scenario.target_vertices, self.start_vertex)
                    else:
                        scenario.extra_target_vertices = np.append(scenario.extra_target_vertices, self.start_vertex)
                if not self.virtual:
                    scenario.target_vertices = np.delete(scenario.target_vertices,
                                                         scenario.target_vertices == self.target_vertex)
                else:
                    scenario.extra_target_vertices = np.delete(scenario.extra_target_vertices,
                                                               scenario.extra_target_vertices == self.target_vertex)
                self.waiting_time = st
                return current_node
            self.waiting_time = st

            if current_node.ID in closed_set:
                continue
            closed_set.add(current_node.ID)

            #if current_node.ID != start_node.ID:
            #    safety_node = self.update_safety_node(current_node, safety_node, scenario, OCM_v)

            for child_idx in list(scenario.graph[current_node.idx].keys()):
                # if the drone is real, use only the observable part of the environment
                if (not self.virtual) and scenario.graph.nodes[child_idx]['Virtual']:
                    continue

                for i, speed in enumerate(self.speeds):

                    # traditional A* use only the max speed
                    if not modified and speed != max_speed:
                        continue

                    # can not travel on edge with 0 speed
                    if current_node.speed == 0 and speed == 0:
                        continue

                    # Stop at target
                    if not hovering_possible:
                        pass
                    elif not modified and child_idx == self.target_vertex:
                        speed = 0
                    elif child_idx == self.target_vertex and speed != 0:
                        continue

                    child_node = Node(parent=current_node, idx=child_idx, speed=speed,
                                      time_step=time_step,
                                      max_speed=max_speed,
                                      graph=scenario.graph,
                                      target_idx=self.target_vertex,
                                      waiting_time=self.waiting_time,
                                      modified=modified)

                    if child_node.time_to_reach+child_node.time_to_goal > scenario.occupancy_matrix_time_interval:
                        continue

                    if not self.feasible_maneuver(node=child_node, scenario=scenario):
                        continue

                    if self.collisin_on_edge(OCM_e=OCM_e, node=child_node, scenario=scenario):
                        if plot_search:
                            # noinspection PyUnboundLocalVariable
                            edge_plots.extend(plot_edge(ax=ax, graph=scenario.graph, v1=child_idx,
                                                        v2=current_node.idx, color='r'))
                            plt.pause(0.000001)
                        continue
                    self.revard_edge(node=child_node, scenario=scenario)

                    if child_node.ID not in closed_set:
                        if plot_search:
                            edge_plots.extend(plot_edge(ax=ax, graph=scenario.graph, v1=child_idx,
                                                        v2=current_node.idx, color='b'))
                            plt.pause(0.000001)
                        heapq.heappush(open_list, child_node)

            if modified and current_node.speed == 0:
                if current_node.time_to_reach + self.waiting_time > scenario.occupancy_matrix_time_interval:
                    #self.print_info('!!! WARNING !!!: A* reached the end of the time representation')
                    continue

                if self.collisin_on_vertex(OCM_v=OCM_v, node=current_node, scenario=scenario):
                    continue
                child_idx = current_node.idx
                child_node = Node(parent=current_node, idx=child_idx, speed=0,
                                  time_step=time_step,
                                  max_speed=max_speed,
                                  graph=scenario.graph,
                                  target_idx=self.target_vertex,
                                  waiting_time=self.waiting_time,
                                  modified=modified)
                self.revard_vertex(node=child_node, scenario=scenario)

                if child_node.ID not in closed_set:
                    heapq.heappush(open_list, child_node)

        self.print_info("No complete path found")
        self.intermediate_vertex = safety_node.idx
        self.target_vertex = safety_node.idx
        return safety_node

    def update_safety_node(self, node: Node, safety_node: Node, scenario, OCM_v):
        safety_time = 6

        if node.time_to_goal <= safety_node.time_to_goal \
                and node.time_to_reach + safety_time < scenario.occupancy_matrix_time_interval \
                and node.speed == 0\
                and (node.idx != safety_node.idx or node.time_to_reach < safety_node.time_to_reach):
            st = self.waiting_time
            self.waiting_time = safety_time
            if not self.collisin_on_vertex(OCM_v=OCM_v, node=node, scenario=scenario):
                    safety_node = node
            self.waiting_time = st
        return safety_node

    def feasible_maneuver(self, node: Node, scenario: Scenario) -> bool:
        p1 = scenario.graph.nodes.data('pos')[node.idx]
        p2 = scenario.graph.nodes.data('pos')[node.parent.idx]

        # Check turn rate
        if node.parent.parent is None:
            return True
        p3 = scenario.graph.nodes.data('pos')[node.parent.parent.idx]
        angle = 180-angle_between_points(A=p1, B=p2, C=p3)
        if angle > self.turnrate_bounds[node.parent.speed == self.speeds][0]:
            return False
        return True

    @staticmethod
    def collisin_on_edge(OCM_e: np.ndarray, node: Node, scenario: Scenario) -> bool:
        edge_idx = scenario.graph.edges[node.idx, node.parent.idx]['index']
        t1 = round(node.time_to_reach/scenario.time_step)
        t0 = round(node.parent.time_to_reach/scenario.time_step)
        return OCM_e[t0:t1, edge_idx].any()

    def collisin_on_vertex(self, OCM_v: np.ndarray, node: Node, scenario: Scenario) -> bool:
        t0 = round(node.time_to_reach/scenario.time_step)
        t1 = round((node.time_to_reach + self.waiting_time)/scenario.time_step)
        return OCM_v[t0:t1, node.idx].any()

    @staticmethod
    def revard_edge(node: Node, scenario: Scenario) -> None:
        if scenario.leader_plane is not None:
            edge_idx = scenario.graph.edges[node.idx, node.parent.idx]['index']
            t1 = round((node.time_to_reach+scenario.ocm_times[0])/scenario.time_step)
            t0 = round((node.parent.time_to_reach+scenario.ocm_times[0])/scenario.time_step)
            if len(scenario.leader_plane.edge_revards) > t0 and not scenario.leader_plane.edge_revards[t0:t1, edge_idx].any():
                node.g += scenario.leader_plane.revard
            elif not scenario.leader_plane.edge_revards[-1, edge_idx]:
                node.g += scenario.leader_plane.revard

    @staticmethod
    def revard_vertex(node: Node, scenario: Scenario) -> None:
        if scenario.leader_plane is not None:
            t1 = round((node.time_to_reach+scenario.ocm_times[0])/scenario.time_step)
            t0 = round((node.parent.time_to_reach+scenario.ocm_times[0])/scenario.time_step)
            if len(scenario.leader_plane.vertex_revards) > t0 and not scenario.leader_plane.vertex_revards[t0:t1, node.idx].any():
                node.g += scenario.leader_plane.revard
            elif not scenario.leader_plane.vertex_revards[-1, node.idx]:
                node.g += scenario.leader_plane.revard

    #@staticmethod
    def get_route(self, node: Node, scenario: Scenario) -> np.ndarray:
        route = []
        risk_of_route = 0
        while node.parent is not None:
            route += [np.append(np.append(node.time_to_reach, scenario.graph.nodes[node.idx]['pos']), node.speed)]
            node = node.parent
            risk_of_route += node.risk_save
        route += [np.append(np.append(node.time_to_reach, scenario.graph.nodes[node.idx]['pos']), node.speed)]
        self.route = np.flip(np.array(route), axis=0)
        route = [wp[:4] for wp in route]
        risk_of_route += node.risk_save
        self.print_info(f"Risk of route:{risk_of_route}")
        return np.flip(np.array(route), axis=0)

    # TODO: Line buffer is not a good solution. It causes high acceleration small movements instead of howering.
    def fit_trajectory(self, route: np.ndarray, t: float) -> None:
        LINE_BUFFER = 5
        spline_points = np.array([])
        for i in range(len(route) - 1):
            if i == 0:
                spline_points = np.linspace(route[i], route[i + 1], 2 + LINE_BUFFER)[:-1]
            else:
                spline_points = np.row_stack((spline_points, np.linspace(route[i], route[i + 1], 2 + LINE_BUFFER)[:-1]))
        spline_points = np.row_stack((spline_points, route[-1]))
        self.trajectory, *_ = interpolate.splprep([spline_points[:, 1], spline_points[:, 2], spline_points[:, 3]], u=spline_points[:, 0] + t, k=3, s=0)
        #self.trajectory, *_ = interpolate.splprep([route[:, 1], route[:, 2], route[:, 3]], u=route[:, 0] + t, k=3, s=0)

    def trajectory_start_time(self) -> int | float:
        if self.trajectory:
            return round(self.trajectory[0][0], 5)
        return -1

    def trajectory_final_time(self) -> int | float:
        if self.trajectory:
            return round(self.trajectory[0][-1], 5)
        return -1

    def move(self, t: np.ndarray | float) -> np.ndarray:
        t = np.where(t > self.trajectory_start_time(), t, self.trajectory_start_time())
        t = np.where(t < self.trajectory_final_time(), t, self.trajectory_final_time())
        position = interpolate.splev(t, self.trajectory)
        return np.transpose(position)

    def land(self, scenario, t: float) -> None:
        self.landed = True

        # Visual update
        fig = plt.gcf()
        ax = fig.gca()
        for axes in self.trajectory_plot:
            axes.remove()
        self.my_plot.set_alpha(0.1)

        # Landing trajectory route = np.array([[t,x,y,z],[t,x,y,z]]), t = 0
        final_pos = self.move(t=self.trajectory_final_time())
        h = final_pos[2] + self.bounding_box[1]
        land_pos = final_pos - np.array([0,0,h])
        landing_time = 3 # under the surface to compensate for the upper half of the downwash
        landing_route = np.concatenate((np.array([[t], [t+landing_time]]), np.array([final_pos,land_pos])), axis=1)
        self.fit_trajectory(route=landing_route, t=0)


    def plot_trajectory(self) -> None:
        fig = plt.gcf()
        ax = fig.gca()

        dt = 0.1
        t = np.arange(self.trajectory_start_time(), self.trajectory_final_time() + dt, dt)
        pos = self.move(t)

        points = np.array([pos[:, 0], pos[:, 1], pos[:, 2]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        for axes in self.trajectory_plot:
            try:
                axes.remove()
            except ValueError:
                print("")

        self.trajectory_plot = [
            ax.scatter(points[0][0][0], points[0][0][1], points[0][0][2], s=10, alpha=1, c='green'),
            ax.scatter(points[-1][0][0], points[-1][0][1], points[-1][0][2], s=10, alpha=1, c='red'),
            ax.add_collection3d(Line3DCollection(segments, alpha=0.5, lw=1, color=self.color))]

    def plot(self) -> None:
        fig = plt.gcf()
        ax = fig.gca()
        self.position = self.move(t=np.array(self.trajectory_start_time()))
        self.my_plot = ax.add_collection3d(Poly3DCollection(self.surface + self.position, alpha=1,
                                                            facecolors=self.color))

    def animate(self, t: float) -> None:
        if self.my_plot is None:
            self.plot()
        self.position = self.move(np.array(t))
        self.my_plot.set_verts(self.surface + self.position)

    def print_info(self, info: str) -> None:
        if not self.silent:
            c_print(f"Drone {self.ID}: {info}", self.color)
