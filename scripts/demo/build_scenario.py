import math

import numpy as np
import matplotlib.pyplot as plt

from Classes.scenario import Scenario
from Classes.terrain import Terrain
from Classes.static_obstacle import StaticObstacleS
from Classes.dynamic_obstacle import DynamicObstacle
from Classes.radar import RadarS
from Classes.drone import Drone
from Classes.leader_plane import Leader_plane
from Utils.Utils import save, get_trajectories, save_trajectories, set_rand_seed

#-----------------------------------------------------------------------------------------------------------------------
# INITIALIZE SCENARIO

DIMENSION = np.array([6.4, 6.4, 0.1, 3])
REAL_DIMENSION = np.array([3, 3, 0.1, 1.4])
#corridor: np.array([10.0, 4.0, 0.2, 3.0])
#city: np.array([3, 3, 0.1, 1.5])
#multidrone_area: np.array([10, 10, 0.1, 4])

#set_rand_seed(1764389414)
set_rand_seed(17638914)

#-----------------------------------------------------------------------------------------------------------------------
# OPTIONAL (1/2):

#terrain = Terrain(dimension=DIMENSION,
#                  max_height=1.5,
#                  file_path="Saves/Maps/dead_river.png",
#                  safety_distance=0.01)

static_obstacles = StaticObstacleS(layout_id=6, # Add more options in Classes/startic_obstacle/obstacle_layout_set
                                   safety_distance=0.2,
                                   target_elevation=0.3,
                                   measurement_file="new_measurement",
                                   new_measurement=True,
                                   real_obs_base={'bu': 0.3,
                                                  'cf': 0.1,
                                                  'ob': 0.1},
                                   )

#radars = RadarS(layout_id=4) # Modify in Classes/radar/radar_layout_set

#-----------------------------------------------------------------------------------------------------------------------
# DEFINE SCENARIO
scenario = Scenario(dimension=DIMENSION,
                    real_dimension=REAL_DIMENSION,
                    time_step=0.1, # TODO: set it automaticaly
                    occupancy_matrix_time_interval=30.0,
                    vertex_number=50,
                    max_edge_length=0.5,
                    min_vertex_distance=0.2,
                    fix_vertex_layout_id=9,
                    point_cloud_density=0.05, # TODO: set it automaticaly
                    #terrain=terrain,
                    static_obstacles=static_obstacles,
                    #radars=radars
                    )

scenario.plot(plt_graph=False,
              plt_terrain=True,
              plt_targets=True,
              plt_static_obstacles=True,
              plt_dynamic_obstacle_paths=False,
              alpha_static_obstacles=1)
scenario.print_data()

drone_colors = ['blue', 'red', 'green', 'orange', 'lime']
drones = [Drone(ID=i, # do not modify
                radius=0.12,
                down_wash=1,
                safety_distance=0.15,
                speeds=np.array([0, 0.25, 0.5, 1]),
                turnrate_bounds=np.array([180, 150, 90, 30]),
                waiting_time=0.5,
                reaction_time=0.0,
                color='blue' #drone_colors[i] if i < len(drone_colors) else 'grey'
                ) for i in range(math.floor(len(scenario.target_positions)/2))] # max num of drones is N/2
                                                                                  # every drone has a home possition
                                                                                  # not accesable for the others

#-----------------------------------------------------------------------------------------------------------------------
# OPTIONAL (2/2):

trajectories = get_trajectories(#file_name="100_trajectories_corridor",
                                #add_in_area=[-6.0, 6.0, -5, 5, 0, 4]
                                )

#save_trajectories(trajectories, "uknown_trajectories_corridor")
scenario.dynamic_obstacles = [DynamicObstacle(radius=0.5,
                                              height=3.2,
                                              start_time=5,
                                              rest_time=max(1.0, drones[0].reaction_time),
                                              safety_distance=0.15 + drones[0].radius,
                                              trajectory_list=trajectory_list,
                                              repeate_movement=False
                                              ) for i, trajectory_list in enumerate(trajectories)]

#-----------------------------------------------------------------------------------------------------------------------
# SAVE
name = "city_scenario"

save(data={'Scenario': scenario, 'Drones': drones},
     file=f"Saves/Scenarios/{name}")

plt.show()
