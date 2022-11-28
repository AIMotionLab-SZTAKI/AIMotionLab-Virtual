import numpy as np
import os
from classes.active_simulation import ActiveSimulator
import classes.drone as drone
import time
from util.util import sync, FpsLimiter


# init drones


# init simulator
simulator = ActiveSimulator(os.path.join("..", "xml_models", "built_scene.xml"), record_video=[0, 1], connect_to_optitrack=False)

for d in simulator.drones:
    d.print_info()
    print()


# init scenario





# generate trajectories
#simulator.drones[0].trajectories

fps_limiter = FpsLimiter(1.0 / simulator.timestep)

while not simulator.glfw_window_should_close():
    fps_limiter.begin_frame()

    data = simulator.update()

    simulator.log()

    fps_limiter.end_frame()

simulator.plot_log()

simulator.save_log()

simulator.close()