import numpy as np
import os
from classes.active_simulation import ActiveSimulator
import classes.drone as drone
import time
from util.util import sync, FpsLimiter
from classes.trajectory import Trajectory


# init drones


# init simulator

sim_step, control_step, graphics_step = 0.001, 0.01, 0.02

simulator = ActiveSimulator(os.path.join("..", "xml_models", "built_scene.xml"), [0, 1], sim_step, control_step, graphics_step, connect_to_optitrack=False)

simulator.timestep = graphics_step


# init scenario





# generate trajectories
for d in simulator.drones:
    d.print_info()
    print()
    traj = Trajectory(simulator.model, simulator.data, sim_step, control_step, graphics_step)
    d.trajectories = traj

#simulator.drones[0].trajectories

print(simulator.data.qpos)
#fps_limiter = FpsLimiter(1.0 / control_step)
i = 0
simulator.start = time.time()
while not simulator.glfw_window_should_close():
    #fps_limiter.begin_frame()

    data = simulator.update(i)

    simulator.log()

    i += 1
    #fps_limiter.end_frame()

simulator.plot_log()

simulator.save_log()

simulator.close()