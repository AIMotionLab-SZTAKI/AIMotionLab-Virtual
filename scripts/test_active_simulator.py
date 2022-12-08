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

simulator.graphics_step = graphics_step


# init scenario





# generate trajectories
for d in simulator.drones:
    d.print_info()
    print()
    traj = Trajectory(simulator.model, simulator.data, control_step)
    d.trajectories = traj


#print(simulator.data.qpos)
#fps_limiter = FpsLimiter(1.0 / control_step)
i = 0

simulator.drones[0].set_ctrl(np.array((5.13685937e+00, -1.13470652e-01, 1.99699326e-02, 4.82128511e-04)))
simulator.start = time.time()
while not simulator.glfw_window_should_close():
    #fps_limiter.begin_frame()

    data = simulator.update(i)

    #print("site: " + str(data.site("virtbumblebee_hooked_0_cog")))
    #print("ctrl: " + str(data.ctrl))

    simulator.log()

    i += 1
    #fps_limiter.end_frame()

simulator.plot_log()

simulator.save_log()

simulator.close()