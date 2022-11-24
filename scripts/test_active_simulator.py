import numpy as np
import os
from classes.active_simulation import ActiveSimulator
import classes.drone as drone


# init drones


# init simulator
simulator = ActiveSimulator(os.path.join("..", "xml_models", "built_scene.xml"), record_video=[0, 1], connect_to_optitrack=False)

for d in simulator.drones:
    d.print_info()
    print()


# init scenario





# generate trajectories
#simulator.drones[0].trajectories



while not simulator.glfw_window_should_close():

    data = simulator.update()

    simulator.log()

simulator.plot_log()

simulator.save_log()

simulator.close()