import numpy as np
import os
from classes.active_simulation import ActiveSimulator


# init drones


# init simulator
simulator = ActiveSimulator(os.path.join("..", "xml_models", "scene.xml"), [], record_video=[0, 1], connect_to_optitrack=False)




# init scenario





# generate trajectories
#simulator.drones[0].trajectories



while not simulator.glfw_window_should_close():

    data = simulator.update()

    simulator.log()

simulator.plot_log()

simulator.save_log()

simulator.close()