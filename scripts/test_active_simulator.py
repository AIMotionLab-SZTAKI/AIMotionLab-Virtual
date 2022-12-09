import numpy as np
import os
from classes.active_simulation import ActiveSimulator
import classes.drone as drone
import time
import sys
from util.util import sync, FpsLimiter
from classes.trajectory import Trajectory

sys.path.insert(1, '/home/mate/Desktop/mujoco/crazyflie-mujoco/')
sys.path.insert(2, '/home/crazyfly/Desktop/mujoco_digital_twin/crazyflie-mujoco/')

from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl

HOOK_UP_3_LOADS = "hook up 3 loads"
FLY = "fly"

SCENARIO = FLY
#SCENARIO = HOOK_UP_3_LOADS

# init simulator

if SCENARIO == HOOK_UP_3_LOADS:

    sim_step, control_step, graphics_step = 0.001, 0.01, 0.02

    simulator = ActiveSimulator(os.path.join("..", "xml_models", "built_scene_3loads.xml"), [0, 1], sim_step, control_step, graphics_step, connect_to_optitrack=False)

elif SCENARIO == FLY:
    sim_step, control_step, graphics_step = 0.001, 0.01, 0.02

    simulator = ActiveSimulator(os.path.join("..", "xml_models", "built_scene_fly.xml"), [0, 1], sim_step, control_step, graphics_step, connect_to_optitrack=False)

simulator.cam.azimuth, simulator.cam.elevation = 70, -20
simulator.cam.distance = 3



# generate trajectories
for d in simulator.drones:

    d.print_info()
    print()

    if SCENARIO == HOOK_UP_3_LOADS:


        controller = RobustGeomControl(simulator.model, simulator.data, drone_type='large_quad')
        controller.delta_r = 0
        controller_lqr = PlanarLQRControl(simulator.model)
        controllers = {"geom" : controller, "lqr" : controller_lqr}
        traj = Trajectory(control_step, HOOK_UP_3_LOADS)

        d.set_qpos(traj.pos_ref[0, :], traj.q0)

        if isinstance(d, drone.DroneHooked):
            d.set_hook_qpos(0)
        else:
            print("Error: drone is not hooked")
        d.set_mass(controller.mass)
        d.set_trajectory(traj)
        d.set_controllers(controllers)
    
    elif SCENARIO == FLY:

        controller = GeomControl(simulator.model, simulator.data, drone_type='large_quad')

        controllers = {"geom" : controller}

        traj = Trajectory(control_step, FLY)

        d.set_qpos(np.array((0.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0, 0.0)))

        d.set_trajectory(traj)
        d.set_controllers(controllers)



i = 0

simulator.start_time = time.time()
while not simulator.glfw_window_should_close():

    data = simulator.update(i)


    simulator.log()

    i += 1

simulator.plot_log()

simulator.save_log()

simulator.close()