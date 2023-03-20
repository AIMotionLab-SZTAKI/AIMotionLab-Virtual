import numpy as np
import os
from classes.active_simulation import ActiveSimulator
import classes.drone as drone
import classes.car as car
import time
import sys
from util.util import sync, FpsLimiter
from util import xml_generator
import classes.trajectory as traj

sys.path.insert(1, '/home/mate/Desktop/mujoco/crazyflie-mujoco/')
sys.path.insert(2, '/home/crazyfly/Desktop/mujoco_digital_twin/crazyflie-mujoco/')

from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl


RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"


SCENARIO = traj.HOOK_UP_3_LOADS
#SCENARIO = traj.FLY


# init simulator

abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_filename = "scene.xml"
save_filename = "built_scene.xml"

scene = xml_generator.SceneXmlGenerator(xml_base_filename)

virt_parsers = [drone.Drone.parse, car.Car.parse]
mocap_parsers = [drone.DroneMocap.parse, car.CarMocap.parse]


if SCENARIO == traj.HOOK_UP_3_LOADS:

    # adding the necessary objects to the scene
    drone0_name = scene.add_drone("1 1 1", "1 0 0 0", RED_COLOR, True, "bumblebee", True)
    #scene.add_drone("0 1 0", "1 0 0 0", BLUE_COLOR, False)
    #scene.add_drone("-1 -1 1", "1 0 0 0", BLUE_COLOR, False, "bumblebee", False)
    car0_name = scene.add_car("-0.5 1 0.2", "1 0 0 0", RED_COLOR, True)
    #scene.add_car("0.5 1 0.2", "1 0 0 0", BLUE_COLOR, False)
    scene.add_load("0 0 0", ".1 .1 .1", ".15", "1 0 0 0", "0.1 0.1 0.9 1.0")
    scene.add_load("-.6 .6 0", ".075 .075 .075", ".05", "1 0 0 0", "0.1 0.9 0.1 1.0")
    scene.add_load("-.3 -.6 0", ".075 .075 .1", ".1", "1 0 0 0", "0.9 0.1 0.1 1.0")

    # saving the scene as xml so that the simulator can load it
    scene.save_xml(os.path.join(xml_path, save_filename))

    control_step, graphics_step = 0.01, 0.02

    # initializing simulator
    simulator = ActiveSimulator(os.path.join(xml_path, save_filename), None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False)

    simulator.onBoard_elev_offset = 15

    # creating controllers and trajectory objects for the drone
    controller = RobustGeomControl(simulator.model, simulator.data, drone_type='large_quad')
    controller.delta_r = 0
    controller_lqr = PlanarLQRControl(simulator.model)
    controllers = {"geom" : controller, "lqr" : controller_lqr}
    traj_ = traj.TestTrajectory(control_step, traj.HOOK_UP_3_LOADS)


    d0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)

    # setting initial position of the drone
    d0.set_qpos(traj_.pos_ref[0, :], traj_.q0)
    # quick check if the drone indeed has a hook
    if isinstance(d0, drone.DroneHooked):
        d0.set_hook_qpos(0)
    else:
        print("Error: drone is not hooked")
    
    d0.set_mass(controller.mass)
    # binding trajectory and controllers to the drone
    d0.set_trajectory(traj_)
    d0.set_controllers(controllers)

    car_0 = simulator.get_MovingObject_by_name_in_xml(car0_name)
    car_0.max_vel = 1
    car_0.cacc = 0.05
    def up_press():
        car_0.up_pressed = True
    def up_release():
        car_0.up_pressed = False
    def down_press():
        car_0.down_pressed = True
    def down_release():
        car_0.down_pressed = False
    def left_press():
        car_0.left_pressed = True
    def left_release():
        car_0.left_pressed = False
    def right_press():
        car_0.right_pressed = True
    def right_release():
        car_0.right_pressed = False

    simulator.set_key_up_callback(up_press)
    simulator.set_key_up_release_callback(up_release)
    simulator.set_key_down_callback(down_press)
    simulator.set_key_down_release_callback(down_release)
    simulator.set_key_left_callback(left_press)
    simulator.set_key_left_release_callback(left_release)
    simulator.set_key_right_callback(right_press)
    simulator.set_key_right_release_callback(right_release)

elif SCENARIO == traj.FLY:
    # no objects needed except a drone
    scene.add_drone("1 1 1", "1 0 0 0", RED_COLOR, True, "bumblebee", False)

    scene.save_xml(os.path.join(xml_path, save_filename))

    # initializing simulator
    control_step, graphics_step = 0.01, 0.04
    #simulator = ActiveSimulator(os.path.join(xml_path, save_filename), [0, 1, 2, 3], control_step, graphics_step, connect_to_optitrack=False)
    simulator = ActiveSimulator(os.path.join(xml_path, save_filename), None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False)

    controller = GeomControl(simulator.model, simulator.data, drone_type='large_quad')

    controllers = {"geom" : controller}

    traj_ = traj.TestTrajectory(control_step, traj.FLY)

    # grabbing the drone from the list of simulated drones
    # this time it only contains one drone as only one was added to the xml up top

    virt_drones, virt_cars = get_virtdrones_virtcars(simulator.all_virt_vehicles) 
    d0 = virt_drones[0]

    d0.set_qpos(np.array((0.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0, 0.0)))

    d0.set_trajectory(traj_)
    d0.set_controllers(controllers)


simulator.cam.lookat = [0, 0, 1]
simulator.cam.azimuth, simulator.cam.elevation = 70, -20
simulator.cam.distance = 3



for d in simulator.all_vehicles:
    d.print_info()
    print()




#controller = GeomControl(simulator.model, simulator.data)
#
#controllers = {"geom" : controller}
#
#traj_ = traj.TestTrajectory(control_step, traj.FLIP)
#
#d1.set_qpos(np.array((0.0, 0.0, 0.5)), np.array((1.0, 0.0, 0.0, 0.0)))
#
#d1.set_trajectory(traj_)
#d1.set_controllers(controllers)

i = 0

while not simulator.glfw_window_should_close():

    data = simulator.update(i)


    simulator.log()

    i += 1

simulator.plot_log()

simulator.save_log()

simulator.close()