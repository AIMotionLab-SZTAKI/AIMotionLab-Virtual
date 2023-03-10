import os

from classes.active_simulation import ActiveSimulator

from util import xml_generator

from classes.drone import Drone, DroneMocap
from classes.car import Car, CarMocap

from classes.controller_base import DummyDroneController, DummyCarController
from classes.trajectory_base import DummyDroneTrajectory, DummyCarTrajectory


RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"


xml_path = os.path.join("..", "xml_models")
# scene.xml is required
xml_base_filename = "scene.xml"
save_filename = "built_scene.xml"


# create xml with a drone and a car
scene = xml_generator.SceneXmlGenerator(os.path.join(xml_path, xml_base_filename))
drone0_name = scene.add_drone("1 1 1", "1 0 0 0", RED_COLOR, True, "bumblebee", True)
car0_name = scene.add_car("-0.5 1 0.2", "1 0 0 0", RED_COLOR, True)

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers
virt_parsers = [Drone.parse, Car.parse]
mocap_parsers = [DroneMocap.parse, CarMocap.parse]


control_step, graphics_step = 0.01, 0.02
xml_filename = os.path.join(xml_path, save_filename)

# initializing simulator
simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, virt_parsers, mocap_parsers, connect_to_optitrack=False)

# grabbing the drone and the car
drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
car0 = simulator.get_MovingObject_by_name_in_xml(car0_name)

# creating trajectory and controller for drone0
drone0_trajectory = DummyDroneTrajectory()
drone0_controller0 = DummyDroneController(drone0.mass, drone0.inertia, simulator.gravity)
drone0_controller1 = DummyDroneController(drone0.mass, drone0.inertia, simulator.gravity)

drone0_controllers = [drone0_controller0, drone0_controller1]

# creating trajectory and controller for car0
car0_trajectory = DummyCarTrajectory()
car0_controller0 = DummyCarController(car0.mass, car0.inertia, simulator.gravity)
car0_controller1 = DummyCarController(car0.mass, car0.inertia, simulator.gravity)

car0_controllers = [car0_controller0, car0_controller1]

def update_controller_type(state, setpoint, time, i):
    # return the index of the controller in the list?
    return 0

# setting update_controller_type method, trajectory and controller for drone0
drone0.set_update_controller_type_method(update_controller_type)
drone0.set_trajectory(drone0_trajectory)
drone0.set_controllers(drone0_controllers)

# setting update_controller_type method, trajectory and controller for car0
car0.set_update_controller_type_method(update_controller_type)
car0.set_trajectory(car0_trajectory)
car0.set_controllers(car0_controllers)


# start simulation
i = 0
#car0.d = 0.1
#car0.set_steer_angle(0.1)
while not simulator.glfw_window_should_close():
    simulator.update(i)
    #print(car0.get_state())

    i += 1

simulator.close()