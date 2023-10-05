import os
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.trajectory.hooked_drone_trajectory import HookedDroneTrajectory
from aiml_virtual.controller.drone_geom_control import GeomControl
from aiml_virtual.controller.hooked_drone_lq_control import LqrLoadControl
from aiml_virtual.object.drone import BUMBLEBEE_PROP, DRONE_TYPES
import numpy as np
import matplotlib.pyplot as plt
from aiml_virtual.airflow.airflow_sampler import AirflowSampler
from aiml_virtual.object import parseMovingObjects
from aiml_virtual.util import plot_payload_and_airflow_volume


def update_controller_type(state, setpoint, time, i):
    # return the index of the controller in the list?
    return 0


if __name__ == '__main__':

    RED_COLOR = "0.85 0.2 0.2 1.0"
    BLUE_COLOR = "0.2 0.2 0.85 1.0"

    abs_path = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(abs_path, "..", "xml_models")
    xmlBaseFileName = "scene_base.xml"
    save_filename = "built_scene.xml"
    
    rod_length = float(BUMBLEBEE_PROP.ROD_LENGTH.value)

    # Set scenario parameters
    drone_init_pos = np.array([-0.76, 1.13, 1, 0])  # initial drone position and yaw angle
    load_init_pos = np.array([-0.01, -1, 0.1995 + rod_length])  # TODO: Do some transformations in z direction
    load_target_pos = np.array([0.76, 1.13, 0.19 + rod_length])
    load_mass = 0.03


    # create xml with a drone and a car
    scene = SceneXmlGenerator(xmlBaseFileName)
    drone0_name = scene.add_drone(np.array2string(drone_init_pos[0:3])[1:-2], "1 0 0 0", RED_COLOR, DRONE_TYPES.BUMBLEBEE_HOOKED, 1)
    payload0_name = scene.add_payload(np.array2string(load_init_pos)[1:-2], ".05 .05 .025", str(load_mass), "1 0 0 0", BLUE_COLOR)

    # saving the scene as xml so that the simulator can load it
    scene.save_xml(os.path.join(xml_path, save_filename))

    # create list of parsers
    virt_parsers = [parseMovingObjects]

    control_step, graphics_step = 0.01, 0.02
    xml_filename = os.path.join(xml_path, save_filename)

    # initializing simulator
    simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step)

    # grabbing the drone and the car
    drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
    payload0 = simulator.get_MovingObject_by_name_in_xml(payload0_name)

    # creating trajectory and controller for drone0
    drone0_trajectory = HookedDroneTrajectory()
    drone0_trajectory.set_control_step(control_step)
    drone0_trajectory.set_rod_length(drone0.rod_length)
    drone0_controller = LqrLoadControl(drone0.mass, drone0.inertia, simulator.gravity)
    drone0_controller.L = rod_length

    drone0_controllers = [drone0_controller]

    # setting update_controller_type method, trajectory and controller for drone0
    drone0.set_update_controller_type_method(update_controller_type)
    drone0.set_trajectory(drone0_trajectory)
    drone0.set_controllers(drone0_controllers)

    pressure_lut_path = os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_pressure_shifted.txt")
    velocity_lut_path = os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_velocity_shifted.txt")

    airflow_sampl = AirflowSampler(pressure_lut_path, drone0, velocity_lut_path)
    payload0.create_surface_mesh(0.00001)

    # start simulation
    sensor_data = []
    q_data = []
    drone0.qvel[0] = 0
    drone0.qvel[1] = 0

    ctrl0_max = 0
    ctrl1_max = 0
    ctrl2_max = 0
    ctrl3_max = 0

    def is_greater_than(new_value, current_max):

        if new_value > current_max:
            return new_value
        
        return current_max


    # Plan trajectory
    drone0_trajectory.construct(drone_init_pos, load_init_pos - np.array([-.02, 0, rod_length + 0.04]), load_target_pos - np.array([0, 0, rod_length + .05]), load_mass)

    while not simulator.glfw_window_should_close():
        simulator.update()

        force, torque = airflow_sampl.generate_forces_opt(payload0)
        payload0.set_force_torque(force, torque)
        
        ctrl0_max = is_greater_than(drone0.ctrl0[0], ctrl0_max)
        ctrl1_max = is_greater_than(drone0.ctrl1[0], ctrl1_max)
        ctrl2_max = is_greater_than(drone0.ctrl2[0], ctrl2_max)
        ctrl3_max = is_greater_than(drone0.ctrl3[0], ctrl3_max)

    simulator.close()

    print(np.max(ctrl0_max))
    print(np.max(ctrl1_max))
    print(np.max(ctrl2_max))
    print(np.max(ctrl3_max))

    plot_payload_and_airflow_volume(payload0, airflow_sampl)