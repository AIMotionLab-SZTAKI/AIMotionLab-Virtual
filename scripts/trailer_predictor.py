import os
import time
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.trajectory import CarTrajectory
from aiml_virtual.controller import CarLPVController
from aiml_virtual.util import mujoco_helper, carHeading2quaternion
from aiml_virtual.object.drone import DRONE_TYPES
from aiml_virtual.object.payload import PAYLOAD_TYPES, TeardropPayload, BoxPayload
from aiml_virtual.object.car import Fleet1Tenth
import numpy as np
import matplotlib.pyplot as plt

from aiml_virtual.object import parseMovingObjects

from scipy.spatial.transform import Rotation

RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"
BLACK = "0.1 0.1 0.1 1.0"


def get_car_trajectory():
    # create a trajectory
    car_trajectory = CarTrajectory()
    # define path points and build the path
    path_points = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 2],
            [4, 1],
            [4.5, 0],
            [4, -1],
            [3, -2],
            [2, -2],
            [1, -1],
            [0, 0],
            [-1, 1],
            [-2, 2],
            [-3, 2],
            [-4, 1],
            [-4.5, 0],
            [-4, -2.1],
            [-3, -2.3],
            [-2, -2],
            [-1, -1],
            [0, 0],
        ]
    )
    path_points /= 1.5
    car_trajectory.build_from_points_const_speed(path_points=path_points, path_smoothing=0.01, path_degree=4,
                                                 const_speed=1., start_delay=0.0)
    return car_trajectory


class TrailerPredictor:
    def __init__(self, car_trajectory: CarTrajectory):
        # Initialize simulation
        abs_path = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(abs_path, "..", "xml_models")
        xml_base_filename = "scene_base.xml"
        save_filename = "built_scene.xml"

        # TODO: get these from car_trajectory
        car_pos = np.array([0, 0, 0.052])
        car_heading = 0.64424
        car_quat = carHeading2quaternion(car_heading)
        car_rot = np.array([[np.cos(car_heading), -np.sin(car_heading), 0],
                            [np.sin(car_heading), np.cos(car_heading), 0],
                            [0, 0, 1]])
        payload_offset = np.array([-0.5, 0, 0.125])
        payload_pos = car_pos + car_rot @ payload_offset
        payload_quat = car_quat

        # create xml with a car
        scene = SceneXmlGenerator(xml_base_filename)
        car_name = scene.add_car(pos=np.array2string(car_pos)[1:-1], quat=car_quat, color=RED,
                                 is_virtual=True, has_rod=False, has_trailer=True)
        trailer_name = car_name + "_trailer"
        payload_name = scene.add_payload(pos=np.array2string(payload_pos)[1:-1], size="0.05 0.05 0.05", mass="1",
                                         quat=payload_quat, color=BLACK, type=PAYLOAD_TYPES.Box)

        # saving the scene as xml so that the simulator can load it
        scene.save_xml(os.path.join(xml_path, save_filename))

        # create list of parsers
        virt_parsers = [parseMovingObjects]

        control_step, graphics_step = 0.01, 0.02
        xml_filename = os.path.join(xml_path, save_filename)

        # initializing simulator
        self.simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, with_graphics=False)
        self.simulator.model.opt.timestep = 0.01
        self.simulator.sim_step = self.simulator.model.opt.timestep

        # grabbing the car and the payload
        self.car: Fleet1Tenth = self.simulator.get_MovingObject_by_name_in_xml(car_name)
        self.payload: TeardropPayload = self.simulator.get_MovingObject_by_name_in_xml(payload_name)
        self.car_to_rod = self.car.data.joint("car_to_rod")  # ball joint
        self.rod_to_front = self.car.data.joint("rod_to_front")  # hinge joint
        self.front_to_rear = self.car.data.joint("front_to_rear")  # hinge joint

        # car_trajectory.plot_trajectory()

        car_controller = CarLPVController(self.car.mass, self.car.inertia)

        car_controllers = [car_controller]

        # setting trajectory and controller for car
        self.car.set_trajectory(car_trajectory)
        self.car.set_controllers(car_controllers)

        self.trailer_top_plate_height = 0.123
        self.rod_pitch = -0.1  # these should be constant in normal operation
        self.rod_to_front.qpos[:] = -self.rod_pitch
        rod_yaw = np.pi / 2
        self.init_state = np.hstack((car_pos, np.fromstring(car_quat, sep=" "),
                                     np.zeros(6), rod_yaw, 0, 0, 0,
                                     np.nan * payload_pos, np.nan * np.fromstring(payload_quat, sep=" ")))

    def simulate(self, init_state, prediction_time):
        # reset simulation
        self.simulator.goto(0)
        self.simulator.reset_data()
        self.car.set_trajectory(get_car_trajectory())
        self.car.set_controllers([CarLPVController(self.car.mass, self.car.inertia)])

        # set initial states
        # state: car position, orientation, velocity, angular velocity; car_to_rod orientation, ang_vel;
        # front_to_rear orientation, ang_vel; payload position, orientation  --  ndim = 24
        self.car.joint.qpos = init_state[0:7]
        self.car.joint.qvel = init_state[7:13]
        self.car_to_rod.qpos = np.roll(Rotation.from_euler('xyz', [0, self.rod_pitch, init_state[13]]).as_quat(), 1)
        self.car_to_rod.qvel = np.array([0, 0, init_state[14]])
        self.front_to_rear.qpos = init_state[15]
        self.front_to_rear.qvel = init_state[16]

        if np.isnan(np.sum(init_state[17:24])):  # Compute payload configuration from trailer configuration
            self.simulator.update()
            self.payload.qpos[0:2] = self.car.data.site(self.car.name_in_xml + "_trailer_middle").xpos[0:2]
            self.payload.qpos[2] = self.trailer_top_plate_height
            payload_rotm = np.reshape(self.car.data.site(self.car.name_in_xml + "_trailer_middle").xmat, (3, 3))
            payload_quat = Rotation.from_matrix(payload_rotm).as_quat()
            self.payload.qpos[3:7] = np.roll(payload_quat, 1)
        else:
            self.payload.qpos = init_state[17:24]

        # start simulation
        payload_trajectory = []
        while not self.simulator.should_close(prediction_time):
            self.simulator.update()
            # get payload state and save
            payload_trajectory += [np.hstack((self.payload.sensor_posimeter,
                                              self.payload.sensor_velocimeter,
                                              self.payload.sensor_orimeter))]
        return np.asarray(payload_trajectory)


if __name__ == "__main__":
    predictor = TrailerPredictor(car_trajectory=get_car_trajectory())
    start_time = time.time()
    # plt.figure()
    for _ in range(20):
        payload_traj = predictor.simulate(predictor.init_state, 10)
        # plt.plot(payload_traj[:, 0:2])
    print((time.time()-start_time)/20)
    # plt.show()