import os
import time
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.trajectory import CarTrajectory
from aiml_virtual.controller import CarLPVController
from aiml_virtual.util import mujoco_helper, carHeading2quaternion
from aiml_virtual.object.payload import PAYLOAD_TYPES, TeardropPayload, BoxPayload
from aiml_virtual.object.car import Fleet1Tenth
import numpy as np
import matplotlib.pyplot as plt
from aiml_virtual.object import parseMovingObjects
from scipy.spatial.transform import Rotation
from aiml_virtual.trajectory.car_path_point_generator import paperclip
from typing import Literal


RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"
BLACK = "0.1 0.1 0.1 1.0"


def get_car_trajectory():
    # create a trajectory
    car_trajectory = CarTrajectory()
    # define path points and build the path
    # path_points = np.vstack((paperclip(), paperclip()[1:, :], paperclip()[1:, :]))
    path_points = paperclip()
    car_trajectory.build_from_points_smooth_const_speed(path_points=path_points, path_smoothing=1e-4, path_degree=5,
                                                        virtual_speed=0.6)
    return car_trajectory


class TrailerPredictor:
    def __init__(self, car_trajectory: CarTrajectory, 
                 payload_type: Literal[PAYLOAD_TYPES.Box, PAYLOAD_TYPES.Teardrop] = PAYLOAD_TYPES.Teardrop):
        # Initialize simulation
        abs_path = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(abs_path, "..", "..", "xml_models")
        xml_base_filename = "scene_base.xml"
        save_filename = "built_scene_for_predictor.xml"

        car_pos = np.array([car_trajectory.pos_tck[1][0][0], car_trajectory.pos_tck[1][1][0], 0.052])
        heading_smoothing_index = 5
        car_heading = np.arctan2(car_trajectory.pos_tck[1][1][heading_smoothing_index] - car_trajectory.pos_tck[1][1][0],
                                 car_trajectory.pos_tck[1][0][heading_smoothing_index] - car_trajectory.pos_tck[1][0][0])
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
        payload_name = scene.add_payload(pos=np.array2string(payload_pos)[1:-1], size="0.05 0.05 0.05", mass="0.01",
                                         quat=payload_quat, color=BLACK, type=payload_type)

        # saving the scene as xml so that the simulator can load it
        scene.save_xml(os.path.join(xml_path, save_filename))

        # create list of parsers
        virt_parsers = [parseMovingObjects]

        control_step, graphics_step = 0.01, 0.02
        xml_filename = os.path.join(xml_path, save_filename)

        # initializing simulator
        self.simulator = ActiveSimulator(xml_filename, None, control_step, graphics_step, with_graphics=False)
        self.simulator.model.opt.timestep = 0.005
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

        self.trailer_top_plate_height = 0.119  # for box payload: 0.119, for teardrop payload: 0.123
        self.rod_pitch = -0.1  # these should be constant in normal operation
        self.rod_to_front.qpos[:] = -self.rod_pitch
        rod_yaw = 0
        self.init_state = np.hstack((car_pos, np.fromstring(car_quat, sep=" "),
                                     np.zeros(6), rod_yaw, 0, 0, 0,
                                     np.nan * payload_pos, np.nan * np.fromstring(payload_quat, sep=" ")))

    def simulate(self, init_state, cur_time, prediction_time, predicted_obj='payload'):
        # reset simulation
        self.simulator.reset_data()
        self.simulator.goto(cur_time)

        # set initial states
        # state: car position, orientation, velocity, angular velocity; car_to_rod orientation, ang_vel;
        # front_to_rear orientation, ang_vel; payload position, orientation  --  ndim = 24
        self.car.joint.qpos = init_state[0:7]
        self.car.joint.qvel = init_state[7:13]
        self.car_to_rod.qpos = np.roll(Rotation.from_euler('xyz', [0, self.rod_pitch, init_state[13]]).as_quat(), 1)
        self.car_to_rod.qvel = np.array([0, 0, init_state[14]])
        self.front_to_rear.qpos = init_state[15]
        self.front_to_rear.qvel = init_state[16]
        
        self.simulator.update()
        if np.isnan(np.sum(init_state[17:24])):  # Compute payload configuration from trailer configuration
            self.payload.qpos[0:2] = self.car.data.site(self.car.name_in_xml + "_trailer_middle").xpos[0:2]
            self.payload.qpos[2] = self.trailer_top_plate_height
            payload_rotm = np.reshape(self.car.data.site(self.car.name_in_xml + "_trailer_middle").xmat, (3, 3))
            payload_quat = Rotation.from_matrix(payload_rotm).as_quat()
            self.payload.qpos[3:7] = np.roll(payload_quat, 1)
        else:
            self.payload.data.qpos[-7:] = init_state[17:24]  # TODO
            self.payload.data.qpos[-5] = self.trailer_top_plate_height
            self.payload.data.qvel[-6:-3] = init_state[24:27]  # TODO
        # start simulation
        payload_trajectory = []  # maybe preallocate numpy array later
        #self.simulator.update()
        #self.simulator.pause()
        for _ in range(int(prediction_time / self.simulator.control_step)):
            self.simulator.update()
            # get predicted_obj state and save
            if predicted_obj == 'payload':
                pos_actual = self.payload.sensor_posimeter + np.array([0, 0, 0.18])# + Rotation.from_quat(np.roll(self.payload.sensor_orimeter, -1)).as_matrix() @\
                    #np.array([0.05, 0, 0.14]) # Compensate difference of box and teardrop payload
            elif predicted_obj == 'car':
                pos_actual = np.copy(self.car.joint.qpos[0:3])
            else:
                raise ValueError('Predicted object has to be either payload or car')
            payload_trajectory += [np.hstack((pos_actual,
                                              self.payload.sensor_velocimeter,
                                              self.payload.sensor_orimeter))]

        payload_predicted_points = np.asarray(payload_trajectory)

        def load_init_pos(t, t0):
            t_interp = self.simulator.control_step * np.arange(payload_predicted_points.shape[0])
            return [np.interp(t-t0, t_interp, dim) for dim in payload_predicted_points[:, :3].T]

        def load_init_vel(t, t0):
            t_interp = self.simulator.control_step * np.arange(payload_predicted_points.shape[0])
            return [np.interp(t-t0, t_interp, dim) for dim in payload_predicted_points[:, 3:6].T]

        def load_init_yaw(t, t0):
            t_interp = self.simulator.control_step * np.arange(payload_predicted_points.shape[0])
            payload_yaw = Rotation.from_quat(payload_predicted_points[:, [7, 8, 9, 6]]).as_euler('xyz')[:, 2]
            for i in range(1, payload_yaw.shape[0]):
                if payload_yaw[i] < payload_yaw[i-1] - np.pi:
                    payload_yaw[i:] += 2*np.pi
                elif payload_yaw[i] > payload_yaw[i-1] + np.pi:
                    payload_yaw[i:] -= 2*np.pi
            return np.interp(t-t0, t_interp, payload_yaw)
        
        return load_init_pos, load_init_vel, load_init_yaw


class DummyPredictor(TrailerPredictor):
    def __init__(self, car_trajectory: CarTrajectory):
        super().__init__(car_trajectory)
        self.prediction_time = 10
        self.payload_nominal_points = self.simulate(self.init_state, self.prediction_time)
        self.payload_disturbed_points = self.simulate(self.init_state, self.prediction_time,
                                                      np.array([0.05, 0.05, 0]))
    
    def simulate(self, init_state, prediction_time, payload_pos_error=np.zeros(3)):
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
        self.payload.qpos[0:3] += payload_pos_error
        # start simulation
        payload_trajectory = []  # maybe preallocate numpy array later
        for _ in range(int(prediction_time / self.simulator.control_step)):
            self.simulator.update()
            # get payload state and save
            payload_trajectory += [np.hstack((self.payload.sensor_posimeter,
                                              self.payload.sensor_velocimeter,
                                              self.payload.sensor_orimeter))]
        return np.asarray(payload_trajectory)
    
    def payload_pos(self, t, disturbed):
        t_interp = np.linspace(0, self.prediction_time, self.payload_nominal_points.shape[0])
        if not disturbed:
            payload_pos = self.payload_nominal_points[:, :3].T
        else:
            payload_pos = self.payload_disturbed_points[:, :3].T
        return [np.interp(t, t_interp, dim) for dim in payload_pos]


def measure_computation_time(prediction_time=10):
    predictor = TrailerPredictor(car_trajectory=get_car_trajectory())
    num_simu = 20
    start_time = time.time()
    # plt.figure()
    for _ in range(num_simu):
        payload_traj = predictor.simulate(predictor.init_state, prediction_time)
        # plt.plot(payload_traj[:, 0:2])
    comp_time_ms = (time.time()-start_time) / num_simu / prediction_time * 1000
    print(f"Car + trailer simulation takes {round(comp_time_ms, 2)} ms for each simulated second.")
    # plt.show()


def test_dummy_predictor():
    predictor = DummyPredictor(get_car_trajectory())
    t = np.linspace(0, predictor.prediction_time, 500)
    pos_nom = np.asarray(predictor.payload_pos(t, disturbed=False)).T
    pos_dist = np.asarray(predictor.payload_pos(t, disturbed=True)).T
    plt.figure()
    plt.plot(pos_nom[:, 0], pos_nom[:, 1])
    plt.plot(pos_dist[:, 0], pos_dist[:, 1])
    plt.show()


if __name__ == "__main__":
    predictor = TrailerPredictor(get_car_trajectory())
    # predictor.simulate(predictor.init_state, 25)  # Car state is required for proper trajectory evaluation
