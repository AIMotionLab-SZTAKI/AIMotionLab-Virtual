from classes.bezier_traj import BezierTraj
from classes.active_simulation import ActiveSimulator
from classes.object_parser import parseMovingObjects
from util.xml_generator import SceneXmlGenerator
from classes.drone import DRONE_TYPES
from classes.drone_classes.drone_geom_control import GeomControl

import os
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))


demo_path0 = os.path.join(abs_path, "..", "Demo", "drones", "drone_0")
traj0 = BezierTraj(demo_path0)
demo_path1 = os.path.join(abs_path, "..", "Demo", "drones", "drone_1")
traj1 = BezierTraj(demo_path1)
demo_path2 = os.path.join(abs_path, "..", "Demo", "drones", "drone_2")
traj2 = BezierTraj(demo_path2)

eval = traj0.evaluate_trajectory([0])
drone0_init_pos = np.array((eval[1], eval[2], eval[3]))
eval = traj1.evaluate_trajectory([0])
drone1_init_pos = np.array((eval[1], eval[2], eval[3]))
eval = traj2.evaluate_trajectory([0])
drone2_init_pos = np.array((eval[1], eval[2], eval[3]))


xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_file_name = "scene_base.xml"
save_filename = "built_scene.xml"

scene = SceneXmlGenerator(xml_base_file_name)

drone0_name = scene.add_drone(np.array2string(drone0_init_pos)[1:-1], "1 0 0 0", "0.1 0.1 0.9 1.0", DRONE_TYPES.BUMBLEBEE)
drone1_name = scene.add_drone(np.array2string(drone1_init_pos)[1:-1], "1 0 0 0", "0.1 0.1 0.9 1.0", DRONE_TYPES.BUMBLEBEE)
drone2_name = scene.add_drone(np.array2string(drone2_init_pos)[1:-1], "1 0 0 0", "0.1 0.1 0.9 1.0", DRONE_TYPES.BUMBLEBEE)

scene.save_xml(os.path.join(xml_path, save_filename))


xml_filename = os.path.join(xml_path, save_filename)
simulator = ActiveSimulator(xml_filename, None, 0.01, 0.02, [parseMovingObjects], None)

drone0 = simulator.get_MovingObject_by_name_in_xml(drone0_name)
drone1 = simulator.get_MovingObject_by_name_in_xml(drone1_name)
drone2 = simulator.get_MovingObject_by_name_in_xml(drone2_name)

controller0 = GeomControl(drone0.mass, drone0.inertia, simulator.gravity)
controller1 = GeomControl(drone1.mass, drone1.inertia, simulator.gravity)
controller2 = GeomControl(drone2.mass, drone2.inertia, simulator.gravity)

drone0.set_controllers([controller0])
drone0.set_trajectory(traj0)

drone1.set_controllers([controller1])
drone1.set_trajectory(traj1)

drone2.set_controllers([controller2])
drone2.set_trajectory(traj2)

while not simulator.glfw_window_should_close():

    simulator.update()

simulator.close()