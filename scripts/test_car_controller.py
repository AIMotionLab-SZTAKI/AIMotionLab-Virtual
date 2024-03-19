import os

from aiml_virtual.simulator import ActiveSimulator

from aiml_virtual.xml_generator import SceneXmlGenerator

from aiml_virtual.trajectory import CarTrajectory
from aiml_virtual.controller import CarLPVController
from aiml_virtual.util import mujoco_helper, carHeading2quaternion

import numpy as np
import matplotlib.pyplot as plt

from aiml_virtual.object import parseMovingObjects

# setting up matplotlib for Qt5 on macos
if os.name == "posix":
    import matplotlib
    matplotlib.use("Qt5Agg")

RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xml_base_filename = "car_obstackle_scene.xml"
save_filename = "built_scene.xml"

# create xml with a car
scene = SceneXmlGenerator(xml_base_filename)
car0_name = scene.add_car(pos="0 0 0.052", quat=carHeading2quaternion(0.64424), color=RED_COLOR, is_virtual=True, has_rod=False)
 

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers 
virt_parsers = [parseMovingObjects]



control_step, graphics_step = 0.025, 0.025 # the car controller operates in 40 Hz
xml_filename = os.path.join(xml_path, save_filename)

# recording interval for automatic video capture
rec_interval=[1,25]
#rec_interval = None # no video capture

# initializing simulator
simulator = ActiveSimulator(xml_filename, rec_interval, control_step, graphics_step)

# ONLY for recording
#simulator.activeCam
#simulator.activeCam.distance=9
#simulator.activeCam.azimuth=230

# grabbing the drone and the car
car0 = simulator.get_MovingObject_by_name_in_xml(car0_name)


# create a trajectory
car0_trajectory=CarTrajectory()

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

car0_trajectory.build_from_points_smooth_const_speed(path_points=path_points, path_smoothing=0.01, path_degree=4, virtual_speed=1)
car0_trajectory.plot_trajectory(block=False)



car0_controller = CarLPVController(car0.mass, car0.inertia) # kiszedtem a gravitációt, az a kocsit nem érdekli

car0_controllers = [car0_controller]

def update_controller_type(state, setpoint, time, i):
    # ha csak 1 controller van ez akkor is kell?
    return 0



# setting update_controller_type function, trajectory and controller for car0
car0.set_update_controller_type_function(update_controller_type)
car0.set_trajectory(car0_trajectory)
car0.set_controllers(car0_controllers)


# start simulation
x=[]
y=[]
while not simulator.glfw_window_should_close():
    simulator.update()
    st=car0.get_state()
    x.append(st["pos_x"])
    y.append(st["pos_y"])
    
simulator.close()

plt.figure()
plt.plot(x,y)
plt.axis('equal')
plt.show()