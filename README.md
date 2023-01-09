# 3D model of the test environment in the building

## Purpose
Active simulator tool to simulate drones in the virtual 3D model of Sztaki 6th floor. The model currently supports the following objects:
  * Drones
    * crazyflie (simulated and mocap)
    * bumblebee (simulated; hooked simulated; mocap; (hooked mocap in the future))
  * Hospital
  * Post office
  * Sztaki landing zone
  * Poles
  * Landing zones
  * Airport
  * Parking lot

## Installation
1. Create and activate a virtual environment

2. Prerequisite for motioncapture (haven't been able to get it to work on windows):
```
$ sudo apt install libboost-system-dev libboost-thread-dev libeigen3-dev ninja-build
```
3.
```
$ pip install -e .
```
4. Clone https://github.com/AIMotionLab-SZTAKI/crazyflie-mujoco in some folder
Add the crazyflie-mujoco folder to path in classes/trajectory.py and scripts/test_active_simulator like so:

```
sys.path.insert(2, '/home/crazyfly/Desktop/mujoco_digital_twin/crazyflie-mujoco/')

```
5. On Windows
```
$ pip install windows-curses
```

6.

```
$ cd scripts
```

7.
```
$ python3.8 test_active_simulator.py
```
or

```
$ python3.8 build_scene.py
```
or

```
$ python3.8 load_and_display_scene.py
```

## Usage build_scene.py

For drone naming convention see naming_convention_in_xml.txt

To add a building:
  * Press 'b' (as in building)
  * In the pop-up window select the bulding with the dropdown list
  * Specify the position and the orientation (as quaternion). Default quaternion is 1 0 0 0.
  * Click ok, or hit enter

To add drones:
  * Press 'd' (as in drone)
  * Select drone type in the drop down menu
  * Set the position
  * Set the quaternion
  * Click ok, or press enter

To name drones:
  * Press 'n' (short for name)
  * In the pop-up window enter the name of the drones that are 'ticked' in Motive
  * Click ok, or hit enter

To connect to Motive:
  * Press 'c' (short for connect)

To automatically build a scene based on Motive stream:
  * Press 'o' (short for optitrack)

To start and stop video recording:
  * Press 'r' (short for record)
  * The location of the saved video will be printed in the terminal

To move the camera around, use mouse buttons and wheel.

## Usage load_and_display_scene.py

...
