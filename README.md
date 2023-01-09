# 3D model of the test environment in the building

## Purpose
Active simulator tool to simulate drones in the virtual 3D model of Sztaki 6th floor. The model currently supports the following objects:
  * Drones
    * crazyflie (simulated and mocap)
    * bumblebee (simulated; simulated with hook; mocap; (mocap with hook in the future))
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
4. On Windows
```
$ pip install windows-curses
```
5. Clone https://github.com/AIMotionLab-SZTAKI/crazyflie-mujoco in some folder
Add the crazyflie-mujoco folder to path in classes/trajectory.py and scripts/test_active_simulator like so:

```
sys.path.insert(2, '/home/crazyfly/Desktop/mujoco_digital_twin/crazyflie-mujoco/')

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

## Usage of build_scene.py

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
  * This will connect to Motive if not already connected and try to construct the scene based on data coming in. If Motive is not available, unfortunately the program freezes because motioncapture library does not seem to provide functionality for this possibility.
  * Building names that the program looks for in the stream:
    * hospital: bu11
    * Sztaki: bu12
    * post office: bu13
    * airport: bu14
    * poles: anything that starts with 'obs'
    * A landing zone will automatically be put under each drone
  * Drone names in the stream:
    * crazyflie: anything that starts with 'cf'
    * bumblebee: anything that starts with 'bb'

To start and stop video recording:
  * Press 'r' (short for record)
  * The location of the saved video will be printed in the terminal

To switch back and forth between drones' "on board camera" and main camera:
  * Press TAB
  * When in "on board mode" to switch amongst drones:
    * Press SPACE

To move the camera around, use mouse buttons and wheel.

## Usage of load_and_display_scene.py

To load a MuJoCo model from xml:
  * Press 'l'
  * In the pop-up window, select the xml file
  * Click ok or hit enter

To connect to Motive:
  * Press 'c' (short for connect)


To start and stop video recording:
  * Press 'r' (short for record)
  * The location of the saved video will be printed in the terminal

To switch back and forth between drones' "on board camera" and main camera:
  * Press TAB
  * When in "on board mode" to switch amongst drones:
    * Press SPACE

To move the camera around, use mouse buttons and wheel.
