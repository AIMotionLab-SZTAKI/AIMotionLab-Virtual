# Virtual 3D model of the test environment and simulation with MuJoCo

The documentation can be accessed [here](https://github.com/AIMotionLab-SZTAKI/Mujoco-Simulator/wiki).

## Installation
1. Create and activate a virtual environment

2. Install libraries with pip
```
$ pip install -e .
```
3. On Windows additionally
```
$ pip install windows-curses
```
4. Navigate to scripts

```
$ cd scripts
```

5. Run one of the scripts like below
```
$ python3.8 simulate_dummies.py
or
$ python3.8 build_scene.py
or
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

To add payloads:
  * Press 't' (as in teher in Hungarian)
  * A pop-up window appears
  * Enter its color, mass, size, position and orientation
  * Clik ok, or hit enter

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
  > Note: In order to create video recording on a windows platform, the ffmpeg binaries also need to be installed! The latest build can be accessed [here](https://www.gyan.dev/ffmpeg/builds/). After, the zip file has been successfully downloaded extract it to a convinient location (C drive is recommended) and set the environment path variable for 'ffmpeg' by running
  	
```
  $ setx /m PATH "<your-path-to-the-unzipped-folder>/ffmpeg\bin;%PATH%"
```

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

## Usage of test_active_simulator.py

This script demonstrates how to use the ActiveSimulator class. It's pretty well commented in the file.

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
