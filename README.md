# 3D model of the test environment in the building

## Purpose
Development of a simple model generator that can be used to add buildings and objects to the 3D scene. Currently the program supports the following objects:
  * Drones
  * Hospital
  * Post office
  * Sztaki landing zone
  * Poles
  * Landing zones
  * Airport
  * Parking lot

## Installation
1. Create and activate a virtual environment

2.
```
$ pip install -r requirements.txt
```
3.
```
$ python build_scene.py
```

## Usage

To add a building:
  * Press 'b'
  * In the pop-up window select the bulding with the dropdown list.
  * Specify the position and the orientation (as quaternion). Default quaternion is 1 0 0 0.
  * Click ok, or hit enter.

To add drones:
  * Press 'd'
  * The drones will be added at hard-coded positions, because they will be updated anyway, once data streaming starts from Optitrack.
