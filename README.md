# Virtual 3D model of AIMotion-Lab and simulation with MuJoCo engine

## Installation
1.  Clone this repository.
```
$ git clone https://github.com/AIMotionLab-SZTAKI/AIMotionLab-Virtual.git
```
2.  Create and activate a virtual environment.
```
$ cd AIMotionLab-Virtual
$ python3 -m venv venv
$ source venv/bin/activate
```
3.  Install dependencies with pip. Note that currently compatibility between the 
    package's dependencies as well as different python versions is not enforced. For a 
    guaranteed working configuration, check out the comments in pyproject.toml
```
$ pip install -e .
```
4. Generate docs, they will be generated in docs/build/html. Open index.html in any web
    browser to start reading. (Note: The legacy version's documentation can be accessed 
    [here](https://github.com/AIMotionLab-SZTAKI/Mujoco-Simulator/wiki).)
```
cd docs
make clean
make html
```

## Usage
To get a broad look at how you may use the aiml_virtual package, check out 
the examples folder, where you will find a series of tutorial scripts. These
scripts are designed to be as simple as possible, while showcasing the
functionality of the package. They are annotated by comments explaining each
line of code. Using them as a template, you can start writing your own 
scripts using the aiml_virtual package. If you want to extend the classes
of the package, or want more in-depth knowledge, check out the docs 
which you generated during installation. 

### To run the scripts 
1. Navigate to the correct folder.
```
$ cd scripts/examples
```
2. Run one of the scripts like below. 
```
$ python3 01_load_scene.py 
```