# AIMotionLab-Virtual

Python package for simulating the vehicles of AIMotionLab (SZTAKI) using the [MuJoCo](https://mujoco.org/) physics engine. 

## Installation

### Which Python version

**Use Python 3.11.**

Most Python packages are distributed as *wheels*: prebuilt archives that pip only has to unpack. Packages containing compiled C/C++ code must publish a separate wheel for every Python version, because compiled code is tied to the interpreter it was built against. When no matching wheel exists, pip falls back to downloading the source code and compiling it on your machine, which requires a full build toolchain and frequently fails for older packages on newer interpreters.

The OptiTrack connection relies on `motioncapture`, pinned to version `1.0a2`, which publishes wheels only up to CPython 3.11. On Python 3.12 or newer its installation falls back to compiling from source and fails. At the other end, recent releases of `mujoco` require at least Python 3.10. Python 3.11 is the version the lab uses, and the one the installation below is known to work with. If a newer Python version is desired, the `motioncapture` dependency must be pinned to a newer version, and the parts of the code relying on `motioncapture` adjusted to use the newer API.

### Getting Python 3.11 with Miniconda

Recent Linux distributions often ship only a newer Python than 3.11. Installing an older one system-wide is possible, but invasive, and risks disturbing other software on the machine that relies on the system Python. If you don't have a Python 3.11 installation, conda solves this cleanly:

- **conda** is a package and environment manager. Unlike a plain virtual environment, a conda environment can contain *its own Python interpreter*, of whatever version you ask for.
- **Anaconda** is a large distribution bundling conda with hundreds of preinstalled scientific packages.
- **Miniconda** is a minimal installer containing just conda, Python and a few essentials. This is all we need.

Install Miniconda by following the [official instructions](https://docs.anaconda.com/miniconda/), then create an environment containing Python 3.11. This only has to be done once per machine, and the same environment can serve several projects:

```bash
conda create -n py311 python=3.11
```

`py311` is just a name; choose whatever you like, and use it consistently below.

We only use this conda environment as a *source of a Python 3.11 interpreter*. The project's packages are installed into a separate virtual environment inside the repository, which keeps them next to the code, out of the shared conda environment, and easy to delete and recreate.

### Installing

**1. Clone the repository:**

```bash
git clone https://github.com/AIMotionLab-SZTAKI/AIMotionLab-Virtual.git
cd AIMotionLab-Virtual
```

**2/a. If required, activate the conda environment to get the correct Python version, which the venv will inherit:**

```bash
conda activate py311
```

**2/b. Create and activate the virtual environent**

```bash
python -m venv venv
source venv/bin/activate
```

`python -m venv venv` uses whichever interpreter `python` currently refers to, which after `conda activate` is conda's Python 3.11. The new environment lives in the `venv/` folder (ignored by git), starts out with no packages, and records which interpreter it was made from in `venv/pyvenv.cfg`. Because it links back to that interpreter, deleting or recreating the conda environment breaks the venv; if that happens, delete `venv/` and repeat this step. On Windows, the activation command is `venv\Scripts\activate` instead of `source venv/bin/activate`.

Check that the right interpreter is active:

```bash
python --version   # should print Python 3.11.x
```

**3. Upgrade pip:**

```bash
pip install --upgrade pip
```

This step is required. A fresh venv is seeded with the copy of pip that was bundled with the interpreter, which may be several releases behind, and older pip versions are a common cause of failures with editable installs and with dependencies fetched from git.

**4. Install the package and whatever optional dependency packs you desire:**

```bash
pip install -e ".[docs,skyc,mocap]"
```

- `.` means "the project in the current directory", as described by its `pyproject.toml`.
- `-e` (editable) installs the package as a link to this folder rather than a copy: edit the source, and the change takes effect immediately, with no reinstall.
- `[docs,skyc,mocap]` selects *optional dependency groups* on top of the core dependencies. The quotes stop shells such as zsh from interpreting the square brackets.

In later sessions, activating the venv is all you need:

```bash
source venv/bin/activate
```

To leave it again, run `deactivate`. As a sanity check, see if the package can be imported:

```bash
python -c "import aiml_virtual.scene, aiml_virtual.simulator"
```

No output means success.

### Dependencies

Dependencies are declared in `pyproject.toml`. The version numbers in the comments next to them are versions known to work together, meant as a reference rather than a hard requirement. Core dependencies are always installed, Optional groups are declared under `[project.optional-dependencies]`, and selected in square brackets at install time, separated by commas. The core dependencies are meant to be up-to-date and compatible with each other as well as the intended python version; optional dependency groups are not guaranteed to be compatible with each other.

## Generating the documentation

The package is documented through docstrings, from which [Sphinx](https://www.sphinx-doc.org/) generates an HTML API reference. That reference describes every module, class and function in detail; this readme is meant as a guide to where to start.

With the venv active, and the `docs` group installed:

```bash
cd docs
make clean
make html
```

Then open `docs/build/html/index.html` in a browser. The same commands work on Windows, where `make` is provided by `docs/make.bat`. The pages mirror the structure of the package. Rather than being written by hand, the `.rst` source file for each page is generated by `docs/source/conf.py` at the start of every build, into `docs/source/aiml_virtual/`. That folder is ignored by git and overwritten on each build, so there is no point editing it. Each page links to the highlighted source code of its module, which is often the quickest way to jump from the reference to the implementation.

## How it works

### Scene and Simulator

A `Scene` keeps three representations of the simulated world consistent: the MJCF XML tree, the `mujoco.MjModel` compiled from it, and the list of Python objects in it. Every object knows how to describe itself in MJCF (its `create_xml_element` method), so adding an object means merging its XML elements into the tree, saving the tree to a file, and recompiling the model from that file. The saved file includes the base scene it was built on.

A `Simulator` takes a scene, and runs it. It owns the `mujoco.MjData`, which holds the state of the simulation. Python objects are created independently of both, meaning that you can create Python objects without adding them to the scene, or operate on them before adding them to the scene. Whenever they *are* added to the scene, the model is recompiled and they are bound to it (`bind_to_model`); they are bound to the data when the simulation is launched (`bind_to_data`).

### Simulated objects

Objects that are involved in the simulation, and have a Python representation share the base class `SimulatedObject`:

```
SimulatedObject
├── DynamicObject                  pose calculated by MuJoCo physics
│   ├── BoxPayload, TeardropPayload
│   └── ControlledObject           has actuators, a controller and a trajectory
│       ├── Bicycle
│       ├── Car
│       └── Drone
│           ├── Crazyflie
│           └── Bumblebee
│               └── HookedBumblebee1DOF, HookedBumblebee2DOF
└── MocapObject                    pose read from a mocap source
    ├── MocapPayload, MocapHook, MocapCar, MocapTrailer, buildings, obstacles...
    ├── MocapDrone
    │   └── MocapCrazyflie, MocapBumblebee
    └── MocapSkeleton              several rigid mocap bodies acting as one object
        └── MocapHookedBumblebee2DOF, MocapHitchedCar
```

When a class descending from `SimulatedObject` is defined, it registers itself under an identifier, its class name by default. Objects are named after their class identifier and a counter (`Crazyflie_0`, `Crazyflie_1`), and when a scene is loaded from a file, top-level bodies with such names get a Python object of the registered class. Classes in `aiml_virtual.simulated_object` are always registered, but any other class (including one defined in your own script) is only registered once it is imported, so it must be imported before loading a file that contains it. Identifiers must be unique: if two classes share a name, one of them has to override `get_identifier`. Virtual classes which shall not be instantiated can override `get_identifier` to return `None`.

To add a new kind of object, subclass the appropriate class and implement `create_xml_element` (its MJCF description), `bind_to_data` (saving references to its sensors and actuators in the data) and `update` (what it does each time it gets its turn).

### The simulation loop

The simulator does not run a fixed loop; it keeps a dictionary of `Processes`, each a function with a target frequency, and every `tick()` calls the processes that are due. One tick is one physics step, whose length is the `timestep` option in the scene's XML. By default, the processes are:

- the physics step, every tick,
- the `update` of each object, at the object's `update_frequency` (500 Hz by default, 40 Hz for cars, and the mocap source's frame rate for mocap objects),
- scheduled events, every tick,
- when there is a display: rendering it, and waiting to keep the simulation in sync with wall clock time (scaled by `speed`), at the display's frame rate. Without a display, nothing slows the simulation down.

A process can only run every N-th tick, so its frequency is effectively rounded down to fit the timestep. Further processes can be registered with `add_process`, and one-off functions can be scheduled for a given simulation time with `add_event`.

### Controllers and trajectories

In a controlled object's `update`, its trajectory is evaluated at the current time, giving a setpoint, and its controller turns the setpoint and the object's sensor readings into actuator inputs. For a drone, the setpoint is a dictionary of target position, velocity, orientation and angular velocity; the controller outputs a thrust and three torques, which the drone's input matrix distributes between the four propellers. Both `Controller` and `Trajectory` are abstract base classes, so new ones only have to implement `compute_control` and `evaluate` respectively. If no controller is set, each controlled object creates its default one.

### Mocap sources

A `MocapSource` produces frames in a background thread: dictionaries mapping rigid body names to a position and a quaternion (in MuJoCo's w-x-y-z order). Frames are read through its `data` property, which returns a copy, guarded by a lock. Each mocap object looks up its own name in the latest frame when it updates, applies its offset (which should be calibrated in case the marker set's center doesn't coincide with the model's origin), and writes the pose into MuJoCo.

Which class belongs to which rigid body name, and which rigid bodies make up each mocap skeleton, is set in `aiml_virtual/resources/mocap_config.json`. A new rigid body tracked in Motive has to be added there, before `scene.add_mocap_objects` can recognize it.

### Resources

`aiml_virtual/resources` holds the files the package relies on: base scenes and example scenes with their meshes and textures (`xml_models`), sample `.skyc` files (`skyc`), airflow lookup tables (`airflow_data`), and the mocap configuration. They are installed along with the package, and their paths are available as `aiml_virtual.xml_directory`, `aiml_virtual.skyc_folder`, etc., so scripts do not need to know where the package is installed.

## Examples

The scripts in `scripts/examples` are a tutorial of the package's functionality, commented line by line. Each one is short, and meant to be used as a template for your own scripts. Run them from their own folder:

```bash
cd scripts/examples
python 01_load_scene.py
```

Some examples save files (scene XMLs, videos) into the current directory, and `.gitignore` only covers them in `scripts/examples`. Most examples run until their window is closed; on a machine without a display, no window opens, so they run until stopped with Ctrl+C.

### 01: Loading a scene

A `Scene` is read from an MJCF file, and handed to a `Simulator`. The scene corresponds to the MuJoCo *model* (the initial setup), the simulator to the MuJoCo *data* (a run of the simulation through time). Objects in the file whose names match a class of the package (such as `Bicycle_0`) are recognized, and get a Python object of that class. `sim.launch()` opens the window, and each `sim.tick()` advances the simulation.

In the window, drag with the left mouse button to rotate the camera, drag with the right mouse button to move the point it looks at, scroll to zoom, and use W, A, S, D to move horizontally. Space pauses and resumes the physics, and R starts and stops recording a video, saved as `simulator.mp4` in the current directory when the simulation ends.

### 02: Building a scene

Scenes are usually built in a script, on top of a base scene from `aiml_virtual/resources/xml_models`; here, `scene_base.xml`, the model of the lab. Objects are added with `scene.add_object`, which also saves the resulting scene to the file given as `save_filename`. The two payloads added show the two kinds of objects: a `BoxPayload` is a *dynamic* object, subject to physics, so it falls, while a `MocapPayload` is a *mocap* object, whose pose comes from a motion capture source instead, so without one it stays in place. The simulation is launched at a lower frame rate and a fifth of real time speed.

### 03: Dynamic objects

*Requires the `skyc` group.*

Controlled objects are dynamic objects with actuators, driven by a controller that follows a trajectory. A `Bicycle` drives with a constant motor torque. A `HookedBumblebee1DOF` holds its position using a `DummyDroneTrajectory`, while a `TeardropPayload` dropped on it disturbs it. A `Crazyflie` flies a trajectory read from a `.skyc` file, starting 3 seconds into the simulation. Drones use a geometric controller by default.

### 04: Mocap objects

A `DummyMocapSource` imitates a motion capture system: it runs in its own thread, and calls a frame generator function to produce frames, which map object names to poses. Here, the objects move in circles. `scene.add_mocap_objects` adds an object for every name in the frame, choosing its class based on `aiml_virtual/resources/mocap_config.json` (for example, `cf0` becomes a `MocapCrazyflie`). Objects can be removed by name or by reference, and a mocap object can also be created by hand, given its source and the name to look for in the frames.

### 05: Recording

`DummyMocapSource.freeze` takes a snapshot of another mocap source, and keeps providing that single frame; useful for saving the positions of static objects, such as the model buildings and obstacles in the lab. Setting `sim.visualizer.recording` to `True` records the simulation into `simulator.mp4`. Close the simulation or press R to end the recording, then you can view the resulting video.

### 06: Rendering without a display

Launching with `with_display=False` runs the simulation without opening a window, and without rendering anything: only the physics and the objects' updates run, and the simulation is not slowed down to wall clock time. This is useful for generating simulation data as fast as possible. Videos can still be recorded without a display: `sim.visualizer.toggle_record()` starts the recording, and renders only the frames that go into the video. This example records 3000 physics steps (3 seconds of simulated time), then stops by itself.

### 07: Cars

A `Car` follows a trajectory using its default LPV controller. The first car follows a `CarTrajectory`, which is parametrized by time; the second one tows a trailer, and follows a `CarTrajectorySpatial`, which is parametrized by the distance travelled along the path.

### 08: OptiTrack

*Requires the `mocap` group, and the lab's OptiTrack system streaming to the computer.*

`OptitrackMocapSource` connects to OptiTrack on construction. The scene file already contains three model buildings; objects read from a file have no mocap source, so each one is assigned the source and its rigid body name with `assign_mocap`. The buildings must be tracked in Motive for them to move.

### 09: Mocap skeletons

*Requires the `mocap` group, and the lab's OptiTrack system streaming to the computer.*

MuJoCo mocap bodies cannot have joints, so an object such as a drone with a hook hanging under it is represented by a `MocapSkeleton`: one object for the simulator, made up of several rigid mocap bodies in MuJoCo. `MocapHookedBumblebee2DOF` combines a `MocapBumblebee` and a `MocapHook`, and places the hook at the bottom of the drone, in the orientation read from OptiTrack. Which rigid bodies make up which skeleton is set in `mocap_config.json`.

### 10: Airflow

*Requires the `airflow` group.*

Two hooked Bumblebees hover above two payloads, and the airflow of their propellers pushes the payloads. Payloads implement the `AirflowTarget` interface, which splits their surface into small rectangles, and an airflow sampler added to them calculates the forces from lookup tables of air pressure (and velocity) under a drone, found in `aiml_virtual/resources/airflow_data`. A `SimpleAirflowSampler` uses the pressure at a single rotor speed, while a `ComplexAirflowSampler` interpolates both pressure and velocity at the drone's current rotor speed. The calculation is expensive, so this simulation may run slower than real time.

### 11: Viewing a skyc file

*Requires the `skyc` group.*

`SkycViewer` previews a `.skyc` drone show before it is flown. `plot()` plots the trajectories. `play_raw()` replays them exactly, using mocap drones, while `play_with_controller()` flies them with simulated Crazyflies and their controllers. Both show the drones' light colors, and warn about drones getting closer than 0.2 m to each other. Each playback runs until its window is closed, after which the next one starts.

