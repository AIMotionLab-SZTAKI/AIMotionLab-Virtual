Quick Tutorial: Shell Scripts in `scripts/windflow`
====================================================

The `scripts/windflow` folder contains all the essential shell scripts required to use the wind simulation and visualization features. Below is a short guide for each script:

---

1. start_container.sh – Start Docker Environment
------------------------------------------------
Usage:
    source start_container.sh /ABSOLUTE/PATH/TO/app # app is the root folder of AIMotionLab

- Launches a Docker container with OpenFOAM, ParaView, and Python preinstalled.
- Required for running GUI apps (e.g., ParaView) inside Docker.

---

2. view_objects.sh – Preview XML Scene
--------------------------------------
Usage:
    source view_objects.sh /path/to/your_scene.xml

- Opens a MuJoCo-based preview of the scene defined in the XML.
- Useful for verifying your geometry before simulation.
- If no argument is passed, uses the default static scene.

---

3. run_simulation.sh – Full Workflow Execution
----------------------------------------------
Usage:
    source run_simulation.sh /path/to/your_scene.xml

- Parses the XML, converts to STL, runs OpenFOAM, and exports the result to `data.csv`.
- Can be customized using:
    export MERGE_MODE=--merge       # or --separate, --semi-merge (default)

---

4. run_parser.sh – Parser Only
------------------------------
Usage (outside Docker):
    source run_parser.sh /path/to/your_scene.xml

- Converts XML geometry into OpenFOAM STL meshes.
- Use this in hybrid setups where OpenFOAM runs inside a container, but parsing is done natively.

---

5. run_openfoam.sh – OpenFOAM Simulation
----------------------------------------
Usage (inside Docker):
    source run_openfoam.sh

- Uses the preprocessed mesh and configuration to run the simulation.
- Exports results using ParaView Python scripting into `data.csv`.

---

6. view_drone.sh – Visualize Drone in Wind Field
------------------------------------------------
Usage:
    source view_drone.sh /path/to/scene.xml /path/to/data.csv

- Simulates a drone flying through the wind field using the generated data.
- If only `data.csv` is provided, it defaults to the built-in static scene.

---

Notes
--------
- Always run scripts from inside the `scripts/windflow` directory.
- All scripts support fallback to a default `.xml` if no custom path is provided.
- Make sure your Docker container is running (if using containerized mode) before calling scripts like `run_openfoam.sh`.
