"""
This script shows how you can render a simulation without a display
"""

import os
import sys
import pathlib

# make sure imports work by adding the necessary folders to the path:
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())
xml_directory = os.path.join(project_root.resolve().as_posix(), "xml_models")
project_root = project_root.resolve().as_posix()

from aiml_virtual import scene, simulator

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "example_scene_2.xml"), save_filename="example_scene_2.xml")
    # As noted in 5_record.py, the render fps is different from the display fps. This is due to the fact that these
    # two processes are completely separate: You can have a display without rendering anything, like we did in the
    # first four examples. You can also render without displaying anything. In fact, you can run a simulation with
    # any number of processes disabled (no display, no rendering), to gather data for diagrams for example.
    # For now, let's render a video from the second example's scene without actually displaying anything.
    sim = simulator.Simulator(scn)
    with sim.launch(with_display=False, renderer_fps=144):  # the with_display argument is True by default
        sim.processes["render"].toggle()  # let's turn the recording on
        while sim.tick_count < 3000:  # let's step the physics engine 3 thousand times!
            sim.tick()




