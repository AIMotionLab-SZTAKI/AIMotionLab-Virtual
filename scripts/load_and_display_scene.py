import os
from aiml_virtual.simulator import ActiveSimulator
import tkinter
from tkinter import filedialog

from aiml_virtual.object.object_parser import parseMovingObjects, parseMocapObjects


# open the base
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene_base.xml"

# create list of parsers
virt_parsers = [parseMovingObjects]
mocap_parsers = [parseMocapObjects]
simulator = ActiveSimulator(os.path.join(xml_path, xmlBaseFileName), None, 0.01, 0.02, virt_parsers, mocap_parsers, False)

simulator.set_title("AIMotionLab-Virtual")

def load_model():
    tkinter.Tk().withdraw()
    filetypes = (('XML files', '*.xml'),('all files', '*.*'))
    filename = filedialog.askopenfilename(title="Open XML", initialdir=xml_path, filetypes=filetypes)
    if filename:
        simulator.reload_model(filename)

    tkinter.Tk().destroy()

def main():
    simulator.set_key_l_callback(load_model)

    simulator.cam.azimuth = 0

    load_model()
    
    while not simulator.glfw_window_should_close():

        simulator.update()
    
    simulator.close()


if __name__ == '__main__':
    main()
