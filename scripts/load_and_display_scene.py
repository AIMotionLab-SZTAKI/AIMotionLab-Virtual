import os
from classes.active_simulation import ActiveSimulator
import tkinter
from tkinter import filedialog

from classes.object_parser import parseMovingObjects, parseMovingMocapObjects


# open the base on which we'll build
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene.xml"

# create list of parsers
virt_parsers = [parseMovingObjects]
mocap_parsers = [parseMovingMocapObjects]
display = ActiveSimulator(os.path.join(xml_path, xmlBaseFileName), None, 0.01, 0.02, virt_parsers, mocap_parsers, False)


def load_model():
    tkinter.Tk().withdraw()
    filetypes = (('XML files', '*.xml'),('all files', '*.*'))
    filename = filedialog.askopenfilename(title="Open XML", initialdir=xml_path, filetypes=filetypes)
    if filename:
        display.reload_model(filename)

    tkinter.Tk().destroy()

def main():
    display.set_key_l_callback(load_model)
    
    i = 0
    while not display.glfw_window_should_close():

        display.update(i)
        i += 1


if __name__ == '__main__':
    main()
