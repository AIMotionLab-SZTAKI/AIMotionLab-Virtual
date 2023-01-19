from dataclasses import dataclass
import numpy as np
import os
from classes.passive_display import PassiveDisplay
import tkinter
from tkinter import filedialog


# open the base on which we'll build
xml_path = "../xml_models"
xmlBaseFileName = "scene.xml"

display = PassiveDisplay(os.path.join(xml_path, xmlBaseFileName), 0.02, False)


def load_model():
    tkinter.Tk().withdraw()
    filetypes = (('XML files', '*.xml'),('all files', '*.*'))
    filename = filedialog.askopenfilename(title="Open XML", initialdir=xml_path, filetypes=filetypes)
    if filename:
        display.reload_model(filename)

    tkinter.Tk().destroy()

def main():
    display.set_key_l_callback(load_model)
    
    display.run()


if __name__ == '__main__':
    main()
