from ast import Pass
import math

#import motioncapture
import time
import numpy as np
import mujoco
import glfw
import os
import numpy as np
import time
from util import mujoco_helper
import cv2
from gui.drone_name_gui import DroneNameGui
from util.util import sync
import scipy.signal
from util.mujoco_helper import LiveLFilter
from classes.mujoco_display import Display


class ActiveSimulator(Display):

    def __init__(self, xml_file_name, connect_to_optitrack=True):
        super().__init__(xml_file_name, connect_to_optitrack)
    
    def update():
        pass
