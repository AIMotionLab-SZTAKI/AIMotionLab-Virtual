import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum


class MovingObject:
    """ Base class for any moving vehicle or object
    """

    def update(self, i):
        # must implement this method
        raise NotImplementedError("Derived class must implement evaluate()")