import matplotlib.pyplot as plt
import numpy as np
import math
from aiml_virtual.util import mujoco_helper
from stl import mesh
import os


def is_point_inside_lobe(point, center, a, exponent):

    d = math.sqrt((center[0] - point[0])**2 + (center[1] - point[1])**2)

    if d <= 2 * a:

        z_lim = a * math.sin(math.acos((a - d) / a)) * math.sin(math.acos((a - d) / a) / 2.0)**exponent

        if point[2] < center[2] + z_lim and point[2] > (center[2] - z_lim):
            return d, True
    
    return d, False


def test_collision_func(point, a=5., exp=1.3, center=np.array((0.0, 0.0, 0.0))):

    xs = np.linspace(0., 2 * a, 1000)

    zps = (a * np.sin(np.arccos((-xs + a) / a)) * (np.sin(np.arccos((-xs + a) / a) / 2.))**exp)
    zns = -zps.copy()

    zps += center[2]
    zns += center[2]

    xs += center[0]

    d, check = is_point_inside_lobe(point, center, a, exp)
    print(check)

    plt.plot(xs, zps)
    plt.plot(xs, zns)
    plt.plot([d + center[0]], [point[2]], marker="x", markersize=5, markeredgecolor="red", markerfacecolor="green")
    plt.show()



mujoco_helper.create_radar_field_stl(20.0, 1.3, 45, 50)

#test_collision_func(np.array((1., 14.75, 7.)), center=np.array((3., 5., 7.)))