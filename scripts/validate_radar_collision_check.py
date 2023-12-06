import matplotlib.pyplot as plt
import numpy as np
import math


def is_point_inside_lobe(point, center, a, exponent):

    d = math.sqrt((center[0] - point[0])**2 + (center[1] - point[1])**2)

    z_lim = a * math.sin(math.acos((a - d) / a)) * math.sin(math.acos((a - d) / a) / 2.0)**exponent

    if point[2] < z_lim and point[2] > (-z_lim):
        return d, True
    
    return d, False

a = 5.
exp = 2.

xs = np.linspace(0., 10., 1000)

yps = a * np.sin(np.arccos((-xs + a) / a)) * (np.sin(np.arccos((-xs + a) / a) / 2.))**exp
yns = -yps.copy()

p = np.array((-1., -5., 2.))

d, check = is_point_inside_lobe(p, np.array((0.0, 0.0, 0.0)), a, exp)
print(check)

plt.plot(xs, yps)
plt.plot(xs, yns)
plt.plot([d], [p[2]], marker="x", markersize=5, markeredgecolor="red", markerfacecolor="green")
plt.show()