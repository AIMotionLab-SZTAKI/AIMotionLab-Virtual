import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.integrate import quad


def cosine_arc_length(amplitude, frequency, start, end):
    # Define the derivative of the cosine function
    def derivative_cos(x):
        return -amplitude * frequency * np.sin(frequency * x)

    # Define the integrand
    def integrand(x):
        return np.sqrt(1 + derivative_cos(x) ** 2)

    # Integrate the integrand function using scipy's quad function
    arc_length, _ = quad(integrand, start, end)

    return arc_length


def paperclip():
    focus_x = [0, 0]
    focus_y = [-0.7, 0.7]
    r = 0.8
    len_straight = focus_y[1] - focus_y[0]
    len_turn = r * np.pi
    ppm = 6
    num_straight = int(len_straight * ppm)
    num_turn = int(len_turn * ppm)
    x = np.hstack((np.linspace(focus_x[0] + r, focus_x[1] + r, num_straight),
                   focus_x[1] + r * np.cos(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_x[1] - r, focus_x[0] - r, num_straight),
                   focus_x[0] + r * np.cos(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    y = np.hstack((np.linspace(focus_y[0], focus_y[1], num_straight),
                   focus_y[1] + r * np.sin(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_y[1], focus_y[0], num_straight),
                   focus_y[0] + r * np.sin(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    points = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    delete_idx = []
    for i, point in enumerate(points):
        if i > 0:
            if np.linalg.norm(point - points[i-1, :]) < 0.1:
                delete_idx += [i]
    points = np.delete(points, delete_idx, 0)
    return points


def dented_paperclip():
    focus_x = [0, 0]
    focus_y = [-1, 1]
    r = 0.8
    len_straight = focus_y[1] - focus_y[0]
    len_turn = r * np.pi
    r_dent = 0.2
    len_dent = cosine_arc_length(r_dent, 2*np.pi/len_straight, 0, len_straight)
    ppm = 4
    num_straight = int(len_straight * ppm)
    num_turn = int(len_turn * ppm)
    num_dent = int(len_dent * ppm)
    x = np.hstack((np.linspace(focus_x[1] + r, focus_x[0] + r, num_straight),
                   focus_x[1] + r * np.cos(np.linspace(0, np.pi, num_turn)),
                   -r + r_dent - r_dent * np.cos(np.linspace(0, 2*np.pi, num_dent)),
                   focus_x[0] + r * np.cos(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    y = np.hstack((np.linspace(focus_y[0], focus_y[1], num_dent),
                   focus_y[1] + r * np.sin(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_y[1], focus_y[0], num_straight),
                   focus_y[0] + r * np.sin(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    points = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    delete_idx = []
    for i, point in enumerate(points):
        if i > 0:
            if np.linalg.norm(point - points[i-1, :]) < 0.1:
                delete_idx += [i]
    points = np.delete(points, delete_idx, 0)
    return points

def lissajous():
    r_x = 0.6
    r_y = 1.9
    num_points = 30
    t = np.linspace(0, 2*np.pi, num_points)
    x = r_x * np.sin(2 * t - np.pi)
    y = r_y * np.sin(t)
    points = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    delete_idx = []
    for i, point in enumerate(points):
        if i > 0:
            if np.linalg.norm(point - points[i-1, :]) < 0.1:
                delete_idx += [i]
    points = np.delete(points, delete_idx, 0)
    return points

if __name__ == "__main__":
    points = paperclip()
    # points = dented_paperclip()
    # points = lissajous()
    plt.figure()
    plt.plot(-points[:, 1], points[:, 0], '*')
    plt.axis("equal")
    plt.show()
