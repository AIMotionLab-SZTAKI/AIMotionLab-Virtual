from aiml_virtual.simulated_object.mocap_object.mocap_object import MocapObject
from typing import Optional, cast, Type
import xml.etree.ElementTree as ET
import numpy as np
import casadi as cs
import scipy as cp


class Trajectory_Marker(MocapObject):
    """
    Mocap object to display a piece of paper with the airport sign on it.
    """

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "trajectory_marker"

    def __init__(self, source=None, mocap_name=None, x=[0], y=[0]):
        super().__init__(source, mocap_name)
        self.x = x
        self.y = y

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")

        object = ET.Element("body", name="marker")
        n = np.shape(self.x)[0]
        for i in range(n):
            ET.SubElement(object, "geom", type="sphere", name=f"_marker{i}", contype="0", conaffinity="0",
                          pos=f"{self.x[i]} {self.y[i]} 0.02", size=f"{0.05}")
        ret = {"worldbody": [object]}
        return ret


class Spline_2D:
    def __init__(self, points: np.array, bc_type: str = 'natural'):
        """

        :param points: Points to which the time-parametrized spline is interpolated
        :param bc_type: Type of boundary condition
        """

        self.shape = np.shape(points)
        self.original_points = points
        self.bc_type = bc_type
        self.equally_space_points = None

        self.spl_t = cp.interpolate.CubicSpline(points[:, 0], points[:, 1:], bc_type=self.bc_type)
        self.__find_arclength_par()

        self.spl_border_left = None

        self.spl_border_right = None

    def make_border_splines(self, offset=0.1):

        i = np.linspace(0, self.L, 1000)
        points, v = self.get_path_parameters(i)

        points = points

    def __find_arclength_par(self, m: int = 30, e: float = 10 ** (-3), dt: float = 10 ** (-4)):
        """Class method that finds approximation of the arc length parametrisation of the time-parameterized spline

        :param m: Number of sections
        :param e: Error margin of bisection method
        :param dt: Precision of numerical integration
        """

        # Calculation of the total arc length
        t = np.arange(min(self.original_points[:, 0]), max(self.original_points[:, 0]) + dt, dt)
        dxydt = self.spl_t.derivative(1)(t)  # calculating the derivative of the time-parametrized spline
        ds = np.sqrt(dxydt[:, 0] ** 2 + dxydt[:, 1] ** 2)  # Length of arc element
        self.L = cp.integrate.simpson(y=ds, dx=dt)  # Simpsons 3/4 rule

        # Splitting the spline into m sections with length l using bisection
        self.l = self.L / m

        # Initializing bisection method
        tstart = min(self.original_points[:, 0])
        tend = max(self.original_points[:, 0])
        tmid = (tstart + tend) / 2
        t_arr = np.array([0])
        self.s_arr = np.array([0])
        s_mid = 10000000

        # Solving problem with bisection
        for i in range(1, m):
            if i != 1:
                tstart = tmid
                tend = max(self.original_points[:, 0])
                tmid = (tstart + tend) / 2
                s_mid = 10000000

            while abs(s_mid - self.l) >= e:
                tmid_arr = np.arange(t_arr[-1], tmid + dt, dt)
                grad_mid = self.spl_t.derivative(1)(tmid_arr)
                ds_mid = np.sqrt(grad_mid[:, 0] ** 2 + grad_mid[:, 1] ** 2)
                s_mid = cp.integrate.simpson(y=ds_mid, dx=dt)

                if self.l < s_mid:
                    tend = tmid
                    tmid = (tend + tstart) / 2
                else:
                    tstart = tmid
                    tmid = (tend + tstart) / 2
            self.s_arr = np.append(self.s_arr, s_mid + i * self.l)
            t_arr = np.append(t_arr, tmid)

        self.s_arr = np.reshape(self.s_arr, (-1, 1))

        self.equally_space_points = np.concatenate((self.s_arr, self.spl_t(t_arr)),
                                                   1)  # array that contains the new points
        if (self.original_points[0, 1:] == self.original_points[-1, 1:]).all():
            self.equally_space_points = np.concatenate((self.equally_space_points, [
                [self.L + self.l, self.original_points[-1, 1], self.original_points[-1, 2]]]))
            # print(self.equally_space_points)

        self.spl_sx = cs.interpolant('n', 'bspline', [self.equally_space_points[:, 0]],
                                     self.equally_space_points[:, 1])  # fitting casadi spline to the x coordinate
        self.spl_sy = cs.interpolant('n', 'bspline', [self.equally_space_points[:, 0]],
                                     self.equally_space_points[:, 2])  # fitting casadi spline to the y coordinate
        self.spl_s = cp.interpolate.CubicSpline(self.equally_space_points[:, 0], self.equally_space_points[:, 1:],
                                                bc_type=self.bc_type)

    def get_path_parameters_ang(self, theta: cs.MX):
        """ Class method that returns the symbolic path parameters needed to calculate the lag and contouring error
            Path parameters using angles
        :param theta: path parameter (s)
        :return: x, y coordinate and the tangent angle
        """

        x = self.spl_sx(theta)
        y = self.spl_sy(theta)

        jac_x = self.spl_sx.jacobian()
        jac_y = self.spl_sy.jacobian()
        phi = cs.arctan2(jac_y(theta, theta) + 0.001, (jac_x(theta, theta) + 0.001))

        return x, y, phi

    def get_path_parameters(self, theta, theta_0=None):
        """
        Path parameters using vectors
        :param theta:
        :param theta_0:
        :return:
        """
        point = cs.hcat((self.spl_sx(theta), self.spl_sy(theta))).T

        jac_x = self.spl_sx.jacobian()
        jac_y = self.spl_sy.jacobian()

        v = cs.hcat((jac_x(theta, theta), jac_y(theta, theta)))  # unit direction vector
        # l = v**2
        # v = cs.hcat((v[:, 0]/cs.sqrt(l[:,0]+l[:,1]), v[:, 1]/cs.sqrt(l[:,0]+l[:,1])))#cs.hcat((cs.sqrt(l[:, 0]+l[: 1]), cs.sqrt(l[:, 0]+l[: 1])))
        return point, v

    def get_path_parameters_lin(self, theta, theta_0):
        """
        Path parameters using first order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        x_0 = self.spl_sx(theta_0)
        y_0 = self.spl_sy(theta_0)

        jac_x = self.spl_sx.jacobian()
        jac_x = jac_x(theta_0, theta_0)
        jac_y = self.spl_sy.jacobian()
        jac_y = jac_y(theta_0, theta_0)

        x_lin = x_0 + jac_x * (theta - theta_0)
        y_lin = y_0 + jac_y * (theta - theta_0)

        point = cs.hcat((x_lin, y_lin)).T
        v = cs.hcat((jac_x, jac_y)) / cs.sqrt(jac_x ** 2 + jac_y ** 2)

        return point, v

    def get_path_parameters_lin2(self, theta, theta_0, point_0, jac_0):
        """
        Path parameters using first order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        point = point_0 + jac_0 * cs.vcat(((theta - theta_0.T).T, (theta - theta_0.T).T))
        v = (cs.vcat((jac_0[0, :], jac_0[1, :])) / cs.vcat(
            (cs.sqrt(jac_0[0, :] ** 2 + jac_0[1, :] ** 2), cs.sqrt(jac_0[0, :] ** 2 + jac_0[1, :] ** 2)))).T

        return point, v

    def get_path_parameters_quad(self, theta, theta_0):
        """
        Path parameters using second order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        x_0 = self.spl_sx(theta_0)
        y_0 = self.spl_sy(theta_0)

        jac_x = self.spl_sx.jacobian()
        jac2_x = jac_x.jacobian()
        jac_x = jac_x(theta_0, theta_0)
        jac2_x = jac2_x(theta_0, theta_0, theta_0)[:, 1]

        jac_y = self.spl_sy.jacobian()
        jac2_y = jac_y.jacobian()
        jac_y = jac_y(theta_0, theta_0)
        jac2_y = jac2_y(theta_0, theta_0, theta_0)[:, 1]

        x_lin = x_0 + jac_x * (theta - theta_0) + jac2_x / 2 * (theta - theta_0) ** 2
        y_lin = y_0 + jac_y * (theta - theta_0) + jac2_y / 2 * (theta - theta_0) ** 2

        point = cs.hcat((x_lin, y_lin)).T

        jac_x_lin = jac_x + jac2_x * (theta - theta_0)
        jac_y_lin = jac_y + jac2_y * (theta - theta_0)

        v = cs.hcat((jac_x_lin, jac_y_lin))
        l = v ** 2
        v = cs.hcat((v[:, 0] / cs.sqrt(l[:, 0] + l[:, 1]), v[:, 1] / cs.sqrt(l[:, 0] + l[:, 1])))
        return point, v

    def e_c(self, point, theta):
        """
        Contouring error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: contouring error
        """
        point_r, v = self.get_path_parameters(theta, 0)  # point: vertical, v: horizontal
        n = cs.hcat((v[:, 1], -v[:, 0]))  # Creating a perpendicular vector
        # e_c = n*(point_r-point) #Why does this return a 2*1 vector???? format: [value, 0]
        e_c = cs.dot(n.T, (point_r - point))

        # print(f"contouring error: {e_c}")
        # e_c = vec(e_c[0, :] + e_c[1, :]) #old code
        return e_c

    def e_l(self, point, theta):
        """
        Lag error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: lag error
        """
        point_r, v = self.get_path_parameters(theta, 0)
        # e_l = v*(point_r-point) #Why does this return a 2*1 vector???? format: [value, 0]
        e_l = cs.dot(v.T, (point_r - point))
        # print(f"lateral error: {e_l}")
        # e_l = vec(e_l[0, :]+e_l[1, :]) #old code
        return e_l


def paperclip():
    focus_x = [0, 0]
    focus_y = [-1, 1]
    r = 2
    len_straight = focus_y[1] - focus_y[0]
    len_turn = r * np.pi
    ppm = 6
    num_straight = int(len_straight * ppm)
    num_turn = int(len_turn * ppm)
    x = np.hstack((np.linspace(focus_x[0] + r, focus_x[1] + r, num_straight),
                   focus_x[1] + r * np.cos(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_x[1] - r, focus_x[0] - r, num_straight),
                   focus_x[0] + r * np.cos(np.linspace(np.pi, 2 * np.pi, num_turn))
                   ))
    y = np.hstack((np.linspace(focus_y[0], focus_y[1], num_straight),
                   focus_y[1] + r * np.sin(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_y[1], focus_y[0], num_straight),
                   focus_y[0] + r * np.sin(np.linspace(np.pi, 2 * np.pi, num_turn))
                   ))
    x = np.roll(x, 6)
    y = np.roll(y, 6)
    points = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    delete_idx = []
    for i, point in enumerate(points):
        if i > 0:
            if np.linalg.norm(point - points[i - 1, :]) < 0.1:
                delete_idx += [i]
    points = np.delete(points, delete_idx, 0)
    vel = np.ones([points.shape[0], 1])

    return points, vel


def dented_paperclip():
    focus_x = [0, 0]
    focus_y = [-2, 2]
    r = 2
    len_straight = focus_y[1] - focus_y[0]
    len_turn = r * np.pi
    r_dent = 0.4
    len_dent = cosine_arc_length(r_dent, 2 * np.pi / len_straight, 0, len_straight)
    ppm = 4
    num_straight = int(len_straight * ppm)
    num_turn = int(len_turn * ppm)
    num_dent = int(len_dent * ppm)
    x = np.hstack((np.linspace(focus_x[1] + r, focus_x[0] + r, num_straight),
                   focus_x[1] + r * np.cos(np.linspace(0, np.pi, num_turn)),
                   -r + r_dent - r_dent * np.cos(np.linspace(0, 2 * np.pi, num_dent)),
                   focus_x[0] + r * np.cos(np.linspace(np.pi, 2 * np.pi, num_turn))
                   ))
    y = np.hstack((np.linspace(focus_y[0], focus_y[1], num_dent),
                   focus_y[1] + r * np.sin(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_y[1], focus_y[0], num_straight),
                   focus_y[0] + r * np.sin(np.linspace(np.pi, 2 * np.pi, num_turn))
                   ))
    x = np.roll(x, -15)
    y = np.roll(y, -15)
    points = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    delete_idx = []
    for i, point in enumerate(points):
        if i > 0:
            if np.linalg.norm(point - points[i - 1, :]) < 0.1:
                delete_idx += [i]
    points = np.delete(points, delete_idx, 0)
    vel = np.ones([points.shape[0], 1])

    return points, vel


def cosine_arc_length(amplitude, frequency, start, end):
    # Define the derivative of the cosine function
    def derivative_cos(x):
        return -amplitude * frequency * np.sin(frequency * x)

    # Define the integrand
    def integrand(x):
        return np.sqrt(1 + derivative_cos(x) ** 2)

    # Integrate the integrand function using scipy's quad function
    from scipy.integrate import quad
    arc_length, _ = quad(integrand, start, end)

    return arc_length


def slalom(loops=2, r=1, x0=0, y0=0):
    elements = 10
    angle = np.linspace(0, np.pi * (elements) / elements, elements)
    x = np.sin(angle) * r + x0
    y = np.cos(angle) * r - r + y0
    path_points = np.array([x[0], y[0]])  # Use a 2D array (shape (1, 2))

    # Forward loop for creating the slalom path
    for l_index in range(loops):
        neg = 1 if l_index % 2 == 0 else -1  # Alternate direction
        for i in range(1, elements):
            point = np.array([[x[i] * neg, y[i] - l_index * r * 2]])  # Create a 2D array for each point
            path_points = np.vstack([path_points, point])

    # Backward loop for returning along the path
    for l_index in range(loops - 1, 0, -1):
        neg = -1 if l_index % 2 == 0 else 1  # Alternate direction
        for i in range(1, elements):
            point = np.array([[x[elements - i - 1] * neg, -y[i] - (loops - l_index + 1) * 2 * r]])  # Create a 2D array
            path_points = np.vstack([path_points, point])

    # Final loop to close the path
    for i in range(1, elements):
        point = np.array([[-x[i] + x0, -y[i] - 2 * r + y0]])  # Closing the path
        path_points = np.vstack([path_points, point])

    vvelocities = np.ones([path_points.shape[0], 1])

    return path_points, vvelocities


def null_infty(laps=1, scale=1):
    path_points = 1.1 * np.array([[0, 0],
                                  [1 * np.cos(-np.pi / 4), 1.5 + np.sin(-np.pi / 4)],
                                  [1 * np.cos(np.pi / 8), 1.5 + np.sin(-np.pi / 8)],
                                  [1, 1.5],
                                  [1 * np.cos(np.pi / 8), 1.5 + np.sin(np.pi / 8)],
                                  [1 * np.cos(np.pi / 4), 1.5 + np.sin(np.pi / 4)],
                                  [1 * np.cos(3 * np.pi / 8), 1.5 + np.sin(3 * np.pi / 8)],
                                  [0, 2.5],
                                  [1 * np.cos(5 * np.pi / 8), 1.5 + np.sin(5 * np.pi / 8)],
                                  [1 * np.cos(6 * np.pi / 8), 1.5 + np.sin(6 * np.pi / 8)],
                                  [1 * np.cos(7 * np.pi / 8), 1.5 + np.sin(7 * np.pi / 8)],
                                  [-1, 1.5],
                                  [-1, 1],
                                  [-1, .5],
                                  [-1, 0],
                                  [-1, -.5],
                                  [-1, -1],
                                  [-1, -1.5],
                                  [1 * np.cos(np.pi + np.pi / 8), -1.5 + np.sin(np.pi + np.pi / 8)],
                                  [1 * np.cos(np.pi + np.pi / 4), -1.5 + np.sin(np.pi + np.pi / 4)],
                                  [1 * np.cos(np.pi + 3 * np.pi / 8), -1.5 + np.sin(np.pi + 3 * np.pi / 8)],
                                  [0, -2.5],
                                  [1 * np.cos(np.pi + 5 * np.pi / 8), -1.5 + np.sin(np.pi + 5 * np.pi / 8)],
                                  [1 * np.cos(np.pi + 6 * np.pi / 8), -1.5 + np.sin(np.pi + 6 * np.pi / 8)],
                                  [1 * np.cos(np.pi + 7 * np.pi / 8), -1.5 + np.sin(np.pi + 7 * np.pi / 8)],
                                  [1, -1.5],
                                  [1, -1],
                                  [1, -0.5],
                                  [1, 0],
                                  [1, 0.5],
                                  [1, 1],
                                  [1, 1.5],
                                  [1 * np.cos(np.pi / 8), 1.5 + np.sin(np.pi / 8)],
                                  [1 * np.cos(np.pi / 4), 1.5 + np.sin(np.pi / 4)],
                                  [1 * np.cos(3 * np.pi / 8), 1.5 + np.sin(3 * np.pi / 8)],
                                  [0, 2.5],
                                  [1 * np.cos(5 * np.pi / 8), 1.5 + np.sin(5 * np.pi / 8)],
                                  [1 * np.cos(6 * np.pi / 8), 1.5 + np.sin(6 * np.pi / 8)],
                                  [1 * np.cos(7 * np.pi / 8), 1.5 + np.sin(7 * np.pi / 8)],
                                  [-1, 1.5],
                                  [1 * np.cos(9 * np.pi / 8), 1.5 + np.sin(9 * np.pi / 8)],
                                  [1 * np.cos(10 * np.pi / 8), 1.5 + np.sin(10 * np.pi / 8)],
                                  [0, 0],
                                  [1 * np.cos(np.pi / 4), -1.5 + np.sin(np.pi / 4)],
                                  [1 * np.cos(np.pi / 8), -1.5 + np.sin(np.pi / 8)],
                                  [1, -1.5],
                                  [1 * np.cos(-np.pi / 8), -1.5 + np.sin(-np.pi / 8)],
                                  [1 * np.cos(-np.pi / 4), -1.5 + np.sin(-np.pi / 4)],
                                  [1 * np.cos(-3 * np.pi / 8), -1.5 + np.sin(-3 * np.pi / 8)],
                                  [0, -2.5],
                                  [1 * np.cos(-5 * np.pi / 8), -1.5 + np.sin(-5 * np.pi / 8)],
                                  [1 * np.cos(-6 * np.pi / 8), -1.5 + np.sin(-6 * np.pi / 8)],
                                  [1 * np.cos(-7 * np.pi / 8), -1.5 + np.sin(-7 * np.pi / 8)],
                                  [-1, -1.5],
                                  [1 * np.cos(-9 * np.pi / 8), -1.5 + np.sin(-9 * np.pi / 8)],
                                  [1 * np.cos(-10 * np.pi / 8), -1.5 + np.sin(-10 * np.pi / 8)],
                                  ]) * scale
    points = path_points
    for i in range(laps - 1):
        points = np.concatenate((points, path_points))

    points = np.concatenate((points, np.zeros((1, 2))))
    vvelocities = np.ones([points.shape[0], 1])

    return points, vvelocities


def eight():
    path_points = .5 * np.flip(np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 2],
            [4, 1],
            [4.5, 0],
            [4, -1],
            [3, -2],
            [2, -2],
            [1, -1],
            [0, 0],
            [-1, 1],
            [-2, 2],
            [-3, 2],
            [-4, 1],
            [-4.5, 0],
            [-4, -1],
            [-3, -2],
            [-2, -2],
            [-1, -1],
            [0, 0],
        ]), axis=1)
    vel = np.ones([path_points.shape[0], 1])
    return path_points, vel


def null_paperclip():
    points = 1.1 * np.array([[0, 0],
                             [1 * np.cos(-np.pi / 4), 1.5 + np.sin(-np.pi / 4)],
                             [1 * np.cos(np.pi / 8), 1.5 + np.sin(-np.pi / 8)],
                             [1, 1.5],
                             [1 * np.cos(np.pi / 8), 1.5 + np.sin(np.pi / 8)],
                             [1 * np.cos(np.pi / 4), 1.5 + np.sin(np.pi / 4)],
                             [1 * np.cos(3 * np.pi / 8), 1.5 + np.sin(3 * np.pi / 8)],
                             [0, 2.5],
                             [1 * np.cos(5 * np.pi / 8), 1.5 + np.sin(5 * np.pi / 8)],
                             [1 * np.cos(6 * np.pi / 8), 1.5 + np.sin(6 * np.pi / 8)],
                             [1 * np.cos(7 * np.pi / 8), 1.5 + np.sin(7 * np.pi / 8)],
                             [-1, 1.5],
                             [-1, 1],
                             [-1, .5],
                             [-1, 0],
                             [-1, -.5],
                             [-1, -1],
                             [-1, -1.5],
                             [1 * np.cos(np.pi + np.pi / 8), -1.5 + np.sin(np.pi + np.pi / 8)],
                             [1 * np.cos(np.pi + np.pi / 4), -1.5 + np.sin(np.pi + np.pi / 4)],
                             [1 * np.cos(np.pi + 3 * np.pi / 8), -1.5 + np.sin(np.pi + 3 * np.pi / 8)],
                             [0, -2.5],
                             [1 * np.cos(np.pi + 5 * np.pi / 8), -1.5 + np.sin(np.pi + 5 * np.pi / 8)],
                             [1 * np.cos(np.pi + 6 * np.pi / 8), -1.5 + np.sin(np.pi + 6 * np.pi / 8)],
                             [1 * np.cos(np.pi + 7 * np.pi / 8), -1.5 + np.sin(np.pi + 7 * np.pi / 8)],
                             [1, -1.5],
                             [1, -1],
                             [1, -0.5],
                             [1, 0],
                             [1, 0.5],
                             [1, 1],
                             [1, 1.5],
                             [1 * np.cos(np.pi / 8), 1.5 + np.sin(np.pi / 8)],
                             [1 * np.cos(np.pi / 4), 1.5 + np.sin(np.pi / 4)],
                             [1 * np.cos(3 * np.pi / 8), 1.5 + np.sin(3 * np.pi / 8)],
                             [0, 2.5],
                             [1 * np.cos(5 * np.pi / 8), 1.5 + np.sin(5 * np.pi / 8)],
                             [1 * np.cos(6 * np.pi / 8), 1.5 + np.sin(6 * np.pi / 8)],
                             [1 * np.cos(7 * np.pi / 8), 1.5 + np.sin(7 * np.pi / 8)],
                             [-1, 1.5],
                             [1 * np.cos(9 * np.pi / 8), 1.5 + np.sin(9 * np.pi / 8)],
                             [1 * np.cos(10 * np.pi / 8), 1.5 + np.sin(10 * np.pi / 8)],
                             [0, 0]]
                            )
    vel = 1.1 * np.array([.6,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .8,
                          .8,
                          .8,
                          .9,
                          1,
                          1.2,
                          1.3,
                          1.1,
                          .9,
                          .8,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .8,
                          .9,
                          1,
                          1.2,

                          1.3,
                          1.1,
                          .9,
                          .8,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .7,
                          .5]
                         )
    return points, vel


def train8(v=1):
    points = np.array([[0, 0],
                       [0.37, 0.41],
                       [1 * np.cos(-np.pi / 4), 1.5 + np.sin(-np.pi / 4)],
                       [1 * np.cos(np.pi / 8), 1.5 + np.sin(-np.pi / 8)],
                       [1, 1.5],
                       [1 * np.cos(np.pi / 8), 1.5 + np.sin(np.pi / 8)],
                       [1 * np.cos(np.pi / 4), 1.5 + np.sin(np.pi / 4)],
                       [1 * np.cos(3 * np.pi / 8), 1.5 + np.sin(3 * np.pi / 8)],
                       [0, 2.5],
                       [1 * np.cos(5 * np.pi / 8), 1.5 + np.sin(5 * np.pi / 8)],
                       [1 * np.cos(6 * np.pi / 8), 1.5 + np.sin(6 * np.pi / 8)],
                       [1 * np.cos(7 * np.pi / 8), 1.5 + np.sin(7 * np.pi / 8)],
                       [-1, 1.5],
                       [1 * np.cos(9 * np.pi / 8), 1.5 + np.sin(9 * np.pi / 8)],
                       [1 * np.cos(10 * np.pi / 8), 1.5 + np.sin(10 * np.pi / 8)],
                       [-0.37, 0.41],
                       [0, 0],
                       [0.37, -0.41],
                       [1 * np.cos(np.pi / 4), -1.5 + np.sin(np.pi / 4)],
                       [1 * np.cos(np.pi / 8), -1.5 + np.sin(np.pi / 8)],
                       [1, -1.5],
                       [1 * np.cos(-np.pi / 8), -1.5 + np.sin(-np.pi / 8)],
                       [1 * np.cos(-np.pi / 4), -1.5 + np.sin(-np.pi / 4)],
                       [1 * np.cos(-3 * np.pi / 8), -1.5 + np.sin(-3 * np.pi / 8)],
                       [0, -2.5],
                       [1 * np.cos(-5 * np.pi / 8), -1.5 + np.sin(-5 * np.pi / 8)],
                       [1 * np.cos(-6 * np.pi / 8), -1.5 + np.sin(-6 * np.pi / 8)],
                       [1 * np.cos(-7 * np.pi / 8), -1.5 + np.sin(-7 * np.pi / 8)],
                       [-1, -1.5],
                       [1 * np.cos(-9 * np.pi / 8), -1.5 + np.sin(-9 * np.pi / 8)],
                       [1 * np.cos(-10 * np.pi / 8), -1.5 + np.sin(-10 * np.pi / 8)],
                       [-0.37, -0.41],
                       [0, 0]])
    vel = v * np.ones([points.shape[0], 1])
    return points, vel

