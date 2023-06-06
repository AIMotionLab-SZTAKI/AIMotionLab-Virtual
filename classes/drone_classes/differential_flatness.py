import numpy as np
import scipy.interpolate as si
from scipy.spatial.transform import Rotation


def my_dot(a: np.ndarray, b: np.ndarray):
    return np.expand_dims(np.sum(a*b, axis=1), axis=1)


def my_cross(a: np.ndarray, b: np.ndarray):
    return np.vstack([np.cross(a[i, :], b[i, :]) for i in range(a.shape[0])])


def compute_state_trajectory_from_splines(spl, m, hook_mass, payload_mass, L, g, J, dt):
    x_spl = spl[0]
    y_spl = spl[1]
    z_spl = spl[2]
    yaw_spl = spl[3]
    xL = 7 * [[]]  # derivatives of payload position
    yL = 7 * [[]]  # derivatives of payload position
    zL = 7 * [[]]  # derivatives of payload position
    yaw = 3 * [[]]  # derivatives of yaw angle
    t = []
    mL = []
    for phase in range(6):
        for der in range(6):
            t_cur = np.arange(0, x_spl[phase][0][-1], dt)
            if len(t) < 1:
                t = t_cur.tolist()
            else:
                t = t + [t_ + t[-1] for t_ in t_cur]
            xL[der] = xL[der] + si.splev(t_cur, x_spl[phase], der=der).tolist()
            yL[der] = yL[der] + si.splev(t_cur, y_spl[phase], der=der).tolist()
            zL[der] = zL[der] + si.splev(t_cur, z_spl[phase], der=der).tolist()
            if der < 3:
                yaw[der] = yaw[der] + si.splev(t_cur, yaw_spl[phase], der=der).tolist()

        mass = hook_mass if phase in [0, 1, 4] else hook_mass + payload_mass
        mL = mL + [0 * elem + mass for elem in t_cur]
    mL = np.expand_dims(np.array(mL), axis=1)
    rL = [xL, yL, zL]
    for dim in range(3):
        rL[dim][6] = [0 * elem for elem in rL[dim][5]]  # sixth derivative is zero
    yaw = [np.expand_dims(np.array(yaw_), axis=1) for yaw_ in yaw]  # convert to numpy arrays
    rL = [np.asarray(x).T for x in zip(*rL)]  # convert to numpy arrays

    q = 5 * [np.array([])]
    T = 5 * [np.array([])]
    e3 = np.array([[0, 0, 1]])
    q[0] = -mL * (rL[2] + g * e3)
    T[0] = np.expand_dims(np.linalg.norm(q[0], axis=1), 1)
    q[0] = q[0] / T[0]
    T[1] = -mL * (my_dot(q[0], rL[3]))
    q[1] = -(mL * rL[3] + T[1] * q[0]) / T[0]
    T[2] = -mL * (my_dot(q[0], rL[4]) + my_dot(q[1], rL[3]))
    q[2] = -(mL * rL[4] + 2 * T[1] * q[1] + T[2] * q[0]) / T[0]
    T[3] = -mL * (my_dot(q[0], rL[5]) + 2 * my_dot(q[1], rL[4]) + my_dot(q[2], rL[3]))
    q[3] = -(mL * rL[5] + 3 * T[1] * q[2] + 3 * T[2] * q[1] + T[3] * q[0]) / T[0]
    T[4] = -mL * (my_dot(q[0], rL[6]) + 3 * my_dot(q[1], rL[5]) + 3 * my_dot(q[2], rL[4]) + my_dot(q[3], rL[3]))
    q[4] = -(mL * rL[6] + 4 * T[1] * q[3] + 6 * T[2] * q[2] + 4 * T[3] * q[1] + T[4] * q[0]) / T[0]

    r = rL.copy()
    for der in range(5):
        r[der] = rL[der] - L * q[der]

    F = 3 * [np.array([])]
    F[0] = np.expand_dims(np.linalg.norm(m * (r[2] + g * e3) + mL * (rL[2] + g * e3), axis=1), 1)

    Re3 = (m * (r[2] + g * e3) + mL * (rL[2] + g * e3)) / F[0]
    Re2 = my_cross(Re3, np.hstack((np.cos(yaw[0]), np.sin(yaw[0]), np.zeros_like(yaw[0]))))
    Re2 = Re2 / np.expand_dims(np.linalg.norm(Re2, axis=1), 1)
    Re1 = my_cross(Re2, Re3)

    F[1] = my_dot(m * r[3] + mL * rL[3], Re3)
    h_w = ((m * r[3] + mL * rL[3]) - F[1] * Re3) / F[0]
    w = np.hstack((-my_dot(h_w, Re2), my_dot(h_w, Re1), yaw[1] * np.expand_dims(Re3[:, 2], axis=1)))

    temp = my_cross(w, Re3)
    F[2] = my_dot(m * r[4] + mL * rL[4], Re3) - F[1] * my_dot(my_cross(w, temp), Re3)
    h_dw = 1 / F[0] * (m * r[4] + mL * rL[4]) - my_cross(w, my_cross(w, Re3)) - 2 * F[1] / F[0] * my_cross(w, Re3) - \
           1 / F[0] * F[2] * Re3
    dw = np.hstack((-my_dot(h_dw, Re2), my_dot(h_dw, Re1), yaw[2] * np.expand_dims(Re3[:, 2], axis=1)))

    tau = dw * J + my_cross(w, w * J)

    beta = np.arcsin(-q[0][:, 0])
    alpha = np.arcsin(-q[0][:, 1]/np.cos(beta))
    dbeta = -q[1][:, 0]/np.cos(beta)
    dalpha = (q[1][:, 1] + np.sin(alpha)*np.sin(beta)*dbeta) / np.cos(alpha) / np.cos(beta)

    eul = np.asarray([Rotation.from_matrix(np.vstack((Re1[i, :], Re2[i, :], Re3[i, :])).T).as_euler('xyz')
                      for i in range(Re1.shape[0])])
    pole_ang = np.vstack((alpha, beta, dalpha, dbeta)).T
    states = np.hstack((r[0], r[1], eul, w, pole_ang))
    inputs = np.hstack((F[0], tau))
    return states, inputs, mL[:, 0]
