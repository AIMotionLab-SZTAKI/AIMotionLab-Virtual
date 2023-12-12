import numpy as np
from scipy.spatial.transform import Rotation
from aiml_virtual.controller.controller_base import ControllerBase
import casadi as ca
import scipy


class LqrControl(ControllerBase):
    """
    Linear quadratic feedback control for quadcopters.
    """

    ################################################################################

    def __init__(self, mass, inertia, gravity):
        super().__init__(mass, inertia, gravity)
        self.Q = np.diag(np.hstack((100*np.ones(3), 100*np.ones(3),
                                    10*np.ones(3), np.ones(3))))
        self.R = 50*np.eye(4)
        self.K = self.compute_lqr()
        for i in range(self.K.shape[0]):
            for j in range(self.K.shape[1]):
                if np.abs(self.K[i, j]) < 1e-4:
                    self.K[i, j] = 0
        print(self.K)

    ################################################################################
    def compute_control(self, state, setpoint, time) -> np.array:
    
        cur_pos = state['pos']
        cur_quat = state['quat']
        cur_vel = state['vel']
        cur_ang_vel = state['ang_vel']
        target_pos = setpoint['target_pos']
        target_vel = setpoint['target_vel']
        target_rpy = setpoint['target_rpy']

        # print(cur_pos-target_pos)
        cur_quat = np.roll(cur_quat, -1)
        cur_eul = Rotation.from_quat(cur_quat).as_euler('xyz')
        cur_rot = Rotation.from_euler('xyz', target_rpy).as_matrix().T
        # cur_rot = np.eye(3)
        ctrl = -self.K @ np.hstack((cur_rot @ (cur_pos - target_pos), (cur_vel - target_vel), cur_eul - target_rpy, cur_ang_vel))
        ctrl[0] -= self.mass * self.gravity[2]
        return ctrl

    ######################################xxx

    def compute_lqr(self):
        m, g, Jx, Jy, Jz = ca.MX.sym('m'), ca.MX.sym('g'), ca.MX.sym('Jx'), ca.MX.sym('Jy'), ca.MX.sym('Jz')
        params = [m, g, Jx, Jy, Jz]

        # J = ca.MX(3, 3)
        # J[0, 0], J[1, 1], J[2, 2] = Jx, Jy, Jz
        J = ca.diag(ca.vertcat(Jx, Jy, Jz))
        Jinv = ca.diag(ca.vertcat(1/Jx, 1/Jy, 1/Jz))
        phi, theta, psi, rx, ry, rz, F, taux, tauy, tauz = ca.MX.sym('phi'), ca.MX.sym('theta'), ca.MX.sym('psi'), \
                                                                ca.MX.sym('rx'), ca.MX.sym('ry'), ca.MX.sym('rz'), \
                                                                ca.MX.sym('F'), ca.MX.sym('taux'), \
                                                                ca.MX.sym('tauy'), ca.MX.sym('tauz')
        omx, omy, omz, drx, dry, drz = ca.MX.sym('omx'), ca.MX.sym('omy'), ca.MX.sym('omz'), \
                                                    ca.MX.sym('drx'), ca.MX.sym('dry'), ca.MX.sym('drz')
        q_lst = [rx, ry, rz, phi, theta, psi]
        q = ca.vertcat(*q_lst)

        dq_lst = [drx, dry, drz, omx, omy, omz]
        dq = ca.vertcat(*dq_lst)

        r = ca.vertcat(rx, ry, rz)
        lam = ca.vertcat(*[phi, theta, psi])
        v = ca.vertcat(*[drx, dry, drz])
        om = ca.vertcat(*[omx, omy, omz])
        x_lst = [rx, ry, rz, drx, dry, drz, phi, theta, psi, omx, omy, omz]
        x = ca.vertcat(*x_lst)

        u_lst = [F, taux, tauy, tauz]
        u = ca.vertcat(*u_lst)

        R_lst = [[ca.cos(psi) * ca.cos(theta), ca.cos(psi) * ca.sin(phi) * ca.sin(theta) - ca.cos(phi) * ca.sin(psi),
                ca.sin(phi) * ca.sin(psi) + ca.cos(phi) * ca.cos(psi) * ca.sin(theta)],
                [ca.cos(theta) * ca.sin(psi), ca.cos(phi) * ca.cos(psi) + ca.sin(phi) * ca.sin(theta) * ca.sin(psi),
                ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.cos(psi) * ca.sin(phi)],
                [-ca.sin(theta), ca.sin(phi) * ca.cos(theta), ca.cos(phi) * ca.cos(theta)]]
        R = ca.vertcat(ca.horzcat(*R_lst[0]), ca.horzcat(*R_lst[1]), ca.horzcat(*R_lst[2]))

        Q_lst = [[1, 0, -ca.sin(theta)], [0, ca.cos(phi), ca.sin(phi) * ca.cos(theta)],
                [0, -ca.sin(phi), ca.cos(phi) * ca.cos(theta)]]
        Q = ca.vertcat(ca.horzcat(*Q_lst[0]), ca.horzcat(*Q_lst[1]), ca.horzcat(*Q_lst[2]))

        invQ_lst = [[1, ca.sin(phi) * ca.tan(theta), ca.cos(phi) * ca.tan(theta)], [0, ca.cos(phi), -ca.sin(phi)],
                    [0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta)]]
        invQ = ca.vertcat(ca.horzcat(*invQ_lst[0]), ca.horzcat(*invQ_lst[1]), ca.horzcat(*invQ_lst[2]))

        f_expr = ca.vertcat(v, F*R[:2, 2]/m, -g + F*R[2, 2]/m, invQ @ om, Jinv @ (u[1:] - ca.cross(om, J @ om)))
        f = ca.Function('f', x_lst + u_lst + params, [f_expr])
        dfdx = ca.Function('f', x_lst + u_lst + params, [ca.jacobian(f_expr, x)])
        dfdu = ca.Function('f', x_lst + u_lst + params, [ca.jacobian(f_expr, u)])

        x0 = 12*tuple([0])
        param_num = (self.mass, -self.gravity[2], self.inertia[0], self.inertia[1], self.inertia[2])
        u0 = (param_num[0]*param_num[1], 0, 0, 0)
        A = np.array(dfdx(*(x0 + u0 + param_num)))
        B = np.array(dfdu(*(x0 + u0 + param_num)))
        # Q = np.diag(np.hstack((20 * np.ones(3), 5 * np.ones(3), 0.1 * np.ones(3), 0.1 * np.ones(3))))
        # R = np.diag((1, 10, 10, 10))
        # Q = np.eye(12)
        # R = np.eye(4)

        S = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ B.T @ S
        return K