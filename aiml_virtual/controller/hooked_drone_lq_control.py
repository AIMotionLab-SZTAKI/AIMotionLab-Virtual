import numpy as np
import scipy as si
import control
import casadi as ca
from aiml_virtual.controller.controller_base import ControllerBase
from aiml_virtual.controller.differential_flatness import compute_state_trajectory_casadi


class LqrLoadControl(ControllerBase):
    def __init__(self, mass, inertia, gravity):
        super().__init__(mass, inertia, gravity)
        self.gravity = np.abs(self.gravity[2])
        self.mass = self.mass[0]
        self.payload_mass = 0.05
        self.L = 0.4

        # Weight matrices for continuous time LQR
        self.Q = np.diag(np.hstack((10 * np.ones(2), 100, 1 * np.ones(2), 10, 0.1 * np.ones(2), 1, 0.4 * np.ones(3),
                                    0.1, 0.1, 0.05 * np.ones(2))))
        self.R = np.diag([1, 80, 80, 80])

        self.dt = 0.01

        self.f, self.fx, self.fu = self.dyn_model()

        self.K_lti = np.zeros((self.R.shape[0], self.Q.shape[0]))
        self.compute_lti_lqr()
        self.K = None
        self.controller_step = 0

    def dyn_model(self):
        m, g, L, Jx, Jy, Jz = self.mass, self.gravity, self.L, self.inertia[0], \
                              self.inertia[1], self.inertia[2]
        mL = ca.MX.sym('mL')
        phi, theta, psi, rx, ry, rz, alpha, beta, F, taux, tauy, tauz = ca.MX.sym('phi'), ca.MX.sym('theta'), \
                                                                        ca.MX.sym('psi'), ca.MX.sym('rx'),\
                                                                        ca.MX.sym('ry'), ca.MX.sym('rz'), \
                                                                        ca.MX.sym('alpha'), ca.MX.sym('beta'),\
                                                                        ca.MX.sym('F'), ca.MX.sym('taux'), \
                                                                        ca.MX.sym('tauy'), ca.MX.sym('tauz')
        omx, omy, omz, drx, dry, drz, dalpha, dbeta = ca.MX.sym('omx'), ca.MX.sym('omy'), ca.MX.sym('omz'), \
                                                      ca.MX.sym('drx'), ca.MX.sym('dry'), ca.MX.sym('drz'), \
                                                      ca.MX.sym('dalpha'), ca.MX.sym('dbeta')
        om = ca.vertcat(omx, omy, omz)
        # ddalpha, ddbeta = ca.MX.sym('ddalpha'), ca.MX.sym('ddbeta')
        S_phi = ca.sin(phi)
        S_theta = ca.sin(theta)
        S_psi = ca.sin(psi)
        C_phi = ca.cos(phi)
        C_theta = ca.cos(theta)
        C_psi = ca.cos(psi)
        S_alpha = ca.sin(alpha)
        S_beta = ca.sin(beta)
        C_alpha = ca.cos(alpha)
        C_beta = ca.cos(beta)
        W = [[1, 0, -S_theta], [0, C_phi, C_theta * S_phi], [0, -S_phi, C_theta * C_phi]]
        W = self.list_to_casadi_matrix(W)
        R = (self.list_to_casadi_matrix([[1, 0, 0], [0, C_phi, S_phi], [0, -S_phi, C_phi]]) @
             self.list_to_casadi_matrix([[C_theta, 0, -S_theta], [0, 1, 0], [S_theta, 0, C_theta]]) @
             self.list_to_casadi_matrix([[C_psi, S_psi, 0], [-S_psi, C_psi, 0], [0, 0, 1]])).T
        R_q = (self.list_to_casadi_matrix([[1, 0, 0], [0, C_alpha, S_alpha], [0, -S_alpha, C_alpha]]) @
               self.list_to_casadi_matrix([[C_beta, 0, -S_beta], [0, 1, 0], [S_beta, 0, C_beta]])).T
        q = R_q @ ca.vertcat(0, 0, -1)
        ang = ca.vertcat(alpha, beta)
        dang = ca.vertcat(dalpha, dbeta)
        # ddang = ca.vertcat(ddalpha, ddbeta)
        dq = ca.jacobian(q, ang) @ dang
        M1 = ca.jacobian(dq, ang)
        M2 = ca.jacobian(dq, dang)
        # ddq = ca.jacobian(dq, ca.vertcat(ang, dang)) @ ca.vertcat(dang, ddang)
        # ddq = M1 @ dang + M2 @ ddang
        M2_inv = ca.pinv(M2)
        e3 = ca.vertcat(0, 0, 1)
        ddq = (1 / (m * L) * ca.cross(q, ca.cross(q, F * R @ e3)) - (dq.T @ dq) * q)
        # f1 = 1 / (m + mL) * (q.dot(F*R*e3) - m*L*dq.dot(dq))*q - sp.Matrix([0, 0, g])
        f1 = 1 / (m + mL) * F * R @ e3 - g * e3 - 1 / (m + mL) * mL * L * ddq
        f2 = ca.diag(ca.vertcat(1 / Jx, 1 / Jy, 1 / Jz)) @ (ca.vertcat(taux, tauy, tauz) -
             ca.cross(om, ca.vertcat(Jx * omx, Jy * omy, Jz * omz)))
        f3 = M2_inv @ (1 / (m * L) * ca.cross(q, ca.cross(q, F * R @ e3)) - (dq.T @ dq) * q - M1 @ dang)
        f = ca.vertcat(drx, dry, drz, f1, ca.inv(W) @ om, f2, dalpha, dbeta, f3)
        x = ca.vertcat(rx, ry, rz, drx, dry, drz, phi, theta, psi, omx, omy, omz, alpha, beta, dalpha, dbeta)
        u = ca.vertcat(F, taux, tauy, tauz)
        dfdx = ca.jacobian(f, x)
        dfdu = ca.jacobian(f, u)
        f_fun = ca.Function('f', [x, u, mL], [f])
        dfdx_fun = ca.Function('fx', [x, u, mL], [dfdx])
        dfdu_fun = ca.Function('fu', [x, u, mL], [dfdu])
        return f_fun, dfdx_fun, dfdu_fun

    def compute_control(self, state, setpoint, time):
        cur_load_pos = state["pos"]
        cur_load_vel = state["vel"]
        cur_pole_eul = state["pole_eul"]
        cur_pole_ang_vel = state["pole_ang_vel"]
        cur_quat = state["quat"]
        cur_ang_vel = state["ang_vel"]
        target_load_pos = setpoint["target_pos"] #+ np.array([0, 0, 0.4])
        target_load_vel = setpoint["target_vel"]
        target_eul = setpoint["target_eul"]
        target_ang_vel = setpoint["target_ang_vel"]
        target_pole_eul = setpoint["target_pole_eul"]
        target_pole_ang_vel = setpoint["target_pole_ang_vel"]
        cur_quat = np.roll(cur_quat, -1)
        cur_eul = si.spatial.transform.Rotation.from_quat(cur_quat).as_euler('xyz')
        while cur_eul[2] - target_eul[2] > np.pi:
            cur_eul[2] -= 2 * np.pi
        while cur_eul[2] - target_eul[2] < -np.pi:
            cur_eul[2] += 2 * np.pi
        state = np.hstack((cur_load_pos, cur_load_vel, cur_eul, cur_ang_vel, cur_pole_eul, cur_pole_ang_vel))
        target = np.hstack((target_load_pos, target_load_vel, target_eul, target_ang_vel, target_pole_eul,
                            target_pole_ang_vel))
        ctrl = -self.K_lti @ (state - target)
        ctrl[0] = (self.mass + self.payload_mass) * self.gravity + ctrl[0]
        return ctrl

    def compute_lti_lqr(self, yaw=0, states=np.zeros(16), inputs=np.zeros(4)):
        x0 = states
        x0[8] = yaw
        u0 = inputs
        u0[0] = (self.mass + self.payload_mass)*self.gravity

        A = np.array(self.fx(x0, u0, self.payload_mass))
        B = np.array(self.fu(x0, u0, self.payload_mass))

        A_d = np.eye(16) + self.dt * A
        B_d = self.dt * B
        self.K_lti, _, _ = control.dlqr(A_d, B_d, self.Q, self.R, method='scipy')

    @staticmethod
    def list_to_casadi_matrix(lst):
        # convert list of lists to casadi matrix
        return ca.vertcat(*[ca.horzcat(*lst_) for lst_ in lst])


class LtvLqrLoadControl(LqrLoadControl):
    def __init__(self, mass, inertia, gravity):
        super().__init__(mass, inertia, gravity)

    def compute_ltv_lqr(self, target_state, target_input, payload_mass, dt):
        N = target_input.shape[0]
        # compute state and input matrix
        A = np.zeros((N, target_state.shape[1], target_state.shape[1]))
        B = np.zeros((N, target_state.shape[1], target_input.shape[1]))

        for i in range(N):
            x0 = target_state[i, :]
            u0 = target_input[i, :]
            mL = payload_mass[i]
            A[i, :, :] = np.eye(target_state.shape[1]) + self.dt * np.array(self.fx(x0, u0, mL))
            B[i, :, :] = self.dt * np.array(self.fu(x0, u0, mL))

        self.K = np.zeros((N, target_input.shape[1], target_state.shape[1]))
        P = np.zeros((N, target_state.shape[1], target_state.shape[1]))
        P[-1, :, :] = self.Q

        # Backward pass
        for i in range(2, N + 1):
            self.K[N - i, :, :] = np.linalg.inv(self.R + B[N - i, :, :].T @ P[N - i + 1, :, :] @ B[N - i, :, :]) @ \
                                  B[N - i, :, :].T @ P[N - i + 1, :, :] @ A[N - i, :, :]
            P[N - i, :, :] = self.Q + self.K[N - i, :, :].T @ self.R @ self.K[N - i, :, :] + \
                             (A[N - i, :, :] - B[N - i, :, :] @ self.K[N - i, :, :]).T @ P[N - i + 1, :, :] @ \
                             (A[N - i, :, :] - B[N - i, :, :] @ self.K[N - i, :, :])

    def compute_control(self, state, setpoint, time):
        cur_pos = state["pos"]
        cur_vel = state["vel"]
        cur_pole_eul = state["pole_eul"]
        cur_pole_ang_vel = state["pole_ang_vel"]
        cur_quat = state["quat"]
        cur_ang_vel = state["ang_vel"]
        target_pos = setpoint["target_pos"] #+ np.array([0, 0, 0.4])
        target_vel = setpoint["target_vel"]
        target_eul = setpoint["target_eul"]
        target_ang_vel = setpoint["target_ang_vel"]
        target_pole_eul = setpoint["target_pole_eul"]
        target_pole_ang_vel = setpoint["target_pole_ang_vel"]
        target_thrust = setpoint["target_thrust"]
        target_torques = setpoint["target_torques"]
        cur_quat = np.roll(cur_quat, -1)
        cur_eul = si.spatial.transform.Rotation.from_quat(cur_quat).as_euler('xyz')
        while cur_eul[2] - target_eul[2] > np.pi:
            cur_eul[2] -= 2 * np.pi
        while cur_eul[2] - target_eul[2] < -np.pi:
            cur_eul[2] += 2 * np.pi
        state = np.hstack((cur_pos, cur_vel, cur_eul, cur_ang_vel, cur_pole_eul, cur_pole_ang_vel))
        target = np.hstack((target_pos, target_vel, target_eul, target_ang_vel, target_pole_eul,
                            target_pole_ang_vel))
        self.K = self.K_fun(self.ref, time)
        ctrl = np.hstack((target_thrust, target_torques)) - self.K @ (state - target)
        # ctrl = np.array([5.9, 0, 0, 0])
        # print((state-target)[:3])
        if np.isnan(np.sum(ctrl)):
            ctrl = np.zeros(4)
            print("NaN found in control input vector")
        return ctrl

    def setup_hook_up(self, ref, hook_mass, payload_mass):
        self.ref = ref

        x_f_hook, u_f_hook = compute_state_trajectory_casadi(ref, payload_mass=hook_mass)
        x_f_load, u_f_load = compute_state_trajectory_casadi(ref, payload_mass=hook_mass + payload_mass)

        t = np.arange(0.0001, ref.segment_times[-1] + 10, 0.01)

        def x_f(t_):
            if isinstance(t_, np.ndarray):
                t_ = t_.T
            elif isinstance(t_, float):
                t_ = [t_]
            return np.asarray(
                [x_f_load(t__) if ref.segment_times[1] < t__ <= ref.segment_times[3] else x_f_hook(t__) for t__ in t_])[
                   :, :, 0]

        def u_f(t_):
            if isinstance(t_, np.ndarray):
                t_ = t_.T
            elif isinstance(t_, float):
                t_ = [t_]
            return np.asarray(
                [u_f_load(t__) if ref.segment_times[1] < t__ <= ref.segment_times[3] else u_f_hook(t__) for t__ in t_])[
                   :, :, 0]

        idx_sec_1 = [t_ <= ref.segment_times[1] for t_ in t]
        idx_sec_2 = [ref.segment_times[1] < t_ <= ref.segment_times[3] for t_ in t]
        idx_sec_3 = [ref.segment_times[3] < t_ for t_ in t]
        x = x_f(np.expand_dims(t, 0))
        # x[0, :] = 0
        u = u_f(np.expand_dims(t, 0))
        # u[0, :] = 0

        self.compute_ltv_lqr(x[idx_sec_1, :], u[idx_sec_1, :], sum(idx_sec_1) * [hook_mass], 0.01)
        K_sec_1 = self.K
        self.compute_ltv_lqr(x[idx_sec_2, :], u[idx_sec_2, :], sum(idx_sec_2) * [hook_mass + payload_mass], 0.01)
        K_sec_2 = self.K
        self.compute_ltv_lqr(x[idx_sec_3, :], u[idx_sec_3, :], sum(idx_sec_3) * [hook_mass], 0.01)
        K_sec_3 = self.K

        self.K_lst_1 = u.shape[1] * x.shape[1] * [ref.t]
        self.K_lst_2 = u.shape[1] * x.shape[1] * [ref.t]
        self.K_lst_3 = u.shape[1] * x.shape[1] * [ref.t]
        for i in range(u.shape[1]):
            for j in range(x.shape[1]):
                self.K_lst_1[i * x.shape[1] + j] = si.interpolate.interp1d(t[idx_sec_1], K_sec_1[:, i, j],
                                                                         fill_value='extrapolate')
                self.K_lst_2[i * x.shape[1] + j] = si.interpolate.interp1d(t[idx_sec_2], K_sec_2[:, i, j],
                                                                         fill_value='extrapolate')
                self.K_lst_3[i * x.shape[1] + j] = si.interpolate.interp1d(t[idx_sec_3], K_sec_3[:, i, j],
                                                                         fill_value='extrapolate')

    def K_fun(self, ref, t_):
        K_lst = self.K_lst_1 if t_ <= ref.segment_times[1] else self.K_lst_2 if t_ <= ref.segment_times[3] else self.K_lst_3
        n = 16
        m = 4
        return np.asarray([[K_lst[i * n + j](t_) for j in range(n)] for i in range(m)])