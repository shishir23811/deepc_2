import numpy as np
import cvxpy as cp

from sim.references import figure_eight


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ModelPredictiveController:
    def __init__(
        self,
        dt=0.05,
        horizon=20,
        wheel_base=0.3,
        Q=None,
        R=None,
        Qf=None,
        delta_limit=np.deg2rad(25.0),
        a_limit=2.0,
        reference_func=None,
        reference_kwargs=None,
    ):
        self.dt = dt
        self.horizon = horizon
        self.wheel_base = wheel_base
        self.delta_limit = delta_limit
        self.a_limit = a_limit
        self.reference_func = figure_eight if reference_func is None else reference_func
        self.reference_kwargs = {} if reference_kwargs is None else reference_kwargs

        self.Q = np.diag([20.0, 20.0, 5.0, 0.1]) if Q is None else Q
        self.R = np.diag([10.0, 0.1]) if R is None else R
        self.Qf = np.diag([30.0, 30.0, 8.0, 0.1]) if Qf is None else Qf

    def reference_state(self, t):
        eps = 1e-3
        X, Y, psi = self.reference_func(t, **self.reference_kwargs)
        X2, Y2, psi2 = self.reference_func(t + eps, **self.reference_kwargs)
        v = np.hypot(X2 - X, Y2 - Y) / eps
        return np.array([X, Y, psi, v])

    def _reference_input(self, x_ref, x_ref_next):
        _, _, psi_ref, v_ref = x_ref
        _, _, psi_ref_next, v_ref_next = x_ref_next

        if v_ref <= 1e-3:
            delta_ref = 0.0
        else:
            dpsi = wrap_angle(psi_ref_next - psi_ref)
            psi_dot = dpsi / self.dt
            delta_ref = np.arctan((psi_dot * self.wheel_base) / max(1e-3, v_ref))

        a_ref = (v_ref_next - v_ref) / self.dt
        return np.array([delta_ref, a_ref])

    def _dynamics(self, x, u):
        X, Y, psi, v = x
        delta, a = u

        X_n = X + v * np.cos(psi) * self.dt
        Y_n = Y + v * np.sin(psi) * self.dt
        psi_n = psi + (v / self.wheel_base) * np.tan(delta) * self.dt
        v_n = v + a * self.dt

        return np.array([X_n, Y_n, psi_n, v_n])

    def _linearize(self, x_ref, u_ref):
        X, Y, psi, v = x_ref
        delta, a = u_ref

        A = np.eye(4)
        A[0, 2] = -v * np.sin(psi) * self.dt
        A[0, 3] = np.cos(psi) * self.dt
        A[1, 2] = v * np.cos(psi) * self.dt
        A[1, 3] = np.sin(psi) * self.dt
        A[2, 3] = np.tan(delta) / self.wheel_base * self.dt

        B = np.zeros((4, 2))
        B[2, 0] = (v / self.wheel_base) * self.dt / (np.cos(delta) ** 2)
        B[3, 1] = self.dt

        f_ref = self._dynamics(x_ref, u_ref)
        c = f_ref - A.dot(x_ref) - B.dot(u_ref)

        return A, B, c

    def solve(self, x0, t0, u_prev=None):
        x0 = np.asarray(x0, dtype=float).reshape(4)
        u_prev = np.zeros(2, dtype=float) if u_prev is None else np.asarray(u_prev, dtype=float).reshape(2)

        x_ref = [self.reference_state(t0 + i * self.dt) for i in range(self.horizon + 1)]
        u_ref = [self._reference_input(x_ref[i], x_ref[i + 1]) for i in range(self.horizon)]

        # Align reference yaw to avoid large 2*pi discontinuities in the cost.
        # The reference psi is computed with np.unwrap in the path generator.
        x = cp.Variable((self.horizon + 1, 4))
        u = cp.Variable((self.horizon, 2))

        cost = 0.0
        constraints = [x[0, :] == x0]

        for i in range(self.horizon):
            A_i, B_i, c_i = self._linearize(x_ref[i], u_ref[i])
            constraints += [x[i + 1, :] == A_i @ x[i, :] + B_i @ u[i, :] + c_i]

            err_x = x[i, :] - x_ref[i]
            err_u = u[i, :] - u_ref[i]
            cost += cp.quad_form(err_x, self.Q)
            cost += cp.quad_form(err_u, self.R)

            constraints += [u[i, 0] >= -self.delta_limit]
            constraints += [u[i, 0] <= self.delta_limit]
            constraints += [u[i, 1] >= -self.a_limit]
            constraints += [u[i, 1] <= self.a_limit]

        err_final = x[self.horizon, :] - x_ref[self.horizon]
        cost += cp.quad_form(err_final, self.Qf)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            problem.solve(solver=cp.ECOS, verbose=False)

        if x.value is None or u.value is None:
            return np.array([0.0, 0.0], dtype=float)

        first_u = np.array(u.value[0, :]).reshape(2)
        first_u[0] = float(np.clip(first_u[0], -self.delta_limit, self.delta_limit))
        first_u[1] = float(np.clip(first_u[1], -self.a_limit, self.a_limit))
        return first_u
