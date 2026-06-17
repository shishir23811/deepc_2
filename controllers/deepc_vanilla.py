import numpy as np
from sim.references import figure_eight


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class DeepCVanilla:
    """Simple baseline controller (pure pursuit + speed PD).

    API: controller.solve(x, t) -> [delta, a]
    x: 4-element state [X, Y, psi, v]
    t: current time
    """
    def __init__(self, wheel_base=0.3, lookahead_time=0.6, Kp_v=1.2, Kv_v=0.2, max_delta=np.deg2rad(25), max_a=2.0):
        self.wheel_base = wheel_base
        self.lookahead_time = lookahead_time
        self.Kp_v = Kp_v
        self.Kv_v = Kv_v
        self.max_delta = max_delta
        self.max_a = max_a

    def reference_state(self, t):
        return self._ref_state(t)

    def _ref_state(self, t):
        eps = 1e-4
        X, Y, psi = figure_eight(t)
        X2, Y2, psi2 = figure_eight(t + eps)
        v = np.hypot(X2 - X, Y2 - Y) / eps
        return np.array([X, Y, psi, v])

    def solve(self, x0, t0):
        x0 = np.asarray(x0, dtype=float).reshape(4)
        X, Y, psi, v = x0

        # look-ahead reference point
        t_la = t0 + self.lookahead_time
        X_ref, Y_ref, psi_ref, v_ref = self._ref_state(t_la)

        # vector from vehicle to lookahead
        dx = X_ref - X
        dy = Y_ref - Y
        Ld = np.hypot(dx, dy)
        if Ld < 1e-6:
            alpha = 0.0
        else:
            alpha = wrap_angle(np.arctan2(dy, dx) - psi)

        # Pure pursuit steering law
        if Ld < 1e-6:
            delta = 0.0
        else:
            # ensure numerical safety
            delta = np.arctan2(2.0 * self.wheel_base * np.sin(alpha), max(1e-3, Ld))

        # Longitudinal PD control to track reference speed
        v_err = v_ref - v
        a = self.Kp_v * v_err - self.Kv_v * 0.0

        # Clip
        delta = float(np.clip(delta, -self.max_delta, self.max_delta))
        a = float(np.clip(a, -self.max_a, self.max_a))

        return np.array([delta, a])
