import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from controllers.mpc import ModelPredictiveController
from sim.bicycle import KinematicBicycle
from sim.references import figure_eight


def simulate(A, omega, horizon, duration=20.0):
    dt = 0.05
    controller = ModelPredictiveController(
        dt=dt,
        horizon=horizon,
        Q=np.diag([80.0, 80.0, 12.0, 0.1]),
        R=np.diag([1.0, 0.05]),
        Qf=np.diag([100.0, 100.0, 15.0, 0.1]),
        reference_func=figure_eight,
        reference_kwargs={"A": A, "omega": omega},
    )
    plant = KinematicBicycle(dt=dt)
    plant.reset(controller.reference_state(0.0).tolist())
    state = plant.state.copy()

    errors = []
    n_steps = int(duration / dt)
    for step in range(n_steps):
        t = step * dt
        ref_state = controller.reference_state(t)
        u = controller.solve(state, t)
        state = plant.step(u)
        errors.append(np.linalg.norm(state[:2] - ref_state[:2]))

    return float(np.mean(errors)), float(np.max(errors))


if __name__ == '__main__':
    configs = [
        {'A': 5.0, 'omega': 0.05, 'horizon': 12},
        {'A': 5.0, 'omega': 0.05, 'horizon': 15},
        {'A': 5.0, 'omega': 0.08, 'horizon': 15},
        {'A': 4.0, 'omega': 0.05, 'horizon': 12},
        {'A': 4.0, 'omega': 0.08, 'horizon': 15},
        {'A': 10.0, 'omega': 0.1, 'horizon': 20},
    ]

    print('A,omega,horizon,mean_error,max_error')
    for cfg in configs:
        mean_err, max_err = simulate(cfg['A'], cfg['omega'], cfg['horizon'], duration=20.0)
        print(cfg['A'], cfg['omega'], cfg['horizon'], f'{mean_err:.4f}', f'{max_err:.4f}')
