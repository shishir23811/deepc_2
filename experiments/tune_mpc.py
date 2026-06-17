import sys
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from controllers.mpc import ModelPredictiveController
from sim.bicycle import KinematicBicycle


def simulate(controller, duration=10.0):
    plant = KinematicBicycle(dt=controller.dt)
    plant.reset(controller.reference_state(0.0).tolist())
    n_steps = int(duration / controller.dt)
    state = plant.state.copy()
    errors = []
    for step in range(n_steps):
        t = step * controller.dt
        ref = controller.reference_state(t)
        u = controller.solve(state, t)
        state = plant.step(u)
        errors.append(np.linalg.norm(state[:2] - ref[:2]))
    return float(np.mean(errors))

if __name__ == '__main__':
    configs = [
        {'horizon': 10, 'A': 5.0, 'omega': 0.2},
        {'horizon': 15, 'A': 5.0, 'omega': 0.2},
        {'horizon': 10, 'A': 4.0, 'omega': 0.1},
        {'horizon': 15, 'A': 4.0, 'omega': 0.1},
        {'horizon': 12, 'A': 3.0, 'omega': 0.1},
    ]

    print('horizon,A,omega,mean_error')
    for cfg in configs:
        controller = ModelPredictiveController(
            dt=0.05,
            horizon=cfg['horizon'],
            Q=np.diag([40.0, 40.0, 8.0, 0.05]),
            R=np.diag([1.0, 0.05]),
            Qf=np.diag([50.0, 50.0, 10.0, 0.05]),
            reference_func=None,
            reference_kwargs={},
        )
        # patch reference function and kwargs after initialization to avoid import loop
        from sim.references import figure_eight
        controller.reference_func = figure_eight
        controller.reference_kwargs = {'A': cfg['A'], 'omega': cfg['omega']}

        err = simulate(controller, duration=5.0)
        print(cfg['horizon'], cfg['A'], cfg['omega'], err)
