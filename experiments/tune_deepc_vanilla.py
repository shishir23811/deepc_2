import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from controllers.deepc_vanilla import DeepCVanilla
from sim.bicycle import KinematicBicycle
from sim.references import figure_eight


def simulate(controller, sim_duration=40.0):
    dt = 0.05
    plant = KinematicBicycle(dt=dt)
    X0, Y0, psi0, v0 = controller._ref_state(0.0)
    plant.reset([X0, Y0, psi0, v0])
    state = plant.state.copy()

    errors = []
    measurements = []
    references = []

    n_steps = int(sim_duration / dt)
    for step in range(n_steps):
        t = step * dt
        ref_state = controller._ref_state(t)
        u = controller.solve(state, t)
        noisy = plant.step(u)
        state = plant.state.copy()

        references.append(ref_state)
        measurements.append(noisy)
        errors.append(np.linalg.norm(state[:2] - ref_state[:2]))

    return np.array(errors), np.vstack(measurements), np.vstack(references)


if __name__ == '__main__':
    lookaheads = [0.25, 0.3, 0.35, 0.4, 0.5]
    Kps = [1.0, 1.5, 2.0, 2.5]

    best = None
    results = []
    for la in lookaheads:
        for kp in Kps:
            controller = DeepCVanilla(lookahead_time=la, Kp_v=kp, Kv_v=0.2)
            errs, meas, refs = simulate(controller, sim_duration=40.0)
            mean_e = float(np.mean(errs))
            max_e = float(np.max(errs))
            results.append((la, kp, mean_e, max_e))
            print(f'la={la:.3f}, kp={kp:.2f} -> mean {mean_e:.4f}, max {max_e:.4f}')
            if best is None or max_e < best[0]:
                best = (max_e, mean_e, la, kp, meas, refs, errs)

    print('\nBest config:')
    print('lookahead, kp, mean_err, max_err =', best[2], best[3], best[1], best[0])

    # Save plot for best
    max_e, mean_e, la, kp, meas, refs, errs = best

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(refs[:,0], refs[:,1], label='Reference')
    ax.plot(meas[:,0], meas[:,1], label='DeepC-Vanilla')
    ax.axis('equal')
    ax.legend()
    ax.set_title(f'DeepC-Vanilla best: la={la:.2f}, kp={kp:.2f}, max={max_e:.3f} m')
    out = os.path.join(ROOT_DIR, 'experiments', 'deepc_vanilla_best.png')
    fig.tight_layout(); fig.savefig(out, dpi=200)
    print('Saved', out)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(np.arange(len(errs))*0.05, errs, label='Position error')
    ax2.axhline(0.1, color='r', linestyle='--', label='0.1 m')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Error [m]')
    ax2.legend(); fig2.tight_layout(); fig2.savefig(os.path.join(ROOT_DIR,'experiments','deepc_vanilla_best_error.png'), dpi=200)
    print('Saved error plot')
