import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure top-level package imports work when running from experiments/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from controllers.mpc_v2 import ModelPredictiveController
from sim.bicycle import KinematicBicycle
from sim.references_v2 import figure_eight


def run_experiment():
    dt = 0.05
    # One full figure-8 period = 2*pi/omega = 2*pi/0.2 ≈ 31.4 s.
    # Run for ~2 full laps so we exercise both the tip-wrap and the crossing.
    sim_duration = 65.0
    horizon = 20

    controller = ModelPredictiveController(
        dt=dt,
        horizon=horizon,
        Q=np.diag([80.0, 80.0, 12.0, 0.1]),
        R=np.diag([1.0, 0.05]),
        Qf=np.diag([100.0, 100.0, 15.0, 0.1]),
        reference_func=figure_eight,
        reference_kwargs={"A": 10.0, "omega": 0.2},
    )
    x0 = controller.reference_state(0.0)
    plant = KinematicBicycle(dt=dt)
    plant.reset(x0.tolist())

    measurements = []
    references = []
    controls = []
    errors = []

    n_steps = int(sim_duration / dt)
    state = plant.state.copy()

    for step in range(n_steps):
        t = step * dt
        ref_state = controller.reference_state(t)
        references.append(ref_state)

        u = controller.solve(state, t)
        controls.append(u)

        noisy_output = plant.step(u)
        measurements.append(noisy_output)
        state = plant.state.copy()

        position_error = np.linalg.norm(state[:2] - ref_state[:2])
        errors.append(position_error)

    measurements = np.vstack(measurements)
    references = np.vstack(references)
    controls = np.vstack(controls)
    errors = np.array(errors)

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"Figure-8 MPC tracking: mean error = {mean_error:.4f} m, max error = {max_error:.4f} m")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(references[:, 0], references[:, 1], label="Figure-8 reference", color="tab:blue", linewidth=2)
    ax.plot(measurements[:, 0], measurements[:, 1], label="MPC trajectory", color="tab:orange", linewidth=2)
    ax.set_title("Figure-8 Tracking with MPC")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.legend()
    ax.grid(True)

    output_path = os.path.join(ROOT_DIR, "experiments", "figure_eight_mpc_v2.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved trajectory plot to: {output_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(np.arange(len(errors)) * dt, errors, label="Position error")
    ax2.axhline(0.1, color="tab:red", linestyle="--", label="0.1 m target")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [m]")
    ax2.set_title("Figure-8 Tracking Error")
    ax2.grid(True)
    ax2.legend()
    error_plot = os.path.join(ROOT_DIR, "experiments", "figure_eight_error_v2.png")
    fig2.tight_layout()
    fig2.savefig(error_plot, dpi=200)
    print(f"Saved error plot to: {error_plot}")

    return mean_error, max_error, output_path, error_plot


if __name__ == "__main__":
    run_experiment()
