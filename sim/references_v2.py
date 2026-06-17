import numpy as np


def figure_eight(t, A=5.0, omega=0.2):
    '''
    Analytical figure-8 reference path.
    Return: [X, Y, psi]  (psi is unwrapped / continuous — no ±pi jumps)

    The raw arctan2 heading has a ≈360° discontinuity at the tips of the
    figure-8 (where dX passes through zero and arctan2 flips between -pi
    and +pi).  This discontinuity enters the MPC cost function as a ~2pi
    yaw-error spike, causing violent control commands and trajectory loss.

    Fix: compute psi on a fine grid from t=0 to the requested t and
    use np.unwrap to obtain a smooth, continuous angle.
    '''
    t_val = float(t)

    if t_val <= 0.0:
        dX0 = A * omega * np.cos(0.0)
        dY0 = 2.0 * A * omega * np.cos(0.0)
        psi0 = np.arctan2(dY0, dX0)
        return np.array([0.0, 0.0, psi0])

    # Sample at ~100 Hz to resolve the fastest heading changes
    n = max(int(t_val * 100) + 2, 3)
    t_grid = np.linspace(0.0, t_val, n)

    dX_g = A * omega * np.cos(omega * t_grid)
    dY_g = 2.0 * A * omega * np.cos(2.0 * omega * t_grid)
    psi_unwrapped = np.unwrap(np.arctan2(dY_g, dX_g))

    X = A * np.sin(omega * t_val)
    Y = A * np.sin(2.0 * omega * t_val)
    psi = float(psi_unwrapped[-1])

    return np.array([X, Y, psi])
