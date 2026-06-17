import numpy as np

def figure_eight(t, A=5.0, omega=0.2):
    '''
    Analytical figure-8 reference path.
    Return: [X, Y, psi]
    '''

    X = A * np.sin(omega * t)
    Y = A * np.sin(2 * omega * t)

    t_val = float(t)
    if t_val <= 0.0:
        dX = A * omega * np.cos(0.0)
        dY = 2.0 * A * omega * np.cos(0.0)
        psi = np.arctan2(dY, dX)
        return np.array([X, Y, psi])

    n = max(int(t_val * 100) + 2, 4)
    t_grid = np.linspace(0.0, t_val, n)
    dX_grid = A * omega * np.cos(omega * t_grid)
    dY_grid = 2.0 * A * omega * np.cos(2.0 * omega * t_grid)
    psi_grid = np.unwrap(np.arctan2(dY_grid, dX_grid))
    psi = float(psi_grid[-1])

    return np.array([X, Y, psi])