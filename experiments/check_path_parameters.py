import numpy as np
from math import pi

params = [
    (10.0, 0.2),
    (10.0, 0.1),
    (10.0, 0.05),
    (5.0, 0.1),
    (5.0, 0.05),
    (3.0, 0.05),
]

print('A,omega,max_kappa,min_radius,max_v')
for A, omega in params:
    t = np.linspace(0, 2 * pi / omega, 10000)
    dx = A * omega * np.cos(omega * t)
    dy = 2.0 * A * omega * np.cos(2.0 * omega * t)
    ddx = -A * omega ** 2 * np.sin(omega * t)
    ddy = -4.0 * A * omega ** 2 * np.sin(2.0 * omega * t)
    v = np.hypot(dx, dy)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(v ** 3, 1e-9)
    print(A, omega, np.max(kappa), 1.0 / np.max(kappa), np.max(v))
