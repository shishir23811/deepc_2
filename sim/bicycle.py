import numpy as np

class KinematicBicycle:
    '''
     Discrete-time kinematic bicycle simulator.

     parameters:
        wheel_base: float
            Vehicle's wheelbase L (in meters)
            default: 0.3m
        
        dt: float
            Time step in seconds.
            default: 0.05 sec or 1 / 0.05 = 20 Hz
        
        noise_std: dict
            Output noise standard deviation(noise is static)
            default: None
    '''
    def __init__(self, wheel_base= 0.3, dt= 0.05, noise_std= None):
        self.wheel_base = wheel_base
        self.dt = dt
        self.noise_std = noise_std if noise_std is not None else {'X': 0.0, 'Y': 0.0, 'psi': 0.0}

        self.reset() #resets state
    
    def reset(self, x0 = None):
        self.state = np.array(x0 if x0 is not None else [0.0, 0.0, 0.0, 1.0]) # [X, Y, psi(yaw), v]
        return self._noisy_output()
    
    def _noisy_output(self):
        X, Y, psi, v = self.state

        return np.array([
            X + np.random.normal(0.0, self.noise_std.get('X', 0.0)),
            Y + np.random.normal(0.0, self.noise_std.get('Y', 0.0)),
            psi + np.random.normal(0.0, self.noise_std.get('psi', 0.0))
        ])
    
    def step(self, u):
        delta = np.clip(u[0], -np.deg2rad(25), np.deg2rad(25))
        a = np.clip(u[1], -2.0, 2.0)

        X, Y, psi, v = self.state

        X_n = X + v * np.cos(psi) * self.dt
        Y_n = Y + v * np.sin(psi) * self.dt
        psi_n = psi + (v / self.wheel_base) * np.tan(delta) * self.dt
        v_n = max(0.0, v + a * self.dt)

        self.state = np.array([X_n, Y_n, psi_n, v_n ])

        return self._noisy_output()