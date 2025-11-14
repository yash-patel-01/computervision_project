import numpy as np

class KalmanFilterCV:
    """Constant velocity Kalman Filter for bbox center (cx, cy), aspect ratio a=w/h, and height h.
    State vector: [cx, cy, a, h, vx, vy, va, vh]
    Measurement: [cx, cy, a, h]
    """
    def __init__(self):
        # State dimension
        self._ndim = 4
        self._dt = 1.0  # assume unit time step between frames
        # Motion model matrices
        self.F = np.eye(8)
        for i in range(self._ndim):
            self.F[i, i + self._ndim] = self._dt
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        # Process / measurement noise (tuned heuristically)
        self.Q = np.eye(8) * 0.01
        self.R = np.eye(4) * 0.1
    def initiate(self, measurement):
        mean = np.zeros(8)
        mean[:4] = measurement
        cov = np.eye(8)
        cov[4:, 4:] *= 10.0  # higher uncertainty on velocities
        return mean, cov
    def predict(self, mean, cov):
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + self.Q
        return mean, cov
    def update(self, mean, cov, measurement):
        # Kalman update
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - (self.H @ mean)
        mean = mean + K @ innovation
        cov = (np.eye(8) - K @ self.H) @ cov
        return mean, cov
    def project(self, mean, cov):
        # Project state distribution to measurement space
        mean_meas = self.H @ mean
        cov_meas = self.H @ cov @ self.H.T + self.R
        return mean_meas, cov_meas
