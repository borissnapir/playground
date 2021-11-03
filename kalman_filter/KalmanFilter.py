import numpy as np

class KalmanFilter():
    """
    Assumption: dynamic model F and measurement model H are not constant over time.
    """
    def __init__(self, x0, P0, F, Q, R, H, B):
        # initialize state vector
        self.x = x0
        # initialize state cov matrix
        self.P = P0
        # dynamic model
        self.F = F
        # model uncertainty
        self.Q = Q

        # measurements
        # z
        # measurement cov matrix
        self.R = R
        # measurement model
        self.H = H

        # external command
        # u
        # command model
        self.B = B

    def predict(self, u=np.array([0])):
        self.x = self.F @ self.x + self.B @ u
        self.P = (self.F @ self.P) @ self.F.T + self.Q
        return self.x


    def update(self, z):
        # compute Kalman gain
        S = self.R + self.H @ (self.P @ self.H.T)
        K = (self.P @ self.H.T) @ np.linalg.inv(S)
        # update state vector
        self.x = self.x + K @ (z - self.H @ self.x)
        # update state cov matrix
        self.P = self.P - K @ (self.H @ self.P)
        return self.x