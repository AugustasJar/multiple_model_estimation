import numpy as np

class KalmanFilter:
    def __init__(self, F, P, H, R, Q, initial_state):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        - F: State transition matrix.
        - P: Initial state covariance matrix.
        - H: Measurement matrix.
        - R: Measurement noise covariance matrix.
        - Q: Process noise covariance matrix.
        - initial_state: Initial state vector.
        """
        self.F = F
        self.P = P
        self.H = H
        self.R = R
        self.Q = Q
        self.x = initial_state

    def predict(self):
        """
        Predict the next state and covariance.
        """
        # Predict the state
        self.x = np.dot(self.F, self.x)

        # Predict the state covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        Update the state with a new measurement.
        
        Parameters:
        - z: Measurement vector.
        """
        # Compute the Kalman Gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  

        # Update the state estimate
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        # Update the state covariance
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
    