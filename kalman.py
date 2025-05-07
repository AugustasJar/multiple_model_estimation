import numpy as np
import matplotlib.pyplot as plt

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
        self.F = F  # Ensure F is a numpy array
        self.P = P
        self.H = H
        self.R = R
        self.Q = Q
        self.x = initial_state
        self.state_history = []  # Store state history for plotting
        self.cov_history = []    # Store covariance history for plotting
        self.y = np.zeros((4, 1))
        self.S = np.zeros((self.H.shape[0], self.H.shape[0]))

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
        z = z.reshape(-1, 1)  # Reshape z to (4, 1) if it's not already
        # Compute the Kalman Gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        self.S = S
        # Update the state estimate
        y = z - np.dot(self.H, self.x)  # Measurement residual
        self.x = self.x + np.dot(K, y)
        self.y = y
        # Update the state covariance
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        

    def save_state_cov(self):
        self.state_history.append(self.x.copy())
        self.cov_history.append(self.P.copy())


    def get_state_his(self):
        return np.array(self.state_history.copy())
    
    def plot(self):
        positions = self.state_history
        covariance = self.cov_history
        positions = np.array(positions)[:, :2]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(positions[:, 0], positions[:, 1], label="Predicted Trajectory")
        plt.title("Predicted Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()

        # Plot covariance (trace of covariance matrix as a measure of uncertainty)
        cov_trace = [np.trace(cov) for cov in covariance]
        plt.subplot(1, 2, 2)
        plt.plot(range(len(cov_trace)), cov_trace, label="Covariance Trace")
        plt.title("Covariance Trace Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Trace of Covariance")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()