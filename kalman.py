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
        self.y = None  # Innovation
        self.S = None  # Innovation covariance

    def copy(self):
        """Shallow copy of the KalmanFilter object."""
        new_filter = KalmanFilter(
            F=self.F.copy(),
            P=self.P.copy(),
            H=self.H.copy(),
            R=self.R.copy(),
            Q=self.Q.copy(),
            initial_state=self.x.copy()
        )
        if self.y is not None:
            new_filter.y = self.y.copy()
        if self.S is not None:
            new_filter.S = self.S.copy()
        return new_filter

    def __deepcopy__(self, memo):
        """Deep copy of the KalmanFilter object."""
        import copy
        new_filter = KalmanFilter(
            F=copy.deepcopy(self.F, memo),
            P=copy.deepcopy(self.P, memo),
            H=copy.deepcopy(self.H, memo),
            R=copy.deepcopy(self.R, memo),
            Q=copy.deepcopy(self.Q, memo),
            x=copy.deepcopy(self.x, memo)
        )
        if self.y is not None:
            new_filter.y = copy.deepcopy(self.y, memo)
        if self.S is not None:
            new_filter.S = copy.deepcopy(self.S, memo)
        return new_filter

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
        z = z.reshape(self.H.shape[0], 1)  # Reshape z to (4, 1) if it's not already
        self.x = self.x.reshape(self.H.shape[0], 1)
        # Compute the Kalman Gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        self.S = S
        # Update the state estimate
        y = z - self.H @ self.x  # Measurement residual
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