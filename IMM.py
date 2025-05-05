import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kalman import KalmanFilter

class IMM:
    def __init__(self, F_list, H_list, Q_list, R_list, initial_state, measurements):
        """
        Initialize the Interacting Multiple Model (IMM) class.

        Parameters:
        - F_list: List of state transition matrices (one for each Kalman filter).
        - H_list: List of measurement matrices (one for each Kalman filter).
        - Q_list: List of process noise covariance matrices (one for each Kalman filter).
        - R_list: List of measurement noise covariance matrices (one for each Kalman filter).
        - initial_state: Initial state vector (shared across all filters).
        - measurements: List of measurement vectors.
        """
        self.filters = []
        self.measurements = measurements
        self.filtered_history = []
        self.cov = []
        # Create N Kalman filters
        for F, H, Q, R in zip(F_list, H_list, Q_list, R_list):
            P = np.eye(F.shape[0])  # Initialize state covariance as identity matrix
            self.filters.append(KalmanFilter(F, P, H, R, Q, initial_state))
        print("modes ",len(self.filters))
    def run(self):
        """
        Run the IMM algorithm by applying all Kalman filters to the measurements.
        """
        for z in self.measurements:
            filtered_states = []
            cov = []
            for kf in self.filters:
                kf.predict()
                kf.update(z)
                filtered_states.append(kf.x)
                cov.append(np.trace(kf.P))  # Store the trace of the covariance matrix
            self.cov.append(cov)
            self.filtered_history.append(filtered_states)

    def get_filtered_history(self):
        """
        Get the history of filtered measurements.

        Returns:
        - List of filtered states from all Kalman filters at each time step.
        """
        return self.filtered_history
    

    # ...existing code...

    def plot_results(self):
        """
        Plot the measurements as a scatter plot, add line graphs of the filtered states,
        and include subplots for the absolute covariances of each filter.

        Assumes measurements are a time series of (x, y) coordinates.
        """
        measurements = np.array(self.measurements)
        if measurements.shape[1] != 2:
            measurements = measurements[:, :2]  # Ensure we only take the first two columns (x, y)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot of measurements and line graphs of filtered states
        axes[0].scatter(measurements[:, 0], measurements[:, 1], c='blue', label='Measurements', marker='.', alpha=0.6)
        filtered_history = np.array(self.filtered_history)  # Convert to numpy array for easier indexing
        num_filters = len(self.filters)
        for i in range(num_filters):
            filter_states = filtered_history[:, i, :2]  # Extract x, y coordinates for filter i
            axes[0].plot(filter_states[:, 0], filter_states[:, 1], label=f'Filter {i + 1}')
        axes[0].set_title('Scatter Plot of Measurements and Filtered States')
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        axes[0].legend()
        axes[0].grid(True)

        # Subplot for absolute covariances
        for i in range(num_filters):
            axes[1].plot(self.cov[i], label=f'Filter {i + 1}')
        axes[1].set_title('Absolute Covariances of Filters')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Absolute Covariance')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

def get_F_cv(dt):
  """
  Generates the state transition matrix (F) for a 2D Constant Velocity (CV) model.

  Assumes a state vector x = [px, py, vx, vy]^T.

  Args:
    dt: Time step (scalar, in seconds).

  Returns:
    A 4x4 NumPy array representing the CV state transition matrix.
  """
  F = np.array([
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
  ])
  return F

def get_F_ct(dt, omega):
  """
  Generates the state transition matrix (F) for a 2D Constant Turn (CT) model.

  Assumes a state vector x = [px, py, vx, vy]^T.

  Args:
    dt: Time step (scalar, in seconds).
    omega: Turn rate (scalar, in radians per second).

  Returns:
    A 4x4 NumPy array representing the CT state transition matrix.
    Handles the case omega = 0 by returning the CV matrix.
  """
  # Handle the case omega = 0 (or very close to 0) to avoid division by zero
  # and maintain numerical stability. Returns CV matrix in this case.
  if abs(omega) < 1e-8: # Use a small threshold
      return get_F_cv(dt)

  # Calculate terms for omega != 0
  sin_odt = np.sin(omega * dt)
  cos_odt = np.cos(omega * dt)

  # Terms involving division by omega
  term1 = sin_odt / omega
  term2 = (1 - cos_odt) / omega

  F = np.array([
      [1, 0, term1, -term2],
      [0, 1, term2, term1],
      [0, 0, cos_odt, -sin_odt],
      [0, 0, sin_odt, cos_odt]
  ])
  return F