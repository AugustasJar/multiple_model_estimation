import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kalman import KalmanFilter
from scipy.stats import multivariate_normal
class IMM:
    def __init__(self, F_list, H_list, Q_list, R_list, initial_state,p_mode, measurements, MIXING=True):
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
        self.MIXING = MIXING
        self.filters = []
        self.measurements = measurements
        self.state_dim = F_list[0].shape[0]  # State dimension (assumed to be the same for all filters)
        self.num_filters = len(F_list)  # Number of Kalman filters
        self.mu = np.zeros((self.num_filters,1))
        self.P = np.zeros((self.state_dim, self.state_dim))  # Transition probability matrix
        self.mu_history = []
        # Create N Kalman filters
        for F, H, Q, R in zip(F_list, H_list, Q_list, R_list):
            P = np.eye(F.shape[0])  # Initialize state covariance as identity matrix
            self.filters.append(KalmanFilter(F, P, H, R, Q, np.array(initial_state)))

        
        # initialize mode probabilities
        mu = [1/len(self.filters)] * len(self.filters)  # Equal probability for each mode
        self.mu = np.array(mu)
        # initialize transition probabilities
        P = np.zeros((len(self.filters), len(self.filters)))
        for i in range(len(self.filters)):
            for j in range(len(self.filters)):
                if i == j:
                    P[i][j] = 1-p_mode
                else:
                    P[i][j] = p_mode / (len(self.filters) - 1)
        self.P = np.array(P)

    def run(self):
        """
        Run the IMM algorithm by applying all Kalman filters to the measurements.
        """

        for z in self.measurements:
            
            
            # predict model probabilities
            if self.MIXING:
                Z = self.P @ self.mu
                omega = np.zeros((self.num_filters, self.num_filters))

                #calculated mixing coefficient
                for i in range(self.num_filters):
                    for j in range(self.num_filters):
                        omega[i, j] = (self.P[i, j] * self.mu[i]) / Z[j]

                # mix model states and covariances
                for m in range(self.num_filters):
                    x = np.zeros((4,1))
                    P = np.zeros((4,4))
                    #mix the state
                    for i in range(self.num_filters):
                        x = omega[i, m] * self.filters[i].x
                    #mix the covariance
                    for i in range(self.num_filters):
                        x_diff = self.filters[i].x - x
                        P += omega[i, j] * (self.filters[i].P + np.outer(x_diff, x_diff))
                    #update the filter with the mixed state and covariance
                    self.filters[m].x = x
                    self.filters[m].P = P
                    
            likelihoods = np.zeros((self.num_filters,1))  # Initialize likelihoods for each filter
            for j in range(self.num_filters):
                self.filters[j].predict()
                self.filters[j].update(z)
                likelihoods[j] = multivariate_normal.pdf(self.filters[j].y.flatten(),cov=self.filters[j].S,allow_singular=True) # Set allow_singular based on needs
                likelihoods[j] = max(likelihoods[j], 1e-9) # Floor likelihood

            num = np.multiply(Z ,likelihoods.T).flatten()
            denom = np.sum(Z * likelihoods)
            self.mu = (num) / denom
            self.mu_history.append(self.mu.copy())
            #update states and covariances
            for i in range(self.num_filters):
                self.filters[i].x = self.filters[i].x 
                self.filters[i].P = self.filters[i].P 

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
        num_filters = len(self.filters)
        for i in range(num_filters):
            filter_states = self.filters[i].get_state_his()[:, :2]  # Get the first two dimensions (x, y) of the state history
            axes[0].plot(filter_states[:, 0], filter_states[:, 1], label=f'Filter {i + 1}')
        axes[0].set_title('Scatter Plot of Measurements and Filtered States')
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        axes[0].legend()
        axes[0].grid(True)

        # Subplot for absolute covariances
        for i in range(num_filters):
            cov_trace = [np.trace(cov) for cov in self.filters[i].cov_history]
            axes[1].plot(cov_trace, label=f'Filter {i + 1}')
        axes[1].set_title('Absolute Covariances of Filters')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Absolute Covariance')
        axes[1].legend()
        axes[1].grid(True)


        mu_history = np.array(self.mu_history)  # Convert to NumPy array for easier manipulation
        time_steps = range(len(mu_history))  # Time steps
        plt.figure(figsize=(10, 6))
        for i in range(mu_history.shape[1]):
            plt.plot(time_steps, mu_history[:, i], label=f'Mode {i + 1}')
            
        plt.title('Mode Probabilities Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Mode Probability')
        plt.legend()
        plt.grid(True)
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