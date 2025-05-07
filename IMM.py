import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kalman import KalmanFilter
from scipy.stats import multivariate_normal
class IMM:
    def __init__(self, F_list, H_list, Q_list, R_list, initial_state, p_mode, true_mode, measurements, MIXING=True):
        """
        Initialize the Interacting Multiple Model (IMM) class.

        Parameters:
        -----------
        F_list : list of ndarray
            List of state transition matrices (one for each Kalman filter)
        H_list : list of ndarray
            List of measurement matrices (one for each Kalman filter)
        Q_list : list of ndarray
            List of process noise covariance matrices (one for each Kalman filter)
        R_list : list of ndarray
            List of measurement noise covariance matrices (one for each Kalman filter)
        initial_state : ndarray
            Initial state vector (shared across all filters)
        p_mode : float
            Probability of mode transition
        true_mode : list
            List of true modes for each time step
        measurements : list of ndarray
            List of measurement vectors
        MIXING : bool, optional
            Whether to perform mode mixing, defaults to True
        """
        self.MIXING = MIXING
        self.measurements = measurements
        self.state_dim = F_list[0].shape[0]
        self.num_filters = len(F_list)
        self.true_mode = true_mode
        
        # Initialize filters
        self.filters = self._initialize_filters(F_list, H_list, Q_list, R_list, initial_state)
        
        # Initialize mode probabilities and transition matrix
        self.mu = np.ones(self.num_filters) / self.num_filters
        self.P = self._initialize_transition_matrix(p_mode)
        
        # Initialize tracking variables
        self.mu_history = []
        self.best_estimate = np.zeros((len(self.measurements), self.state_dim))

    def _initialize_filters(self, F_list, H_list, Q_list, R_list, initial_state):
        """Initialize Kalman filters with given parameters."""
        filters = []
        for F, H, Q, R in zip(F_list, H_list, Q_list, R_list):
            P = np.eye(F.shape[0])
            filters.append(KalmanFilter(F, P, H, R, Q, np.array(initial_state)))
        return filters

    def _initialize_transition_matrix(self, p_mode):
        """
        Initialize the mode transition probability matrix.
        
        The transition matrix P[i,j] represents the probability of transitioning
        from mode i to mode j. For each mode:
        - Self-transition probability is (1 - p_mode)
        - Transition to other modes is p_mode / (num_filters - 1)
        

        """
        # Initialize matrix with equal transition probabilities to other modes
        P = np.full((self.num_filters, self.num_filters), 
                    p_mode / (self.num_filters - 1))
        
        # Set self-transition probabilities along diagonal
        np.fill_diagonal(P, 1 - p_mode)
        
        return P

    def _calculate_mixing_probabilities(self):
        """Calculate mixing probabilities for mode interaction."""
        Z = self.P @ self.mu
        omega = np.zeros((self.num_filters, self.num_filters))
        for i in range(self.num_filters):
            for j in range(self.num_filters):
                omega[i, j] = (self.P[i, j] * self.mu[i]) / Z[j]
        return omega

    def _mix_states_and_covariances(self, omega):
        """Mix states and covariances based on mixing probabilities."""
        for m in range(self.num_filters):
            x = np.zeros((self.state_dim, 1))
            P = np.zeros((self.state_dim, self.state_dim))
            
            # Mix states
            for i in range(self.num_filters):
                x += omega[i, m] * np.array(self.filters[i].x).reshape(-1, 1)
            
            # Mix covariances
            for i in range(self.num_filters):
                x_diff = np.array(self.filters[i].x).reshape(-1, 1) - x
                P += omega[i, m] * (self.filters[i].P + np.outer(x_diff, x_diff))
            
            self.filters[m].x = x
            self.filters[m].P = P

    def _update_mode_probabilities(self, likelihoods):
        """Update mode probabilities based on measurement likelihoods."""
        Z = self.P @ self.mu
        num = np.multiply(Z, likelihoods)
        denom = np.sum(Z * likelihoods)
        self.mu = num / denom
        self.mu_history.append(self.mu.copy())

    def run(self):
        """Run the IMM algorithm by applying all Kalman filters to the measurements."""
        for idx, z in enumerate(self.measurements):
            # Mode mixing step
            if self.MIXING:
                omega = self._calculate_mixing_probabilities()
                self._mix_states_and_covariances(omega)
            
            # Filter prediction and update
            likelihoods = np.zeros(self.num_filters)
            for j in range(self.num_filters):
                self.filters[j].predict()
                self.filters[j].update(np.array(z))
                self.filters[j].save_state_cov()
                # Calculate measurement likelihood
                cov = self.filters[j].S
                mean = np.array(self.filters[j].y).flatten()
                likelihoods[j] = multivariate_normal.pdf(mean, cov=cov, allow_singular=True)
            
            # Update mode probabilities
            self._update_mode_probabilities(likelihoods)
            
            # Update best estimate
            self.best_estimate[idx] = sum(self.mu[i] * np.array(self.filters[i].x).flatten() 
                                        for i in range(self.num_filters))

    def get_filtered_history(self):
        """
        Get the history of filtered measurements.

        Returns:
        - List of filtered states from all Kalman filters at each time step.
        """
        return self.filtered_history
    

    def plot_true_vs_estimated_mode(self, true_mode):
        """
        Plot the true mode versus the estimated mode with the highest probability as a line graph.

        Parameters:
        - true_mode: List of true modes at each time step.
        """
        estimated_modes = np.argmax(self.mu_history, axis=1)  # Get the index of the highest probability mode at each step
        time_steps = range(len(true_mode))  # Time steps

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, true_mode, label='True Mode', linestyle='-', alpha=0.7)
        plt.plot(time_steps, estimated_modes, label='Estimated Mode', linestyle='--', alpha=0.7)
        plt.title('True Mode vs Estimated Mode')
        plt.xlabel('Time Step')
        plt.ylabel('Mode')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


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
        # for i in range(num_filters):
        #     filter_states = self.filters[i].get_state_his()[:, :2]  # Get the first two dimensions (x, y) of the state history
        #     axes[0].plot(filter_states[:, 0], filter_states[:, 1], label=f'Filter {i + 1}')
        axes[0].plot(self.best_estimate[:, 0], self.best_estimate[:, 1], label=f'best estimate')
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

        self.plot_true_vs_estimated_mode(self.true_mode)

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