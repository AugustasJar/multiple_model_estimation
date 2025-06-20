import numpy as np
from IMM import IMM
from scipy.stats import multivariate_normal

class GPB(IMM):
    def __init__(self, F_list, H_list, Q_list, R_list, initial_state, p_mode, true_mode,true_trajectory, measurements, order=2):
        """
        Initialize the Generalized Pseudo-Bayesian (GPB) class.

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
        order : int, optional
            Order of the GPB algorithm (1 or 2), defaults to 2
        """
        super().__init__(F_list, H_list, Q_list, R_list, initial_state, p_mode, true_mode,true_trajectory, measurements, MIXING=False)
        self.order = order
        if order not in [1, 2]:
            raise ValueError("GPB order must be either 1 or 2")
        
        # Initialize mode history for GPB2
        if order == 2:
            self.mode_history = []
            self.mode_history.append(np.ones(self.num_filters) / self.num_filters)


    def run(self):
        """Run the GPB algorithm by applying all Kalman filters to the measurements."""
        for idx, z in enumerate(self.measurements):
            # Filter prediction and update
            likelihoods = np.zeros((self.num_filters,self.num_filters))
            states = []
            covariances = []

            # for each model pair calculate the joint model probability
            likelihoods = np.zeros((self.num_filters, self.num_filters))
            expanded_states = np.zeros((self.num_filters, self.num_filters, self.state_dim))
            expanded_covariances = np.zeros((self.num_filters, self.num_filters, self.state_dim, self.state_dim))

            # Store previous filter states to avoid modification during expansion
            prev_filter_states = [kf.x.copy() for kf in self.filters]
            prev_filter_covs = [kf.P.copy() for kf in self.filters]
            
            # run prediction and update for all states and covariances for all filters
            expanded_states = np.zeros((self.num_filters, self.num_filters, self.state_dim))
            expanded_covariances = np.zeros((self.num_filters, self.num_filters, self.state_dim, self.state_dim))
            for i in range(self.num_filters):
                for j in range(self.num_filters):
                    # Set initial state for filter i based on previous state of filter j
                    self.filters[i].x = prev_filter_states[j].reshape(-1, 1)
                    self.filters[i].P = prev_filter_covs[j]

                    self.filters[i].predict()
                    self.filters[i].update(np.array(z))

                    expanded_states[i,j] = np.array(self.filters[i].x).flatten()
                    expanded_covariances[i,j,:,:] = self.filters[i].P

                    innovation = self.filters[i].y.flatten()
                    innovation_cov = self.filters[i].S
                    likelihoods[i, j] = multivariate_normal.pdf(innovation, mean=np.zeros_like(innovation), cov=innovation_cov, allow_singular=True)
                    
            # find the conditional likelihood based on the state transition and previous probability density
            # mk,mk-1 | Zk-1
            P_cond = self.P * self.mu[np.newaxis, :]

            #find the joint posterior mode probabilities mk,mk-1 | Zk
            posterior_mode_probs = likelihoods * P_cond

            #find the evidence
            evidence = np.sum(posterior_mode_probs)

            posterior_mode_probs_normalized = posterior_mode_probs / evidence

            #find the marginal posterior mode probabilities mk | Zk
            marginal_posterior = np.sum(posterior_mode_probs_normalized, axis=1)

            #normalization
            sum_marginal = np.sum(marginal_posterior)
            self.mu = marginal_posterior / sum_marginal

            #find the mixing weights
            mixing_weights = np.zeros_like(posterior_mode_probs_normalized)
            for i in range(self.num_filters):
                mixing_weights[i, :] = posterior_mode_probs_normalized[i, :] / self.mu[i]

            #normalization
            for i in range(self.num_filters):
                sum_mixing_weights_i = np.sum(mixing_weights[i, :])
                if sum_mixing_weights_i > 1e-12:
                    mixing_weights[i, :] /= sum_mixing_weights_i


            merged_state_estimates = np.einsum('ij,ijk->ik', mixing_weights, expanded_states)
            updated_covariances = np.zeros((self.num_filters, self.state_dim, self.state_dim))

            # Merge Covariances: P_k|k^i
            for i in range(self.num_filters):
                current_merged_cov_i = np.zeros((self.state_dim, self.state_dim))
                for j in range(self.num_filters):
                    if mixing_weights[i, j] > 1e-12:
                        
                        s_hat_ij = expanded_states[i, j, :]
                        P_ij = expanded_covariances[i, j, :, :]
                        s_hat_i = merged_state_estimates[i, :]

                        diff = s_hat_ij - s_hat_i
                        spread_term = np.outer(diff, diff)
                        current_merged_cov_i += mixing_weights[i, j] * (P_ij + spread_term)
                updated_covariances[i, :, :] = current_merged_cov_i


            #update the filter states
            for i in range(self.num_filters):
                self.filters[i].P = updated_covariances[i, :, :]
                self.filters[i].x = merged_state_estimates[i, :]
                self.filters[i].save_state_cov()
                
            current_best_estimate = np.zeros((self.state_dim,))    
            for i in range(self.num_filters):
                current_best_estimate += self.filters[i].x.flatten() * self.mu[i]
            
            self.best_estimate[idx] = current_best_estimate
            self.mu_history.append(self.mu.copy())
            self.predicted_modes.append(np.argmax(self.mu))

