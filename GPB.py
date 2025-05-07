# import numpy as np
# from IMM import IMM
# from scipy.stats import multivariate_normal

# class GPB(IMM):
#     def __init__(self, F_list, H_list, Q_list, R_list, initial_state, p_mode, true_mode, measurements, order=2):
#         """
#         Initialize the Generalized Pseudo-Bayesian (GPB) class.

#         Parameters:
#         -----------
#         F_list : list of ndarray
#             List of state transition matrices (one for each Kalman filter)
#         H_list : list of ndarray
#             List of measurement matrices (one for each Kalman filter)
#         Q_list : list of ndarray
#             List of process noise covariance matrices (one for each Kalman filter)
#         R_list : list of ndarray
#             List of measurement noise covariance matrices (one for each Kalman filter)
#         initial_state : ndarray
#             Initial state vector (shared across all filters)
#         p_mode : float
#             Probability of mode transition
#         true_mode : list
#             List of true modes for each time step
#         measurements : list of ndarray
#             List of measurement vectors
#         order : int, optional
#             Order of the GPB algorithm (1 or 2), defaults to 2
#         """
#         super().__init__(F_list, H_list, Q_list, R_list, initial_state, p_mode, true_mode, measurements, MIXING=False)
#         self.order = order
#         if order not in [1, 2]:
#             raise ValueError("GPB order must be either 1 or 2")
        
#         # Initialize mode history for GPB2
#         if order == 2:
#             self.mode_history = []
#             self.mode_history.append(np.ones(self.num_filters) / self.num_filters)


#     def run(self):
#         """Run the GPB algorithm by applying all Kalman filters to the measurements."""
#         for idx, z in enumerate(self.measurements):
#             # Filter prediction and update
#             likelihoods = np.zeros((self.num_filters,self.num_filters))
#             states = []
#             covariances = []

#             # for each model pair calculate the joint model probability
#             joint_model_probabilities = self.P * self.mu[np.newaxis,:]
            
#             posterior_probabilities = self.mu * joint_model_probabilities

#             Z = np.sum(posterior_probabilities)

#             posterior_probabilities = posterior_probabilities / Z


#             # save states and covariances
#             for i in range(self.num_filters):
#                 states.append(self.filters[i].x)
#                 covariances.append(self.filters[i].P)
            
#             # run prediction and update for all states and covariances for all filters
#             expanded_states = np.zeros((self.num_filters, self.num_filters, self.state_dim))
#             expanded_covariances = np.zeros((self.num_filters, self.num_filters, self.state_dim, self.state_dim))
#             for i in range(self.num_filters):
#                 for j in range(self.num_filters):
#                     self.filters[i].x = np.array(states[j]).reshape(-1,1)
#                     self.filters[i].P = np.array(covariances[j])
#                     self.filters[i].predict()
#                     self.filters[i].update(np.array(z))
#                     expanded_states[i,j] = np.array(self.filters[i].x).flatten()
#                     expanded_covariances[i,j,:,:] = self.filters[i].P

#                     cov = self.filters[i].S
#                     mean = np.array(self.filters[i].y).flatten()
#                     likelihoods[i,j] = multivariate_normal.pdf(mean, cov=cov, allow_singular=True)
                    

#             normalized_likelihoods = likelihoods / np.sum(likelihoods, axis=1, keepdims=True)

#             # merge likelihoods
#             merged_likelihoods = np.sum(normalized_likelihoods, axis=1)

#             # 2. Calculate Mixing Weights: W_ij = P(m_{k-1}^j | m_k^i, Z_k)
#             # W_ij = P(m_k^i, m_{k-1}^j | Z_k) / P(m_k^i | Z_k)
#             mixing_weights = np.zeros_like(likelihoods)
#             for i in range(self.num_filters):
#                 mixing_weights[i, :] = normalized_likelihoods[i, :] / merged_likelihoods[i]

#             states = expanded_states.reshape(self.num_filters, self.num_filters, self.state_dim)
#             covariances = expanded_covariances.reshape(self.num_filters, self.num_filters, self.state_dim, self.state_dim)
#             merged_state_estimates = np.einsum('ij,ijk->ik', mixing_weights, states)
#             updated_covariances = np.zeros((self.num_filters, self.state_dim, self.state_dim))
#             # 4. Merge Covariances: P_k|k^i
#             for i in range(self.num_filters):
#                 current_merged_cov_i = np.zeros((self.state_dim, self.state_dim))
#                 for j in range(self.num_filters):
#                     if mixing_weights[i, j] > 1e-12:
#                         s_hat_ij = expanded_states[i, j, :]
#                         P_ij = expanded_covariances[i, j, :, :]
#                         s_hat_i = merged_state_estimates[i, :]

#                         diff = s_hat_ij - s_hat_i
#                         # Ensure diff is 1D array for np.outer if state_dim is 1
                        
#                         spread_term = np.outer(diff, diff)
#                         current_merged_cov_i += mixing_weights[i, j] * (P_ij + spread_term)
#                 updated_covariances[i, :, :] = current_merged_cov_i

#             for i in range(self.num_filters):
#                 self.filters[i].P = updated_covariances[i, :, :]
#                 self.filters[i].x = merged_state_estimates[i, :]
                
#             self.mu = posterior_probabilities
#             best_estimate = np.zeros((self.state_dim,))
#             for i in range(self.num_filters):
#                 best_estimate += self.filters[i].x * posterior_probabilities[i]

            
#             self.best_estimate[idx] = best_estimate

import numpy as np
from IMM import IMM # Assuming IMM and its KalmanFilter components are correctly implemented
from scipy.stats import multivariate_normal

class GPB(IMM):
    def __init__(self, F_list, H_list, Q_list, R_list, initial_state, p_mode, true_mode, measurements, order=2):
        super().__init__(F_list, H_list, Q_list, R_list, initial_state, p_mode, true_mode, measurements, MIXING=False)
        self.order = order
        if order not in [1, 2]:
            raise ValueError("GPB order must be either 1 or 2")
        
        # Initialize mode history for GPB2
        if order == 2: # Current implementation is for GPB2
            self.mode_history = []
            # Initial mode probabilities are self.mu, e.g. uniform (1/self.num_filters)
            # self.mode_history.append(self.mu.copy()) # Store initial mode probabilities

    def run(self):
        """Run the GPB algorithm by applying all Kalman filters to the measurements."""
        for idx, z_measurement in enumerate(self.measurements):
            # Ensure z_measurement is a numpy array for consistency
            z = np.array(z_measurement)

            # --- GPB2 Steps ---
            # Based on PDF "SSDP11-JointEstimationModelSelection.pdf" pages 3-4 [cite: 41, 42, 43, 44]

            # 1. Calculate P(m_k, m_{k-1} | Z_{k-1})
            # self.P[i,j] is P(m_k=i | m_{k-1}=j)
            # self.mu[j] is P(m_{k-1}=j | Z_{k-1})
            # P_mk_mk_minus_1_cond_Z_k_minus_1[i,j] = P(m_k=i, m_{k-1}=j | Z_{k-1})
            P_mk_mk_minus_1_cond_Z_k_minus_1 = self.P * self.mu[np.newaxis, :]

            likelihoods = np.zeros((self.num_filters, self.num_filters))
            expanded_states = np.zeros((self.num_filters, self.num_filters, self.state_dim))
            expanded_covariances = np.zeros((self.num_filters, self.num_filters, self.state_dim, self.state_dim))

            # Store previous filter states to avoid modification during expansion
            prev_filter_states = [kf.x.copy() for kf in self.filters]
            prev_filter_covs = [kf.P.copy() for kf in self.filters]

            # 2. Expansion and Filtering: For each pair (m_k=i, m_{k-1}=j)
            for i in range(self.num_filters):  # Current mode m_k = i
                for j in range(self.num_filters):  # Previous mode m_{k-1} = j
                    # Set initial state for filter i based on previous state of filter j
                    self.filters[i].x = prev_filter_states[j].reshape(-1, 1)
                    self.filters[i].P = prev_filter_covs[j]
                    
                    self.filters[i].predict()
                    self.filters[i].update(z) # Pass the current measurement z
                    
                    expanded_states[i, j, :] = self.filters[i].x.flatten()
                    expanded_covariances[i, j, :, :] = self.filters[i].P
                    
                    # Likelihood p(z_k | m_k=i, m_{k-1}=j-derived state, Z_{k-1})
                    # self.filters[i].y should be the innovation (measurement residual)
                    # self.filters[i].S should be the innovation covariance
                    innovation = self.filters[i].y.flatten()
                    innovation_cov = self.filters[i].S
                    likelihoods[i, j] = multivariate_normal.pdf(innovation, mean=np.zeros_like(innovation), cov=innovation_cov, allow_singular=True)
            
            # 3. Calculate Joint Posterior Mode Probabilities P(m_k, m_{k-1} | Z_k)
            P_mk_mk_minus_1_cond_Zk_unnormalized = likelihoods * P_mk_mk_minus_1_cond_Z_k_minus_1
            sum_total_prob = np.sum(P_mk_mk_minus_1_cond_Zk_unnormalized)
            if sum_total_prob < 1e-15 : # Avoid division by zero or near-zero
                 # This case implies all likelihoods are zero or P_mk_mk_minus_1_cond_Z_k_minus_1 made them zero.
                 # Fallback: uniform distribution over joint modes or handle as error.
                 # For now, if sum is zero, all joint_posterior_mode_probs will be zero.
                 joint_posterior_mode_probs = np.zeros_like(P_mk_mk_minus_1_cond_Zk_unnormalized)

            else:
                joint_posterior_mode_probs = P_mk_mk_minus_1_cond_Zk_unnormalized / sum_total_prob
            
            # 4. Calculate Marginal Posterior Mode Probabilities P(m_k | Z_k)
            current_mode_probs_P_mk_cond_Zk = np.sum(joint_posterior_mode_probs, axis=1) # Sum over m_{k-1} (axis 1)
            
            # Normalize P(m_k | Z_k) to ensure it sums to 1 (robustness)
            sum_current_mode_probs = np.sum(current_mode_probs_P_mk_cond_Zk)
            if sum_current_mode_probs < 1e-15:
                self.mu = np.ones(self.num_filters) / self.num_filters # Fallback to uniform if all probabilities are zero
            else:
                self.mu = current_mode_probs_P_mk_cond_Zk / sum_current_mode_probs
            
            # 5. Calculate Mixing Weights P(m_{k-1} | m_k, Z_k)
            mixing_weights = np.zeros_like(joint_posterior_mode_probs) # mixing_weights[i,j] = P(m_{k-1}=j | m_k=i, Z_k)
            for i in range(self.num_filters):
                if self.mu[i] > 1e-12: # Check P(m_k=i | Z_k)
                    mixing_weights[i, :] = joint_posterior_mode_probs[i, :] / self.mu[i]
                else:
                    # If P(m_k=i | Z_k) is zero, P(m_k=i, m_{k-1}=j | Z_k) must also be zero for all j.
                    # So, weights are zero. A uniform distribution could be a fallback.
                    mixing_weights[i, :] = 0.0 # Or 1.0 / self.num_filters for robustness
            
            # Normalize mixing weights for each m_k=i (robustness, should ideally sum to 1 already)
            for i in range(self.num_filters):
                sum_mixing_weights_i = np.sum(mixing_weights[i, :])
                if sum_mixing_weights_i > 1e-12:
                    mixing_weights[i, :] /= sum_mixing_weights_i
                elif self.mu[i] > 1e-12: # If P(m_k=i|Z_k) was non-zero, but weights summed to zero (should not happen if joint_posterior handled correctly)
                    mixing_weights[i, :] = 1.0 / self.num_filters # Fallback to uniform


            # 6. Merge State Estimates and Covariances
            # Merged state for mode i: s_k|k^i = sum_j { P(m_{k-1}=j | m_k=i, Z_k) * s_k|k^{i,j} }
            merged_state_estimates_x = np.einsum('ij,ijk->ik', mixing_weights, expanded_states)
            
            # Merged covariance for mode i: P_k|k^i (see PDF pg. 8 or standard IMM/GPB texts)
            merged_covariances_P = np.zeros((self.num_filters, self.state_dim, self.state_dim))
            for i in range(self.num_filters): # For each current mode m_k=i
                current_merged_cov_i = np.zeros((self.state_dim, self.state_dim))
                for j in range(self.num_filters): # Sum over previous modes m_{k-1}=j
                    if mixing_weights[i, j] > 1e-12: # Process only if weight is non-negligible
                        s_hat_ij = expanded_states[i, j, :] # s_k|k^{i,j}
                        P_ij = expanded_covariances[i, j, :, :] # P_k|k^{i,j}
                        s_hat_i = merged_state_estimates_x[i, :] # s_k|k^i (merged for current mode i)
                        
                        diff = s_hat_ij - s_hat_i
                        spread_term = np.outer(diff, diff)
                        current_merged_cov_i += mixing_weights[i, j] * (P_ij + spread_term)
                merged_covariances_P[i, :, :] = current_merged_cov_i

            # 7. Update individual filter states with merged estimates
            for i in range(self.num_filters):
                self.filters[i].x = merged_state_estimates_x[i, :].reshape(-1, 1)
                self.filters[i].P = merged_covariances_P[i, :, :]
                self.filters[i].save_state_cov()
                
            # 8. Calculate Overall Best Estimate (Optional, but often desired)
            # best_estimate = sum_i { P(m_k=i | Z_k) * s_k|k^i }
            current_best_estimate = np.zeros((self.state_dim,))
            for i in range(self.num_filters):
                current_best_estimate += self.filters[i].x.flatten() * self.mu[i]
            
            self.best_estimate[idx] = current_best_estimate
            self.mu_history.append(self.mu.copy())

            # Store mode probabilities if order == 2 (for analysis)
            # if self.order == 2:
            #    self.mode_history.append(self.mu.copy())