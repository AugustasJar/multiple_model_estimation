import numpy as np
from IMM import IMM
from scipy.stats import multivariate_normal
from kalman import KalmanFilter

class M3H:
    def __init__(self, F_list, H_list, Q_list, R_list, initial_state,
                 P_transition, measurements, epsilon, L_merge, l_max,
                 initial_mode_probabilities=None, true_trajectory=None, true_mode=None):
        """
        Initialize the Multiple Model Multiple Hypothesis (M3H) Filter.

        Parameters:
        -----------
        F_list : list of ndarray
            List of state transition matrices (one for each mode).
        H_list : list of ndarray
            List of measurement matrices (one for each mode).
        Q_list : list of ndarray
            List of process noise covariance matrices (one for each mode).
        R_list : list of ndarray
            List of measurement noise covariance matrices (one for each mode).
        initial_state : ndarray
            Initial state vector.
        p_transition : ndarray
            Mode transition probability matrix P_ij = P(mode(k+1)=j | mode(k)=i).
        measurements : list of ndarray
            List of measurement vectors.
        epsilon : float
            Pruning threshold for normalized hypothesis likelihood.
        L_merge : int
            Hypothesis merging depth. Hypotheses with the same mode history
            over the last L_merge steps are merged.
        l_max : int
            Maximum number of hypotheses to maintain.
        initial_mode_probabilities : ndarray, optional
            Initial probabilities for each mode. Defaults to uniform.
        true_trajectory: ndarray, optional
            True trajectory for plotting comparison.
        true_mode: list, optional
            True mode sequence for plotting comparison.
        """
        self.measurements = measurements
        if not F_list:
            raise ValueError("F_list cannot be empty.")
        self.state_dim = F_list[0].shape[0]
        self.num_modes = len(F_list)  # M from the paper

        self.F_list = F_list
        self.H_list = H_list
        self.Q_list = Q_list
        self.R_list = R_list

        self.P_transition = P_transition
        self.epsilon = epsilon
        self.L_merge = L_merge
        self.l_max = l_max

        self.hypotheses = []

        # History for analysis and plotting
        self.hypotheses_count_history = []
        self.best_estimate_history = []
        self.overall_P_history = []
        self.mode_likelihood_history = [] # Store aggregated mode likelihoods

        # For plotting comparison, similar to IMM class
        self.true_trajectory = true_trajectory
        self.true_mode = true_mode


        self._initialize_hypotheses(initial_state, initial_mode_probabilities)


    def _initialize_transition_matrix(self, p_mode):
        """
        Initialize the mode transition probability matrix.
        
        The transition matrix P[i,j] represents the probability of transitioning
        from mode i to mode j. For each mode:
        - Self-transition probability is (1 - p_mode)
        - Transition to other modes is p_mode / (num_filters - 1)
        

        """
        # Initialize matrix with equal transition probabilities to other modes
        P = np.full((self.num_modes, self.num_modes), 
                    p_mode / (self.num_modes - 1))
        
        # Set self-transition probabilities along diagonal
        np.fill_diagonal(P, 1 - p_mode)
        
        return P
    

    def _initialize_hypotheses(self, initial_state_vector, initial_mode_probs):
        """
        Step 1: Initializing hypotheses.
        Sets initial state estimates, covariances, and likelihoods for M hypotheses,
        one for each possible starting mode.
        """
        if initial_mode_probs is None:
            initial_mode_probs = np.ones(self.num_modes) / self.num_modes

        if len(initial_mode_probs) != self.num_modes:
            raise ValueError("Length of initial_mode_probabilities must match num_modes.")

        for j in range(self.num_modes):  # For each mode j
            P_initial = np.eye(self.state_dim) * 1.0
            
            hypothesis = {
                'state': initial_state_vector.copy().reshape(-1, 1),
                'cov': P_initial.copy(),
                'likelihood': initial_mode_probs[j],
                'mode_history': [j],
            }
            self.hypotheses.append(hypothesis)
        
        self._normalize_likelihoods()
        self.hypotheses_count_history.append(len(self.hypotheses))
        self._log_mode_likelihoods()


    def run(self):
        """
        Run the M3H filter algorithm over all measurements.
        Follows Algorithm 1 from the paper.
        """
        for k_time_step, z_k_plus_1 in enumerate(self.measurements):
            # Step 2: Determine new likelihoods (Hypothesis Expansion)
            expanded_hypotheses = []
            for hyp_j in self.hypotheses:
                mode_m_j_at_k = hyp_j['mode_history'][-1]
                for i_next_mode in range(self.num_modes):
                    child_prior_likelihood = self.P_transition[mode_m_j_at_k, i_next_mode] * hyp_j['likelihood']
                    new_mode_history = hyp_j['mode_history'] + [i_next_mode]
                    child_hyp = {
                        'state': hyp_j['state'].copy(),
                        'cov': hyp_j['cov'].copy(),
                        'likelihood': child_prior_likelihood,
                        'mode_history': new_mode_history,
                    }
                    expanded_hypotheses.append(child_hyp)
            self.hypotheses = expanded_hypotheses
            # Step 3: Merge
            self._merge_hypotheses()

            # Step 4: Prune
            self._prune_hypotheses()
            self.hypotheses_count_history.append(len(self.hypotheses))

            # Steps 5, 6, 7
            for hyp_s in self.hypotheses:
                mode_at_k_plus_1 = hyp_s['mode_history'][-1]

                # Step 5: Predict
                F_current = self.F_list[mode_at_k_plus_1]
                Q_current = self.Q_list[mode_at_k_plus_1]
                temp_kf_predict = KalmanFilter(F_current, hyp_s['cov'].copy(), 
                                               self.H_list[mode_at_k_plus_1], 
                                               self.R_list[mode_at_k_plus_1], 
                                               Q_current, hyp_s['state'].copy())
                temp_kf_predict.predict()
                predicted_state_s_kplus1_k = temp_kf_predict.x
                predicted_cov_s_kplus1_k = temp_kf_predict.P

                # Step 6: Update likelihoods & Step 7: Update means and covariances
                H_current = self.H_list[mode_at_k_plus_1]
                R_current = self.R_list[mode_at_k_plus_1]
                temp_kf_update = KalmanFilter(F_current, 
                                              predicted_cov_s_kplus1_k.copy(), 
                                              H_current, R_current,
                                              Q_current, 
                                              predicted_state_s_kplus1_k.copy())
                temp_kf_update.update(z_k_plus_1.reshape(-1,1))
                
                measurement_likelihood_val = multivariate_normal.pdf(
                    temp_kf_update.y.flatten(),
                    mean=np.zeros(temp_kf_update.y.shape[0]),
                    cov=temp_kf_update.S,
                    allow_singular=True
                )
                measurement_likelihood_val = max(measurement_likelihood_val, 1e-300)
                hyp_s['likelihood'] *= measurement_likelihood_val
                hyp_s['state'] = temp_kf_update.x
                hyp_s['cov'] = temp_kf_update.P
            
            # Step 8: Normalize likelihoods
            self._normalize_likelihoods()
            self._log_mode_likelihoods()

            # Step 9: Output combined estimate and covariance
            if self.hypotheses:
                combined_x = np.zeros((self.state_dim, 1))
                for hyp_s in self.hypotheses:
                    combined_x += hyp_s['likelihood'] * hyp_s['state']
                
                combined_P = np.zeros((self.state_dim, self.state_dim))
                for hyp_s in self.hypotheses:
                    diff = hyp_s['state'] - combined_x
                    combined_P += hyp_s['likelihood'] * (hyp_s['cov'] + diff @ diff.T)
            else:
                if self.best_estimate_history:
                    combined_x = self.best_estimate_history[-1].copy()
                    combined_P = self.overall_P_history[-1].copy() if self.overall_P_history else np.eye(self.state_dim) * np.nan
                else:
                    combined_x = np.full((self.state_dim, 1), np.nan)
                    combined_P = np.full((self.state_dim, self.state_dim), np.nan)

            self.best_estimate_history.append(combined_x.copy())
            self.overall_P_history.append(combined_P.copy())

    def _merge_hypotheses(self):
        """
        Merge hypotheses that share the same recent mode history.
        Hypotheses are grouped by the last L_merge modes in their history.
        For each group, keep the first hypothesis and sum the likelihoods of all hypotheses in the group.
        """
        if not self.hypotheses or self.L_merge <= 0:
            return

        # Group hypotheses by their recent mode history (the "tail")
        groups = {}
        for hyp in self.hypotheses:
            # Determine the tail of the mode history to use as the merging key
            tail_length = min(len(hyp['mode_history']), self.L_merge)
            # Ensure a unique key if history is unexpectedly empty or too short for a meaningful tail
            key = tuple(hyp['mode_history'][-tail_length:]) if tail_length > 0 else (id(hyp),)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(hyp)

        merged_list = []
        for _, hypothesis_group in groups.items():
            if len(hypothesis_group) == 1:
                # If only one hypothesis in the group, keep it as is
                merged_list.append(hypothesis_group[0])
            else:
                # For multiple hypotheses, keep the first one and sum likelihoods
                representative_hyp = hypothesis_group[0].copy()
                total_likelihood = sum(h['likelihood'] for h in hypothesis_group)
                representative_hyp['likelihood'] = total_likelihood
                merged_list.append(representative_hyp)
        
        self.hypotheses = merged_list
        # print(len(self.hypotheses),"merged_hypotheses") # Optional: for debugging

    def _prune_hypotheses(self):
        """
        Prune hypotheses based on likelihood threshold (epsilon) and
        cap the number of hypotheses at l_max.
        """
        if not self.hypotheses:
            return

        current_sum_likelihoods = sum(h['likelihood'] for h in self.hypotheses)
        pruning_threshold_value = self.epsilon * current_sum_likelihoods
        survivors = [h for h in self.hypotheses if h['likelihood'] >= pruning_threshold_value]

        if len(survivors) > self.l_max:
            survivors.sort(key=lambda h: h['likelihood'], reverse=True)
            self.hypotheses = survivors[:self.l_max]
        else:
            self.hypotheses = survivors
        self._normalize_likelihoods()

    def _normalize_likelihoods(self):
        """
        Normalize hypothesis likelihoods so they sum to 1.
        """
        if not self.hypotheses:
            return
        
        sum_l = sum(h['likelihood'] for h in self.hypotheses)
        for h in self.hypotheses:
            h['likelihood'] /= sum_l
    
    def _log_mode_likelihoods(self):
        if not self.hypotheses:
            mode_probs = np.zeros(self.num_modes)
        else:
            mode_probs = np.zeros(self.num_modes)
            for hyp in self.hypotheses:
                current_mode = hyp['mode_history'][-1]
                mode_probs[current_mode] += hyp['likelihood']
        self.mode_likelihood_history.append(mode_probs)

    def get_best_estimates(self):
        return np.array([est.flatten() for est in self.best_estimate_history])

    def get_hypotheses_count_history(self):
        return np.array(self.hypotheses_count_history)
        
    def get_mode_likelihood_history(self):
        return np.array(self.mode_likelihood_history)

    def plot_results(self):
        import matplotlib.pyplot as plt
        num_plots = 2
        if self.true_trajectory is not None:
            num_plots +=1
        if self.true_mode is not None:
            num_plots +=1
        
        fig_idx = 1
        plt.figure(figsize=(15, 5 * ((num_plots +1) // 2)))

        if self.true_trajectory is not None:
            ax1 = plt.subplot((num_plots + 1) // 2, 2, fig_idx)
            fig_idx += 1
            measurements_arr = np.array(self.measurements)
            if measurements_arr.ndim == 2 and measurements_arr.shape[1] >= 2:
                 ax1.scatter(measurements_arr[:, 0], measurements_arr[:, 1], c='gray', marker='.', label='Measurements', alpha=0.5)
            best_estimates_arr = self.get_best_estimates()
            if best_estimates_arr.ndim ==2 and best_estimates_arr.shape[1]>=2:
                ax1.plot(best_estimates_arr[:, 0], best_estimates_arr[:, 1], 'r-', label='M3H Estimate')
            if self.true_trajectory.ndim == 2 and self.true_trajectory.shape[1]>=2:
                ax1.plot(self.true_trajectory[:, 0], self.true_trajectory[:, 1], 'g--', label='True Trajectory')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title('Trajectory: True, Measurements, and M3H Estimate')
            ax1.legend()
            ax1.grid(True)

        ax2 = plt.subplot((num_plots + 1) // 2, 2, fig_idx)
        fig_idx +=1
        ax2.plot(self.get_hypotheses_count_history(), marker='o', linestyle='-')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Number of Hypotheses')
        ax2.set_title('Number of Active Hypotheses Over Time')
        ax2.grid(True)

        ax3 = plt.subplot((num_plots + 1) // 2, 2, fig_idx)
        fig_idx +=1
        mode_history_arr = self.get_mode_likelihood_history()
        if mode_history_arr.size > 0 :
            for i in range(self.num_modes):
                ax3.plot(mode_history_arr[:, i], label=f'Mode {i+1} Likelihood')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Aggregated Mode Likelihood')
        ax3.set_title('Aggregated Mode Likelihoods Over Time')
        ax3.legend()
        ax3.grid(True)
        
        if self.true_mode is not None:
            ax4 = plt.subplot((num_plots + 1) // 2, 2, fig_idx)
            fig_idx += 1
            estimated_modes = np.argmax(mode_history_arr, axis=1)
            
            # Ensure true_mode is sliced to match length of estimated_modes for plotting
            len_est = len(estimated_modes)
            true_mode_plot = self.true_mode[:len_est] if len(self.true_mode) > len_est else self.true_mode
            # If estimated_modes is shorter, time_steps should match its length
            time_steps = range(len_est)
            
            print(len(estimated_modes),"estimated_modes")
            print(len(true_mode_plot),"true_mode_plot")
            print(len(time_steps),"time_steps")

            # Cap lengths to true_mode_plot length
            plot_length = len(true_mode_plot)
            ax4.plot(time_steps[:plot_length], true_mode_plot, label='True Mode', linestyle='-', alpha=0.7)
            ax4.plot(time_steps[:plot_length], estimated_modes[:plot_length], label='Estimated Mode (Highest Likelihood)', linestyle='--', alpha=0.7)
            ax4.set_title('True Mode vs Estimated Mode')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Mode Index')
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout()
        plt.show()