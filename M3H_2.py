import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

from IMM import IMM
from kalman import KalmanFilter


class M3H(IMM):
    def __init__(self, F_list, H_list, Q_list, R_list, initial_state, P_transition, measurements, epsilon, L_merge, l_max, initial_mode_probabilities=None, true_trajectory=None, true_mode=None):
        p_mode = 1 - P_transition[0,0]
        super().__init__(F_list, H_list, Q_list, R_list, initial_state, p_mode, measurements, initial_mode_probabilities, true_trajectory, true_mode)
        self.epsilon = epsilon
        self.L_merge = L_merge
        self.l_max = l_max
        if initial_mode_probabilities is None:
            self.initial_mode_probabilities = np.ones(len(F_list)) / len(F_list)
        else:
            self.initial_mode_probabilities = initial_mode_probabilities
        
        self.F_list = F_list
        self.H_list = H_list
        self.Q_list = Q_list
        self.R_list = R_list
        self.initial_state = initial_state
        self.num_modes = len(F_list)
        self.hypotheses = self._init_hypotheses()
        self.P_transition = P_transition
        self.best_estimate = np.zeros((len(measurements),self.state_dim))
        self.mode_likelihoods = np.zeros((len(measurements),self.num_modes))
        self.active_hypotheses = np.zeros((len(measurements)))
        self.measurements = measurements
        self.true_trajectory = true_trajectory
        self.most_likely_mode = np.zeros((len(measurements)))
        self.true_mode = true_mode
        self.merged_covariance = np.zeros((len(measurements),self.state_dim,self.state_dim))

    def _init_filters(self):
        filters = []
        P = np.eye(self.F_list[0].shape[0])
        for i in range(self.num_modes):
            filter = KalmanFilter(self.F_list[i], P, self.H_list[i], self.R_list[i], self.Q_list[i], self.initial_state)
            filters.append(filter)
        return filters

    def _init_hypotheses(self):
        """Initialize hypotheses for the M3H filter."""
        self.filters = self._init_filters()
        hypotheses = []
        for i in range(self.num_modes):
            hypothesis = {
                'mode_history': [i],
                'likelihood': self.initial_mode_probabilities[i],
                'filter': self.filters[i],
            }
            hypotheses.append(hypothesis)
        return hypotheses
            

    def _expand_hypotheses(self):
        """Expand hypotheses."""
        new_hypotheses = []
        for hypothesesis in self.hypotheses:
            for new_mode in range(self.num_modes):
                active_mode = hypothesesis['mode_history'][-1]
                
                # Get the last L_merge-1 modes from history if long enough, otherwise use full history
                if len(hypothesesis['mode_history']) >= self.L_merge:
                    mode_history = hypothesesis['mode_history'][-self.L_merge+1:]
                else:
                    mode_history = hypothesesis['mode_history']

                # Append the new mode
                mode_history = mode_history + [new_mode]
                new_hypothesis = {
                    #expand the likelihoods based on the transition matrix
                    'mode_history': mode_history,
                    'likelihood': hypothesesis['likelihood'] * self.P_transition[active_mode, new_mode],
                    'filter': hypothesesis['filter'].copy(),  # Using deepcopy for complete independence
                }

                # update filter mode, assume that only the transition matrix is changed
                new_hypothesis['filter'].F = self.F_list[new_mode]

                new_hypotheses.append(new_hypothesis)
        self.hypotheses = new_hypotheses


    def _merge_hypotheses(self):
        """
        Merge hypotheses that share the same recent mode history.
        This method groups hypotheses by their 'mode_history'. For each group,
        if there are multiple hypotheses, their likelihoods are summed, and a
        single merged hypothesis is created. The filter for the merged hypothesis
        is a copy of the filter from the first hypothesis in that group.
        """
        if not self.hypotheses or len(self.hypotheses) < 2:
            # No need to merge if there are 0 or 1 hypotheses.
            # The L_merge parameter's direct role here is implicitly handled by
            # how mode_history is constructed in _expand_hypotheses.
            return

        # Group hypotheses by their mode_history.
        # The mode_history is expected to be managed by _expand_hypotheses
        # to represent the relevant sequence for merging (e.g., fixed length L_merge).
        groups = {}
        for hyp in self.hypotheses:
            key = tuple(hyp['mode_history'])  # Use the tuple of mode_history as the key
            if key not in groups:
                groups[key] = []
            groups[key].append(hyp)

        merged_list = []
        for key, group_hypotheses in groups.items():
            if len(group_hypotheses) == 1:
                # If only one hypothesis in the group, keep it as is.
                merged_list.append(group_hypotheses[0])
            else:
                representative_base = group_hypotheses[0]
                
                # Create the new merged hypothesis.
                merged_hyp = {
                    'mode_history': list(key),  # Convert key back to list for consistency
                    'likelihood': sum(h['likelihood'] for h in group_hypotheses),
                    'filter': representative_base['filter'].copy()
                }
                #update the filter state based on the likelihoods
                merged_hyp['filter'].x = sum(h['likelihood'] * h['filter'].x for h in group_hypotheses) / sum(h['likelihood'] for h in group_hypotheses)
                merged_hyp['filter'].P = sum(h['likelihood'] * h['filter'].P for h in group_hypotheses) / sum(h['likelihood'] for h in group_hypotheses)
                merged_list.append(merged_hyp)

        self.hypotheses = merged_list

    def _prune_hypotheses(self):
        #remove hypotheses with likelihood less than epsilon
        threshold = self.epsilon * sum(hypothesis['likelihood'] for hypothesis in self.hypotheses)
        for hypothesesis in self.hypotheses:
            if hypothesesis['likelihood'] < threshold and len(hypothesesis['mode_history']) > 1:
                self.hypotheses.remove(hypothesesis)
        #keep L_max hypotheses
        #sort hypotheses by likelihood
        self.hypotheses.sort(key=lambda x: x['likelihood'], reverse=True)
        #keep only the top L_max hypotheses
        self.hypotheses = self.hypotheses[:self.l_max]

    def _filter_update(self,z):
        for hypothesesis in self.hypotheses:
            hypothesesis['filter'].predict()
            hypothesesis['filter'].update(np.array(z))
            #no reason to save state and covariance, because the filters degenerate.

    def _update_mode_probabilities(self,z):
        #update the likelihoods based on the measurement
        total_likelihood = 0
        
        for hypothesesis in self.hypotheses:
            # Get the innovation and innovation covariance
            innovation = hypothesesis['filter'].y
            innovation_cov = hypothesesis['filter'].S
            
            # Calculate measurement likelihood using innovation
            measurement_likelihood_val = multivariate_normal.pdf(
                innovation.flatten(),
                mean=np.zeros(innovation.shape[0]),
                cov=innovation_cov,
                allow_singular=True
            )
            
            # Update hypothesis likelihood
            hypothesesis['likelihood'] = measurement_likelihood_val*hypothesesis['likelihood']
            total_likelihood += hypothesesis['likelihood']
        
        #normalize likelihoods, if total likelihood is too small, set all likelihoods to 1/num_modes
        for hypothesesis in self.hypotheses:
            hypothesesis['likelihood'] = hypothesesis['likelihood'] / total_likelihood

    def _log(self,idx):

        best_estimate = np.zeros((self.state_dim,1))
        for hypothesesis in self.hypotheses:
            best_estimate += hypothesesis['likelihood'] * hypothesesis['filter'].x
        self.best_estimate[idx] = best_estimate.reshape(self.state_dim)
        
        #log aggregate mode likelihoods
        mode_likelihoods = np.zeros(self.num_modes)
        for mode in range(self.num_modes):
            for hypothesesis in self.hypotheses:
                if hypothesesis['mode_history'][-1] == mode:
                    mode_likelihoods[mode] += hypothesesis['likelihood']
        self.mode_likelihoods[idx] = mode_likelihoods
        
        #log active hypotheses
        self.active_hypotheses[idx] = len(self.hypotheses)

        #most likely mode
        self.most_likely_mode[idx] = np.argmax(mode_likelihoods)

        #log merged covariance
        merged_covariance = np.zeros((self.state_dim,self.state_dim))
        for hypothesesis in self.hypotheses:
            merged_covariance += hypothesesis['likelihood'] * (hypothesesis['filter'].P + (best_estimate - hypothesesis['filter'].x) @ (best_estimate - hypothesesis['filter'].x).T)
        self.merged_covariance[idx] = merged_covariance


    def run(self):
        """Run the M3H filter."""
        #expand hypotheses
        for idx, z in enumerate(self.measurements):
            self._expand_hypotheses()

            self._merge_hypotheses()

            self._prune_hypotheses()

            self._filter_update(z)

            self._update_mode_probabilities(z)

            self._log(idx)

        
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
            true_trajectory_arr = np.array(self.true_trajectory)
            if measurements_arr.ndim == 2 and measurements_arr.shape[1] >= 2:
                 ax1.scatter(measurements_arr[:, 0], measurements_arr[:, 1], c='gray', marker='.', label='Measurements', alpha=0.5)
            if self.best_estimate.ndim == 2 and self.best_estimate.shape[1] >= 2:
                # Plot error ellipses for each point
                for i in range(len(self.best_estimate)):
                    # Get the 2x2 covariance matrix for position
                    cov = self.merged_covariance[i][:2, :2]
                    # Calculate eigenvalues and eigenvectors
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    # Calculate angle of rotation
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    # Create ellipse with 95% confidence interval (2 sigma)
                    ellipse = Ellipse(
                        xy=self.best_estimate[i, :2],
                        width=2 * np.sqrt(eigenvals[0]) * 2,
                        height=2 * np.sqrt(eigenvals[1]) * 2,
                        angle=angle,
                        alpha=0.1,
                        color='blue'
                    )
                    ax1.add_patch(ellipse)
                
                ax1.plot(self.best_estimate[:, 0], self.best_estimate[:, 1], 'r-', label='M3H Estimate')

            ax1.plot(self.true_trajectory[:, 0], self.true_trajectory[:, 1], label=f'true trajectory',color='green')


            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title('Trajectory: True, Measurements, and M3H Estimate')
            ax1.legend()
            ax1.grid(True)

        ax2 = plt.subplot((num_plots + 1) // 2, 2, fig_idx)
        fig_idx +=1
        ax2.plot(self.active_hypotheses, marker='o', linestyle='-')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Number of Hypotheses')
        ax2.set_title('Number of Active Hypotheses Over Time')
        ax2.grid(True)

        ax3 = plt.subplot((num_plots + 1) // 2, 2, fig_idx)
        fig_idx +=1
        if self.mode_likelihoods.size > 0:
            for i in range(self.num_modes):
                ax3.plot(self.mode_likelihoods[:, i], label=f'Mode {i+1} Likelihood')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Aggregated Mode Likelihood')
        ax3.set_title('Aggregated Mode Likelihoods Over Time')
        ax3.legend()
        ax3.grid(True)

        ax4 = plt.subplot((num_plots + 1) // 2, 2, fig_idx)
        fig_idx += 1
        time_steps = range(len(self.true_mode))  # Time steps

        ax4.plot(time_steps, self.true_mode, label='True Mode', linestyle='-', alpha=0.7)
        ax4.plot(time_steps, self.most_likely_mode, label='Estimated Mode', linestyle='--', alpha=0.7)
        ax4.set_title('True Mode vs Estimated Mode')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Mode')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()
