import numpy as np
from M3H_2 import M3H
import itertools
from tqdm import tqdm

def calculate_rmse(estimated_trajectory, true_trajectory):
    """Calculate Root Mean Square Error between estimated and true trajectory."""
    return np.sqrt(np.mean(np.sum((estimated_trajectory - true_trajectory)**2, axis=1)))

def calculate_mode_accuracy(estimated_modes, true_modes):
    """Calculate accuracy of mode estimation."""
    return np.mean(estimated_modes == true_modes)

def grid_search(F, H, Q, R, P_transition, measurements, true_trajectory, true_mode):
    # Define parameter ranges
    epsilon_range = np.logspace(-3, -1, 20)
    L_merge_range = [1, 2, 3]
    l_max_range = [10]
    
    # Store results
    results = []
    
    # Create all parameter combinations
    param_combinations = list(itertools.product(epsilon_range, L_merge_range, l_max_range))
    
    # Grid search
    for epsilon, L_merge, l_max in tqdm(param_combinations, desc="Grid Search Progress"):
        try:
            # Initialize and run M3H filter
            m3h_filter = M3H(F, H, Q, R, 
                           initial_state=np.array([0, 0, 0, 0]), 
                           P_transition=P_transition, 
                           measurements=measurements,
                           epsilon=epsilon, 
                           L_merge=L_merge, 
                           l_max=l_max,
                           initial_mode_probabilities=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                           true_trajectory=true_trajectory,
                           true_mode=true_mode)
            
            m3h_filter.run()
            
            # Calculate performance metrics
            rmse = calculate_rmse(m3h_filter.best_estimate, true_trajectory)
            mode_accuracy = calculate_mode_accuracy(m3h_filter.most_likely_mode, true_mode)
            avg_hypotheses = np.mean(m3h_filter.active_hypotheses)
            
            # Store results
            results.append({
                'epsilon': epsilon,
                'L_merge': L_merge,
                'l_max': l_max,
                'rmse': rmse,
                'mode_accuracy': mode_accuracy,
                'avg_hypotheses': avg_hypotheses
            })
            
        except Exception as e:
            print(f"Error with parameters epsilon={epsilon}, L_merge={L_merge}, l_max={l_max}: {str(e)}")
            continue
    
    return results

def analyze_results(results):
    """Analyze and print the grid search results."""
    # Convert results to numpy array for easier analysis
    results_array = np.array([(r['epsilon'], r['L_merge'], r['l_max'], 
                              r['rmse'], r['mode_accuracy'], r['avg_hypotheses']) 
                             for r in results])
    
    # Find best parameters based on RMSE
    best_rmse_idx = np.argmin(results_array[:, 3])
    best_rmse_params = results_array[best_rmse_idx]
    
    # Find best parameters based on mode accuracy
    best_mode_idx = np.argmax(results_array[:, 4])
    best_mode_params = results_array[best_mode_idx]
    
    print("\nBest Parameters based on RMSE:")
    print(f"epsilon: {best_rmse_params[0]}")
    print(f"L_merge: {best_rmse_params[1]}")
    print(f"l_max: {best_rmse_params[2]}")
    print(f"RMSE: {best_rmse_params[3]:.4f}")
    print(f"Mode Accuracy: {best_rmse_params[4]:.4f}")
    print(f"Average Hypotheses: {best_rmse_params[5]:.2f}")
    
    print("\nBest Parameters based on Mode Accuracy:")
    print(f"epsilon: {best_mode_params[0]}")
    print(f"L_merge: {best_mode_params[1]}")
    print(f"l_max: {best_mode_params[2]}")
    print(f"RMSE: {best_mode_params[3]:.4f}")
    print(f"Mode Accuracy: {best_mode_params[4]:.4f}")
    print(f"Average Hypotheses: {best_mode_params[5]:.2f}")
    
    # Print all results sorted by RMSE
    print("\nAll Results (sorted by RMSE):")
    sorted_indices = np.argsort(results_array[:, 3])
    for idx in sorted_indices:
        params = results_array[idx]
        print(f"\nepsilon={params[0]}, L_merge={params[1]}, l_max={params[2]}")
        print(f"RMSE: {params[3]:.4f}, Mode Accuracy: {params[4]:.4f}, Avg Hypotheses: {params[5]:.2f}")

if __name__ == "__main__":
    # Import your existing setup from main.py
    from main import F, H, Q, R, P_transition, mes, true_trajectory, true_mode
    
    # Run grid search
    results = grid_search(F, H, Q, R, P_transition, mes, true_trajectory, true_mode)
    
    # Analyze and print results
    analyze_results(results) 