import numpy as np

def calculate_rmse(estimated_trajectory, true_trajectory):
    """Calculate Root Mean Square Error between estimated and true trajectory."""
    return np.sqrt(np.mean(np.sum((estimated_trajectory - true_trajectory)**2, axis=1)))

def calculate_mode_accuracy(estimated_mode, true_mode):
    """Calculate accuracy of mode estimation."""
    return np.mean(estimated_mode == true_mode) 