import numpy as np
def generate_trajectory(type='linear',T=10):
    """
    Generates a trajectory of measurements.
    
    Returns:
        list: A list of measurement values.
    """
    x = np.zeros((T, 2))  # Initialize trajectory array
    # Example trajectory generation (replace with actual logic)
    if type == 'linear':
        for t in range(T):
            x[t] = [t, t]
    
    return x
