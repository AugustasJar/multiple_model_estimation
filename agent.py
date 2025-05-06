import numpy as np
import random

class Agent:
    def __init__(self, initial_pos=(0,0), velocity=(0.1,0.2), acceleration=(0,0), F=None,H =None, Q=None, p_mode=None):
        self.state = np.array([initial_pos[0], initial_pos[1], velocity[0], velocity[1]])
        self.state_history = []
        self.measurements = []
        self.mode_history = []
        # Use provided state transition matrices or default to an empty list
        self.F = F if F is not None else []
        self.H = H
        self.Q = Q

        # Probability matrix for mode transitions
        self.p_mode = p_mode
        
        # Initialize the current mode index
        self.modes = 4
        self.mode = random.randint(0, self.modes)
        
    def update_mode(self):
        """Update the mode probabilistically based on p_mode."""
        # print('mode', self.mode)
        if random.random() < self.p_mode:
            new_mode = random.randint(0, self.modes)
            if new_mode != self.mode:
                self.mode = new_mode
            else:
                self.update_mode() # prevent switching to same mode

    def update_state(self):
        """Update the state based on the current mode."""
        self.update_mode()  # Update the mode probabilistically
        
        x, y, vx, vy = self.state

        # Define angular velocities for the modes
        angular_velocities = {
            0: 0,       # Uniform motion in a straight line
            1: 0.1,     # Constant rate turn with ω = 0.1 rad/sec
            2: -0.1,    # Constant rate turn with ω = -0.1 rad/sec
            3: 0.05,    # Constant rate turn with ω = 0.05 rad/sec
            4: -0.05    # Constant rate turn with ω = -0.05 rad/sec
        }

        # Get the angular velocity for the current mode
        omega = angular_velocities.get(self.mode, 0)

        # Update velocity and position based on the mode
        if omega == 0:
            # Uniform motion in a straight line
            x += vx
            y += vy
        else:
            # Constant rate turn
            v = np.sqrt(vx**2 + vy**2)  # Speed
            theta = np.arctan2(vy, vx)  # Current direction
            theta += omega  # Update direction based on angular velocity
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)
            x += vx
            y += vy

        # Update the state
        self.state = np.array([x, y, vx, vy])
        
        self.state_history.append(self.state.copy())

        noise = np.random.multivariate_normal([0, 0, 0, 0], self.Q)
        s = self.state + noise
        measurement = np.dot(self.H[0],s)

        self.measurements.append(measurement.copy())
        self.mode_history.append(self.mode)
    def get_state_history(self):
        return self.state_history.copy()
    
    def get_measurements(self):
        return self.measurements.copy()
    
    def get_mode_history(self):
        return self.mode_history.copy()
    
    
    def generate_trajectory(self, T=10):
        for t in range(T):
            self.update_state()
        return np.array(self.state_history)