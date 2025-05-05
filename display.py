import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def display(mes, modes, t_delay):
    """
    Displays animated XY graphs of the measurements vector 'mes' and the modes vector 'modes' with a time dimension.
    
    Parameters:
        mes (list): A list of tuples, where each tuple contains (time, x, y).
        modes (list): A list of integers representing mode values.
        t_delay (float): Delay (in seconds) between each frame.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Data for the first subplot (measurements)
    time_data, x_data, y_data = [], [], []
    line1, = ax1.plot([], [], 'b-', marker='.', label="Measurements")
    
    # Data for the second subplot (modes)
    mode_time_data, mode_data = [], []
    line2, = ax2.plot([], [], 'r-', label="Modes")
    
    def init():
        # Initialize the first subplot
        ax1.set_xlim(min(mes[:, 0]) - 1, max(mes[:, 0]) + 1)
        ax1.set_ylim(min(mes[:, 1]) - 1, max(mes[:, 1]) + 1)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.legend()
        
        # Initialize the second subplot
        ax2.set_xlim(0, len(modes) - 1)
        ax2.set_ylim(min(modes) - 1, max(modes) + 1)
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Mode Value")
        ax2.legend()
        
        return line1, line2

    def update(frame):
        # Update the first subplot
        if frame == 0:  # Clear data at the start of the animation
            time_data.clear()
            x_data.clear()
            y_data.clear()
            mode_time_data.clear()
            mode_data.clear()
        time_data.append(mes[frame][0])
        x_data.append(mes[frame][0])
        y_data.append(mes[frame][1])
        line1.set_data(x_data, y_data)
        
        # Update the second subplot
        mode_time_data.append(frame)
        mode_data.append(modes[frame])
        line2.set_data(mode_time_data, mode_data)
        
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=range(len(mes)), init_func=init, blit=True, interval=t_delay * 1000, repeat=False)
    plt.tight_layout()
    plt.show()

def plot(predictions, covariance):
        positions = predictions[:, :2]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(positions[:, 0], positions[:, 1], label="Predicted Trajectory")
        plt.title("Predicted Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()

        # Plot covariance (trace of covariance matrix as a measure of uncertainty)
        cov_trace = [np.trace(cov) for cov in covariance]
        plt.subplot(1, 2, 2)
        plt.plot(range(len(cov_trace)), cov_trace, label="Covariance Trace")
        plt.title("Covariance Trace Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Trace of Covariance")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()