# multiple_model_estimation

This project aims to implement and compare Generalized pseudo baeysian GPB, Interacting multiple model IMM and Multiple model multiple hypothesis filters.

# Multiple Model Tracking System

This project implements various multiple model tracking algorithms for tracking targets with different motion patterns. The system includes implementations of Interacting Multiple Model (IMM), Generalized Pseudo-Bayesian (GPB), and M3H algorithms.

## Project Structure

- `main.py`: Main entry point that demonstrates the usage of different tracking algorithms
- `agent.py`: Implements the target agent with different motion models (constant velocity and constant turn)
- `IMM.py`: Implementation of the Interacting Multiple Model algorithm
- `GPB.py`: Implementation of the Generalized Pseudo-Bayesian algorithm
- `M3H.py` and `M3H_2.py`: Implementation of the M3H tracking algorithm
- `kalman.py`: Basic Kalman filter implementation
- `display.py`: Visualization utilities for trajectories and tracking results
- `generate_trajectory.py`: Helper functions for trajectory generation
- `grid_search.py`: Tools for parameter optimization

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Usage

1. Install the required dependencies:
```bash
pip install numpy matplotlib
```

2. Run the main script:
```bash
python main.py
```

The main script demonstrates the usage of different tracking algorithms. By default, it runs the M3H algorithm, but you can uncomment other algorithm implementations (IMM or GPB) to try them.

## Features

- Multiple motion models (constant velocity and constant turn)
- Different tracking algorithms (IMM, GPB, M3H)
- Visualization of true trajectories and tracking results
- Parameter optimization through grid search
- Measurement noise handling
- Mode transition probability modeling

## Customization

You can modify the following parameters in `main.py`:
- Initial position and velocity
- Angular velocities for different motion modes
- Measurement and process noise matrices
- Mode transition probabilities
- Number of time steps for simulation

## Output

The system provides:
- Real-time visualization of the tracking process
- Comparison between true trajectory and estimated trajectory
- Mode probability plots
- Performance metrics
