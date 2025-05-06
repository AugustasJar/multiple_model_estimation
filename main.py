import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import numpy as np
from display import display
from generate_trajectory import generate_trajectory
from agent import Agent
from IMM import IMM
from IMM import get_F_cv,get_F_ct
from kalman import KalmanFilter
def main():
    
    F = [get_F_cv(1)]
    angular_velocities = {
            0: 0,       # Uniform motion in a straight line
            1: 0.3,     # Constant rate turn with ω = 0.1 rad/sec
            2: -0.3,    # Constant rate turn with ω = -0.1 rad/sec
            3: 0.05,    # Constant rate turn with ω = 0.05 rad/sec
            4: -0.05    # Constant rate turn with ω = -0.05 rad/sec
        }
    for i in range(1, 5):
        F.append(get_F_ct(1, angular_velocities[i]))

    
    H_n = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    H = [H_n, H_n,H_n, H_n,H_n]

    Qn = [[0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.001, 0],
            [0, 0, 0, 0.001 ],]
    
    Z = np.array([[2, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0.5, 0],
                    [0, 0, 0, 0.5],])
    Q = [Qn, Qn, Qn, Qn,Qn]
    Rn = 2 * np.eye(4)
    R = [Rn, Rn,Rn, Rn,Rn]
    p_mode = 0.04
    

    # display(measurements, 0.5)
    agent = Agent(initial_pos=(0, 0), velocity=(1, 1), angular_velocities=angular_velocities, F=F,H=H,Q=Z, p_mode=p_mode)
    _ = agent.generate_trajectory(T=500)
    mes = agent.get_measurements()
    true_mode = agent.get_mode_history()
    imm = IMM(F, H, Q, R, initial_state=[0, 0, 0, 0],p_mode=p_mode,true_mode=true_mode, measurements=mes,MIXING=True)
    imm.run()
    imm.plot_results()

if __name__ == "__main__":
    main()



