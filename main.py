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
from GPB import GPB
from kalman import KalmanFilter
from M3H_2 import M3H
from grid_search import grid_search, analyze_results

def main():
    
    F = [get_F_cv(1)]
    angular_velocities_true = {
            0: 0,       # Uniform motion in a straight line
            1: 0.3,     # Constant rate turn with ω = 0.2 rad/sec
            2: -0.3,    # Constant rate turn with ω = -0.2 rad/sec
            3: 0.05,    # Constant rate turn with ω = 0.05 rad/sec
            4: -0.05    # Constant rate turn with ω = -0.05 rad/sec
        }
    angular_velocities_filter = {
            0: 0,       # Uniform motion in a straight line
            1: 0.3,     # Constant rate turn with ω = 0.2 rad/sec
            2: -0.3,    # Constant rate turn with ω = -0.2 rad/sec
            3: 0.05,    # Constant rate turn with ω = 0.05 rad/sec
            4: -0.05    # Constant rate turn with ω = -0.05 rad/sec
        }
    for i in range(1, 5):
        F.append(get_F_ct(1, angular_velocities_filter[i]))

    
    H_n = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    H = [H_n, H_n,H_n, H_n,H_n]

    Qn = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.001, 0],
            [0, 0, 0, 0.001 ],]
    
    Z = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0.01, 0],
                    [0, 0, 0, 0.01],])
    
    Q = [Qn, Qn, Qn, Qn,Qn]
    Rn = 1 * np.eye(4)
    R = [Rn, Rn,Rn, Rn,Rn]
    p_mode = 0.02

    P_transition = np.full((5, 5), p_mode / (5 - 1))
        
    # Set self-transition probabilities along diagonal
    np.fill_diagonal(P_transition, 1 - p_mode)

    # display(measurements, 0.5)
    agent = Agent(initial_pos=(0, 0), velocity=(1, 1), angular_velocities=angular_velocities_true, F=F,H=H,Q=Z, p_mode=p_mode)
    true_trajectory = agent.generate_trajectory(T=500)
    mes = agent.get_measurements()
    true_mode = agent.get_mode_history()


    # imm = IMM(F, H, Q, R, initial_state=[0, 0, 0, 0],p_mode=p_mode,true_mode=true_mode,true_trajectory=true_trajectory, measurements=mes,MIXING=True)
    # imm.run()
    # imm.plot_results()

    # gpb = GPB(F,H,Q,R,initial_state=[0, 0, 0, 0],p_mode=p_mode,true_mode=true_mode,true_trajectory=true_trajectory, measurements=mes,order=2)
    # gpb.run()
    # gpb.plot_results()
    # plt.show()
    m3h_filter = M3H(F, H, Q, R, 
                       initial_state=np.array([0, 0, 0, 0]), 
                       P_transition=P_transition, 
                       measurements=mes,
                       epsilon=0.004, 
                       L_merge=4, 
                       l_max=25,
                       initial_mode_probabilities=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                       true_trajectory=true_trajectory, # For plotting
                       true_mode=true_mode # For plotting
                       )

    m3h_filter.run()
    m3h_filter.plot_results()
    plt.show()
    # results = grid_search(F, H, Q, R, P_transition, mes, true_trajectory, true_mode)
    
    # # Analyze and print results
    # analyze_results(results) 

if __name__ == "__main__":
    main()
    



