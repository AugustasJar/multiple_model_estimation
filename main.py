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
            1: 0.5,     # Constant rate turn with ω = 0.2 rad/sec
            2: -0.5,    # Constant rate turn with ω = -0.2 rad/sec
            3: 0.05,    # Constant rate turn with ω = 0.05 rad/sec
            4: -0.05    # Constant rate turn with ω = -0.05 rad/sec
        }
    angular_velocities_filter = {
            0: 0,       # Uniform motion in a straight line
            1: 0.5,     # Constant rate turn with ω = 0.2 rad/sec
            2: -0.5,    # Constant rate turn with ω = -0.2 rad/sec
            3: 0.05,    # Constant rate turn with ω = 0.05 rad/sec
            4: -0.05    # Constant rate turn with ω = -0.05 rad/sec
        }
    for i in range(1, len(angular_velocities_filter)):
        F.append(get_F_ct(1, angular_velocities_filter[i]))

    
    H_n = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    H = [H_n, H_n,H_n, H_n,H_n]

    Qn = [[0.5, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0.0001, 0],
            [0, 0, 0, 0.0001 ]]
    
    Z = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0.001, 0],
                    [0, 0, 0, 0.001],])
    
    Q = [Qn, Qn, Qn, Qn,Qn]
    Rn = 1 * np.eye(4)
    R = [Rn, Rn,Rn, Rn,Rn]
    p_mode = 0.05

    P_transition = np.full((5, 5), p_mode / (len(Q) - 1))
        
    # Set self-transition probabilities along diagonal
    np.fill_diagonal(P_transition, 1 - p_mode)

    # display(measurements, 0.5)
    agent = Agent(initial_pos=(0, 0), velocity=(1, 1), angular_velocities=angular_velocities_true, F=F,H=H,Q=Z, p_mode=p_mode)
    true_trajectory = agent.generate_trajectory(T=200)
    mes = agent.get_measurements()
    true_mode = agent.get_mode_history()

    initial_mode = true_mode[0]
    initial_mode_probabilities = np.zeros(5)
    initial_mode_probabilities[initial_mode] = 1.0
    rmse_imm = []
    rmse_gpb = []
    rmse_m3h = []
    mode_acc_imm = []
    mode_acc_gpb = []
    mode_acc_m3h = []
    avg_mode_count_m3h = []
    epsilon = 0.0063
    L_merge = 4
    l_max = 10
    for i in range(10):
        # Initialize agent with same parameters
        agent = Agent(initial_pos=(0, 0), velocity=(1, 1), angular_velocities=angular_velocities_true, F=F, H=H, Q=Z, p_mode=p_mode)
        true_trajectory = agent.generate_trajectory(T=100)
        mes = agent.get_measurements()
        true_mode = agent.get_mode_history()

        # Initialize IMM with same parameters
        imm = IMM(F, H, Q, R, initial_state=[0, 0, 0, 0], p_mode=p_mode, true_mode=true_mode, true_trajectory=true_trajectory, measurements=mes, MIXING=True)
        gpb = GPB(F,H,Q,R,initial_state=[0, 0, 0, 0],p_mode=p_mode,true_mode=true_mode,true_trajectory=true_trajectory, measurements=mes,order=2)
        m3h = M3H(F,H,Q,R,initial_state=[0, 0, 0, 0],P_transition=P_transition,measurements=mes,epsilon=epsilon,L_merge=L_merge,l_max=l_max,initial_mode_probabilities=initial_mode_probabilities,true_trajectory=true_trajectory,true_mode=true_mode)
        imm.run()
        gpb.run()
        m3h.run()
        
        # Save RMSE
        rmse_imm.append(imm.get_rmse())
        rmse_gpb.append(gpb.get_rmse())
        rmse_m3h.append(m3h.get_rmse())
        mode_acc_imm.append(imm.get_mode_accuracy())
        mode_acc_gpb.append(gpb.get_mode_accuracy())
        mode_acc_m3h.append(m3h.get_mode_accuracy())
        avg_mode_count_m3h.append(m3h.get_avg_mode_count())
    

    # Calculate and print average RMSE
    avg_rmse_imm = np.mean(rmse_imm)
    avg_rmse_gpb = np.mean(rmse_gpb)
    avg_rmse_m3h = np.mean(rmse_m3h)
    print(f"IMM RMSE: {avg_rmse_imm}, GPB RMSE: {avg_rmse_gpb}, M3H RMSE: {avg_rmse_m3h}")
    print(f"IMM Mode Accuracy: {np.mean(mode_acc_imm)}, GPB Mode Accuracy: {np.mean(mode_acc_gpb)}, M3H Mode Accuracy: {np.mean(mode_acc_m3h)}")
    print(f"M3H avg mode count: {np.mean(avg_mode_count_m3h)}")
    
    imm = IMM(F, H, Q, R, initial_state=[0, 0, 0, 0], p_mode=p_mode, true_mode=true_mode, true_trajectory=true_trajectory, measurements=mes, MIXING=True)
    imm.run()
    imm.plot_results()

    gpb = GPB(F,H,Q,R,initial_state=[0, 0, 0, 0],p_mode=p_mode,true_mode=true_mode,true_trajectory=true_trajectory, measurements=mes,order=2)
    gpb.run()
    gpb.plot_results()
    m3h = M3H(F, H, Q, R, 
                       initial_state=np.array([0, 0, 0, 0]), 
                       P_transition=P_transition, 
                       measurements=mes,
                       epsilon=epsilon, 
                       L_merge=L_merge, 
                       l_max=l_max,
                       initial_mode_probabilities=initial_mode_probabilities,
                       true_trajectory=true_trajectory, # For plotting
                       true_mode=true_mode # For plotting)
                       )

    m3h.run()
    m3h.plot_results()
    plt.show()
    # # results = grid_search(F, H, Q, R, P_transition, mes, true_trajectory, true_mode)
    
    # # Analyze and print results
    # analyze_results(results) 

if __name__ == "__main__":
    main()
    



