import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.chdir(os.getcwd())
S=49
A=4
R1 = np.load("R1.npy")
R2 = np.load("R2.npy")
T = np.load("T.npy")

r=R1+R2
terminal_state = np.load("terminal.npy")
start_state = np.load("initial.npy")
gamma = 0.9

def compute_q_values(S, A, R, T, gamma):
    # Initialize Q-values to zeros
    Q_new = np.zeros((S, A))
    
    # Maximum number of iterations for value iteration
    max_iterations = 5000
    
    # Value iteration
    for _ in range(max_iterations):
        Q = Q_new.copy()
        for s in range(S):
            for a in range(A):
                q_sa = 0
                for s_prime in range(S):
                    q_sa += T[s][a][s_prime] * (R[s][a][s_prime] + gamma * np.max(Q[s_prime]))
                Q_new[s][a] = q_sa
        if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
            print("Converged in", _ + 1, "iterations")
            break
        Q = Q_new
    
    return Q

Q1 = compute_q_values(S, A, R1, T, gamma)
a1 = np.argmax(Q1,axis=1)
Q2 = compute_q_values(S, A, R2, T, gamma)
a2 = np.argmax(Q2,axis=1)
Q3 = compute_q_values(S, A, R1+R2, T, gamma)
a3 = np.argmax(Q3,axis=1)

cQ = np.zeros((S,A))
for s in range(S):
    for a in range(A):
        cQ[s,a] = max(Q1[s,a],Q2[s,a])
        
a4 = np.argmax(cQ,axis=1)
print(np.where(a4==a3)[0].shape)