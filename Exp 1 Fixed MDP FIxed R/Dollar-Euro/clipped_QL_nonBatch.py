from scipy import stats
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time
from scipy.sparse import csr_matrix

def read_mdp(mdp):
    """Function to read MDP file"""
    f = open(mdp)
    S = int(f.readline())
    A = int(f.readline())

    R = np.zeros((S, A, S))
    R2 = np.zeros((S, A, S))
    T = np.zeros((S, A, S))

    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R[s][a][sPrime] = line[sPrime]
                
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R2[s][a][sPrime] = line[sPrime]
    
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                T[s][a][sPrime] = line[sPrime]

    gamma = float(f.readline().rstrip())
    f.close()

    return S, A, R, R2, T, gamma

S, A, R, R2, T, gamma = read_mdp("gridworld_mdp(r1+r2).txt")
R[31,1,40] = 0.6
R2[31,1,40] = 0.6
R[41,2,40] = 0.6
R2[41,2,40] = 0.6
R[39,3,40] = 0.6
R2[39,3,40] = 0.6

def make_transitions_stochastic(T1, x_percentage):
    S, A, _ = T1.shape
    num_deterministic_states = int(S * x_percentage/100)
    deterministic_states = np.random.choice(S, num_deterministic_states, replace=False)
    T_stochastic = np.zeros_like(T1)
    
    for s in deterministic_states:
        for a in range(A):
            max_prob_index = np.argmax(T1[s, a])
            T_stochastic[s, a, max_prob_index] = 1
     
    stochastic_states = np.setdiff1d(np.arange(S), deterministic_states)
    for s in stochastic_states:
        for a in range(A):
            T_stochastic[s,a] = T1[s,a]
    return T_stochastic

def restrict_transition(matrix, max_bf):
    S, A, S_prime = matrix.shape
    n_values = np.random.randint(1, max_bf + 1, size=S)
    sorted_matrix = np.argsort(matrix, axis=2)
    mask = np.zeros_like(matrix)
    
    for s in range(S):
        for a in range(A):
            mask[s, a, sorted_matrix[s, a, -n_values[s]:]] = 1

    restricted_matrix = matrix * mask
    row_sums = restricted_matrix.sum(axis=2, keepdims=True)
    normalized_matrix = restricted_matrix / row_sums
    return normalized_matrix

def compute_delta(Q, state_idx, action, next_state_idx, reward, gamma):
    """Compute Delta(s,a) = r + γV(s') - Q(s,a)"""
    V_next = np.max(Q[next_state_idx])
    return reward + gamma * V_next - Q[state_idx, action]

# def compute_bounds(Q, T1, R, gamma):
#     """Compute bounds similar to NewSoftQLearning but for standard Q-learning"""
#     S, A, _ = T1.shape
#     Qi = Q  # S×A
    
#     # Compute V values (max Q)
#     V = np.max(Q, axis=1)  # S
    
#     # Compute Qj = T1.dot(V) which is E[V(s')|s,a] = sum_s' T1(s,a,s')*V(s')
#     Qj = np.zeros((S, A))
#     for s in range(S):
#         for a in range(A):
#             Qj[s,a] = np.sum(T1[s,a] * V)
    
#     # Compute expected reward R(s,a) = sum_s' T1(s,a,s')*R(s,a,s')
#     expected_R = np.zeros((S, A))
#     for s in range(S):
#         for a in range(A):
#             expected_R[s,a] = np.sum(T1[s,a] * R[s,a])
    
#     # Compute delta_rwd = R(s,a) + gamma * Qj(s,a) - Q(s,a)
#     delta_rwd = expected_R + gamma * Qj - Q
    
#     # Get min and max delta over all state-action pairs
#     delta_min, delta_max = np.min(delta_rwd), np.max(delta_rwd)
    
#     # Compute bounds
#     lb = Q + delta_rwd + gamma * delta_min / (1 - gamma)
#     ub = Q + delta_rwd + gamma * delta_max / (1 - gamma)
    
#     # Clip bounds to reasonable values
#     min_reward = np.min(R)
#     max_reward = np.max(R)
#     lb = np.maximum(lb, min_reward / (1 - gamma))
#     ub = np.minimum(ub, max_reward / (1 - gamma))
    
#     return lb, ub
def compute_bounds(Q, dynamics_table, rewards_table, gamma):
    """Model-free version matching NewSoftQLearning's approach"""
    S, A = Q.shape
    
    # Compute V values (max Q)
    V = np.max(Q, axis=1)  # shape (S,)
    
    # Reshape dynamics table to (S*A)×S and convert to sparse
    dyn_flat = dynamics_table.reshape(S*A, S)
    dyn_sparse = csr_matrix(dyn_flat.T)  # shape S×(S*A)
    
    # Compute Qj = T*V = E[V(s')|s,a] (equivalent to original code)
    Qj = dyn_sparse.T.dot(V).reshape(S, A)  # shape (S,A)
    
    # Compute delta_rwd = r + γQj - Q (using last observed rewards)
    delta_rwd = rewards_table + gamma * Qj - Q  # shape (S,A)
    
    # Handle infinite/nan values (like original)
    finite_mask = np.isfinite(delta_rwd)
    if not np.all(finite_mask):
        finite_vals = delta_rwd[finite_mask]
        delta_rwd[~finite_mask] = np.mean(finite_vals) if len(finite_vals) > 0 else 0
    
    # Compute min/max delta only over finite values
    delta_min, delta_max = np.min(delta_rwd[finite_mask]), np.max(delta_rwd[finite_mask])
    
    # Compute bounds (identical to original)
    lb = Q + delta_rwd + gamma * delta_min / (1 - gamma)
    ub = Q + delta_rwd + gamma * delta_max / (1 - gamma)
    
    # Clip to global reward bounds (if available)
    lb = np.maximum(lb, -1 / (1 - gamma))  # Default min bound
    ub = np.minimum(ub, 1 / (1 - gamma))   # Default max bound
    
    return lb, ub

def compute_state_value(Q, state_idx):
    """Compute V(s) = max_a Q(s,a)"""
    return np.max(Q[state_idx])

data = []
init = int(input("start: "))
last = int(input("end: "))

for avg in range(init, last):
    for nst in range(1, 5):
        gamma = 0.939
        terminal_state = [0, 8, 40]
        x_percentage = 0
        number_stochastic_transitions = nst
        
        filename = f'{avg}//T_{nst}.npy'
        T1 = np.load(filename)
    
        start_time = time.time()
        num_rows = 5
        num_cols = 9
        num_states = num_rows * num_cols
        num_actions = 4
        start_state = (0,4)
        goal_states = [(0, 0), (0, 8), (4, 4)]
        
        rewards = np.ones((num_rows, num_cols)) * -0.0001
        rewards[0, 0] = 1.0
        rewards[0, 8] = 1.0
        rewards[4, 4] = 1.2
        
        learning_rate = 0.1
        discount_factor = gamma
        epsilon_initial = 1
        epsilon_decay = 0.999
        epsilon_min = 0.01
        max_steps = 30
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        bound_update_freq = 100  # How often to update bounds
        
        def state_to_index(state):
            row, col = state
            return row * num_cols + col
        
        def index_to_state(index):
            row = index // num_cols
            col = index % num_cols
            return row, col
        
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.choice(num_actions)
            else:
                best_actions = np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                return np.random.choice(best_actions)
        
        def test_q(e=30):
            episode_rewards = []
            for episode in range(e):
                state = start_state
                total_reward = 0
                step = 0
                while state not in goal_states and step < max_steps:
                    step += 1
                    action = np.argmax(Q[state_to_index(state)])
                    next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                    next_state = index_to_state(next_state_index)
                    reward = rewards[next_state[0], next_state[1]]
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
            return np.mean(episode_rewards)
        
        def clipped_q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = start_state
            step = 1
            
            # Initialize bounds (as in Algorithm 2)
            L = np.full((num_states, num_actions), -np.inf)
            U = np.full((num_states, num_actions), np.inf)
            
            # Initialize reward table and dynamics table
            rewards_table = np.zeros((num_states, num_actions))
            dynamics_table = np.zeros((num_states, num_actions, num_states))
            
            while step < N_steps + 1:
                if state in goal_states:
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    state = start_state
                
                # Take action and observe transition
                state_idx = state_to_index(state)
                action = epsilon_greedy_policy(state, epsilon)
                next_state_index = np.random.choice(num_states, p=T1[state_idx, action, :])
                next_state = index_to_state(next_state_index)
                reward = rewards[next_state[0], next_state[1]]
                
                # Update reward and dynamics tables
                rewards_table[state_idx, action] = reward
                dynamics_table[state_idx, action, next_state_index] += 1
                
                # Update bounds periodically
                if step % bound_update_freq == 0:
                    # Normalize dynamics table
                    dyn = dynamics_table.reshape(num_states * num_actions, num_states).T
                    for i in range(dyn.shape[1]):
                        if np.sum(dyn[:,i]) > 0:
                            dyn[:,i] /= np.sum(dyn[:,i])
                    dyn = dyn.T.reshape(num_states, num_actions, num_states)
                    
                    # Compute bounds using current estimates
                    # Update this line in clipped_q_learning:
                    L, U = compute_bounds(Q, dyn, rewards_table, discount_factor)
                
                # Compute TD target
                current_q = Q[state_idx, action]
                max_next_q = np.max(Q[next_state_index])
                td_target = reward + discount_factor * max_next_q
                
                # Compute new Q-value without clipping
                new_q = current_q + learning_rate * (td_target - current_q)
                
                # Clip the Q-value to stay within bounds if they exist
                if L[state_idx, action] > -np.inf and U[state_idx, action] < np.inf:
                    Q[state_idx, action] = np.clip(new_q, L[state_idx, action], U[state_idx, action])
                else:
                    Q[state_idx, action] = new_q
                
                state = next_state
                step += 1
                
                if step % test_steps == 0:
                    tr = test_q()
                    episode_rewards.append(tr)
                    
            return episode_rewards
        
        num_runs = 1
        N_steps = 18000
        test_steps = 12
        num_episodes = int(N_steps/test_steps)
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        
        def compute_q_values(S, A, R, T, gamma, terminal_state):
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
        # # Compute Q-values
        # q_p1 = compute_q_values(S, A, R, T1, gamma, terminal_state)
        
        # # # Compute Q-values
        # q_p2 = compute_q_values(S, A, R2, T1, gamma, terminal_state)
        for run in range(num_runs):
            Q =  np.zeros((num_states, num_actions)) #q_p1 + q_p2
            episode_rewards = clipped_q_learning(N_steps, test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
            
        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//clipped_QL_{x_percentage}_{number_stochastic_transitions}.csv')
        
        data.append((avg, f'{x_percentage}_{number_stochastic_transitions}', end_time-start_time))

pd.DataFrame(data, columns=['Run', "Domain info", "clipped_QL"]).to_csv("Data_clipped_QL.csv")