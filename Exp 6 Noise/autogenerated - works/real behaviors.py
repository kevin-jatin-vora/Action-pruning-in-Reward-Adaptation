from scipy import stats
import random
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.chdir(os.getcwd())

def read_mdp(mdp):
    """Function to read MDP file"""
    f = open(mdp)

    S = int(f.readline())
    A = int(f.readline())

    # Initialize Transition and Reward arrays
    R = np.zeros((S, A, S))
    R2 = np.zeros((S, A, S))
    T = np.zeros((S, A, S))

    # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R[s][a][sPrime] = line[sPrime]
                
    # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R2[s][a][sPrime] = line[sPrime]
    
    # Update the Transition Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                T[s][a][sPrime] = line[sPrime]
                 
    gamma = float(f.readline().rstrip())
    f.close()
    return S, A, R, R2, T, gamma

S, A, R, R2, T, gamma = read_mdp("gridworld_mdp(r1+r2).txt")
gamma = 0.939 #0.94 #0.97

R[31,1,40] = 0.6
R2[31,1,40] = 0.6
R[41,2,40] = 0.6
R2[41,2,40] = 0.6
R[39,3,40] = 0.6
R2[39,3,40] = 0.6
T1 = T.copy()
terminal_state = [0, 8, 40]

# Helper function: epsilon-greedy action selection for Q_star and Q_mu
def epsilon_greedy(Q_star, Q_mu, state, epsilon):
    """Epsilon-greedy action selection from a mix of Q_star and Q_mu policies."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, A)  # Random action
    else:
        # Select one action from Q_star using argmax and another from Q_mu using argmin
        action_star = np.argmax(Q_star[state])  # Action that maximizes Q_star
        action_mu = np.argmin(Q_mu[state])  # Action that minimizes Q_mu
        
        # Randomly return one of these two actions
        return np.random.choice([action_star, action_mu], p=[0.5, 0.5])

# Helper function: standard epsilon-greedy action selection
def epsilon_greedy_og(Q, state, epsilon):
    """Standard epsilon-greedy action selection."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, A)  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action

# Function to test policy
def test_policy(Q, episodes, max_steps, r):
    """Tests a greedy policy based on Q values (max Q)."""
    total_return = 0
    for _ in range(episodes):
        state = 4  # Start from the initial state
        episode_return = 0
        steps = 0
        while state not in terminal_state and steps < max_steps:
            steps += 1
            action = np.argmax(Q[state])  # Greedy action
            next_state = np.random.choice(range(S), p=T1[state, action])
            reward = r[state, action, next_state] # Use R for testing, as per original code
            episode_return += reward
            state = next_state
        total_return += episode_return
    return total_return / episodes

def test_policy_mu(Q, episodes, max_steps, r):
    """Tests a greedy policy based on Q values (min Q)."""
    total_return = 0
    for _ in range(episodes):
        state = 4  # Start from the initial state
        episode_return = 0
        steps = 0
        while state not in terminal_state and steps < max_steps:
            steps += 1
            action = np.argmin(Q[state])  # Greedy action
            next_state = np.random.choice(range(S), p=T1[state, action])
            reward = r[state, action, next_state] # Use R2 for testing
            episode_return += reward
            state = next_state
        total_return += episode_return
    return total_return / episodes

# Confidence Interval Function
def confidence_interval(data, confidence=0.95):
    """Calculates the confidence interval for a given data set."""
    n = len(data)
    if n < 2:
        return 0, 0
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h

# Improved Plotting Function
def plot_with_t_ci(avg_returns, label, color, window_size, l, test_steps):
    """Plots the average returns with a t-based confidence interval."""
    moving_avg = []
    lower_ci = []
    upper_ci = []

    # Trim data based on l
    data = avg_returns[:l]
    
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        mean = np.mean(window)
        low, high = confidence_interval(window)
        moving_avg.append(mean)
        lower_ci.append(low)
        upper_ci.append(high)

    x_range = np.arange(len(moving_avg)) * test_steps

    plt.plot(x_range, moving_avg, label=label, color=color, linewidth=2)
    plt.fill_between(x_range, lower_ci, upper_ci, color=color, alpha=0.3)

def compute_behaviors(behavior_id, r, N_steps, max_steps, num_runs, test_steps, alpha, epsilon_initial, epsilon_decay, epsilon_min, T1, S, A, terminal_state, avg, nst):
    """
    Learns or loads Q-tables and other data for a specific behavior.
    """
    folder_path = f'D:/TMLR 2025/Exp 1 Fixed MDP FIxed R - Copy/Dollar-Euro/{avg}/behaviors_{behavior_id}_{nst}'
    os.makedirs(folder_path, exist_ok=True)
    
    q_og_file = os.path.join(folder_path, 'Q_og.npy')
    q_star_file = os.path.join(folder_path, 'Q_star.npy')
    q_mu_file = os.path.join(folder_path, 'Q_mu.npy')
    returns_og_file = os.path.join(folder_path, 'returns_og.npy')
    returns_star_file = os.path.join(folder_path, 'returns_star.npy')
    returns_mu_file = os.path.join(folder_path, 'returns_mu.npy')
    next_state_dict_file = os.path.join(folder_path, 'next_states.pkl')
    memorized_r_file = os.path.join(folder_path, 'memorized_R.npy')
    
    if (os.path.exists(q_og_file) and os.path.exists(q_star_file) and os.path.exists(q_mu_file) and 
        os.path.exists(returns_og_file) and os.path.exists(returns_star_file) and os.path.exists(returns_mu_file) and
        os.path.exists(next_state_dict_file) and os.path.exists(memorized_r_file)):
        
        print(f"Loading pre-trained data for behavior {behavior_id}...")
        Q_og = np.load(q_og_file)
        Q_star = np.load(q_star_file)
        Q_mu = np.load(q_mu_file)
        avg_returns_Q_og = np.load(returns_og_file)
        avg_returns_Q_star = np.load(returns_star_file)
        avg_returns_Q_mu = np.load(returns_mu_file)
        
        with open(next_state_dict_file, 'rb') as f:
            next_state_dict = pickle.load(f)

        memorized_R1 = np.load(memorized_r_file)
        print("Data loaded successfully.")
        return avg_returns_Q_og, avg_returns_Q_star, avg_returns_Q_mu

    else:
        print(f"Training models for behavior {behavior_id}...")
        avg_returns_Q_og_all_runs = []
        avg_returns_Q_star_all_runs = []
        avg_returns_Q_mu_all_runs = []
        
        memorized_R1 = np.zeros((S, A, S))
        next_state_dict = {}

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            Q_og = np.zeros((S, A))
            Q_star = np.zeros((S, A))
            Q_mu = np.zeros((S, A))

            test_avg_returns_Q_star = []
            test_avg_returns_Q_mu = []
            
            # Training and testing for Q_star and Q_mu
            epsilon = epsilon_initial
            state = 4
            steps = 0
            done = False
            ep_step = 0
            while steps < N_steps + 1:
                ep_step += 1
                if done:
                    done = False
                    ep_step = 0
                    epsilon = max(epsilon_min, epsilon * epsilon_decay)
                    while state in terminal_state:
                        state = 4
                steps += 1
                action = epsilon_greedy(Q_star, Q_mu, state, epsilon)
                next_state = np.random.choice(range(S), p=T1[state, action])
                reward = r[state, action, next_state]
                
                # Update next state dictionary
                if (state, action) not in next_state_dict:
                    next_state_dict[(state, action)] = set()
                next_state_dict[(state, action)].add(next_state)

                memorized_R1[state, action, next_state] = reward
                
                # Compute importance sampling ratios
                a_star = np.argmax(Q_star[state])
                a_mu = np.argmin(Q_mu[state])

                # Probability under behavior policy
                b_action = (epsilon / A) + ((1 - epsilon) * 0.5 if action in [a_star, a_mu] else 0)
                
                # Probability under target policy for Q_star
                pi_star_action = 1 if action == a_star else 0
                pi_star_action = (epsilon / A) + (1 - epsilon if action == a_star else 0)
                w_star = pi_star_action / b_action

                # Probability under target policy for Q_mu
                pi_mu_action = (epsilon / A) + (1 - epsilon if action == a_mu else 0)
                w_mu = pi_mu_action / b_action
                
                # Update Q_star and Q_mu using weighted updates
                best_next_action = np.argmax(Q_star[next_state])
                Q_star[state, action] += w_star * alpha * (reward + gamma * Q_star[next_state, best_next_action] - Q_star[state, action])

                worst_next_action_value = np.min(Q_mu[next_state])
                Q_mu[state, action] += w_mu * alpha * (reward + gamma * worst_next_action_value - Q_mu[state, action])

                state = next_state
                if state in terminal_state or ep_step > max_steps:
                    done = True
            
                if steps % test_steps == 0:
                    avg_return_Q_star = test_policy(Q_star, 1, max_steps, r)
                    test_avg_returns_Q_star.append(avg_return_Q_star)
                    avg_return_Q_mu = test_policy_mu(Q_mu, 1, max_steps, r)
                    test_avg_returns_Q_mu.append(avg_return_Q_mu)

            # input(epsilon)

            # Training and testing for Q_og
            epsilon = epsilon_initial
            state = 4
            steps = 0
            done = False
            ep_step = 0
            test_avg_returns_Q_og = []
            while steps < N_steps + 1:
                ep_step += 1
                if done:
                    done = False
                    ep_step = 0
                    epsilon = max(epsilon_min, epsilon * epsilon_decay)
                    while state in terminal_state:
                        state = 4
                steps += 1
                action = epsilon_greedy_og(Q_og, state, epsilon)
                next_state = np.random.choice(range(S), p=T1[state, action])
                reward = r[state, action, next_state]
            
                best_next_action = np.argmax(Q_og[next_state])
                Q_og[state, action] += alpha * (reward + gamma * Q_og[next_state, best_next_action] - Q_og[state, action])
                state = next_state
                if state in terminal_state or ep_step > max_steps:
                    done = True
            
                if steps % test_steps == 0:
                    avg_return_Q_og = test_policy(Q_og, 1, max_steps, r)
                    test_avg_returns_Q_og.append(avg_return_Q_og)

            # Append results for this run
            avg_returns_Q_og_all_runs.append(test_avg_returns_Q_og)
            avg_returns_Q_star_all_runs.append(test_avg_returns_Q_star)
            avg_returns_Q_mu_all_runs.append(test_avg_returns_Q_mu)

        # Calculate the average returns across all runs
        avg_returns_Q_og = np.mean(avg_returns_Q_og_all_runs, axis=0)
        avg_returns_Q_star = np.mean(avg_returns_Q_star_all_runs, axis=0)
        avg_returns_Q_mu = np.mean(avg_returns_Q_mu_all_runs, axis=0)
        
        # Save Q-tables, returns, and next-state dictionary
        np.save(q_og_file, Q_og)
        np.save(q_star_file, Q_star)
        np.save(q_mu_file, Q_mu)
        np.save(returns_og_file, avg_returns_Q_og)
        np.save(returns_star_file, avg_returns_Q_star)
        np.save(returns_mu_file, avg_returns_Q_mu)
        np.save(memorized_r_file, memorized_R1)
        
        with open(next_state_dict_file, 'wb') as f:
            pickle.dump(next_state_dict, f)

        print("Training complete and data saved.")
        return avg_returns_Q_og, avg_returns_Q_star, avg_returns_Q_mu

# --- Main Execution ---
# Define Q-learning parameters
learning_rate = 0.1
epsilon_initial = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
N_steps = 18000
test_steps = 12
num_runs = 1
max_steps = 30
num_episodes = int(N_steps/test_steps)
alpha = learning_rate

init = int(input("start: "))
last = int(input("end: "))
for avg in range(init,last):
    for nst in range(1,5):
        terminal_state=[0,8,40]
        filename = f'{avg}//T_{nst}.npy'
        T1=np.load(filename)
        # T1[terminal_state]=0
        # Run for Behavior 1 (R)
        avg_returns_Q_og_b1, avg_returns_Q_star_b1, avg_returns_Q_mu_b1 = compute_behaviors(1, R, N_steps, max_steps, num_runs, test_steps, alpha, epsilon_initial, epsilon_decay, epsilon_min, T1, S, A, terminal_state, avg, nst)
        # Run for Behavior 2 (R2)
        avg_returns_Q_og_b2, avg_returns_Q_star_b2, avg_returns_Q_mu_b2 = compute_behaviors(2, R2, N_steps, max_steps, num_runs, test_steps, alpha, epsilon_initial, epsilon_decay, epsilon_min, T1, S, A, terminal_state, avg, nst)
        
        
        # --- PLOTTING ---
        # Shared plotting parameters
        l = 4000
        window_size = 100
        # Plot for Behavior 1 (R)
        plt.figure(figsize=(10, 6))
        plot_with_t_ci(avg_returns_Q_og_b1, "Q* (epsilon-greedy)", "blue", window_size, l, test_steps)
        plot_with_t_ci(avg_returns_Q_star_b1, "Q* (modified epsilon-greedy)", "green", window_size, l, test_steps)
        plot_with_t_ci(avg_returns_Q_mu_b1, "Q_mu (modified epsilon-greedy)", "red", window_size, l, test_steps)
        plt.xlabel("Steps", fontsize=16)
        plt.ylabel("Average Return", fontsize=16)
        plt.title("Behavior 1 (R) Learning Curve", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='center right', fontsize=14, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{avg}/behavior_1_learning_curve_{nst}.png", bbox_inches='tight', dpi=300)
        plt.show()
        # input()
        
        
        # Plot for Behavior 2 (R2)
        plt.figure(figsize=(10, 6))
        plot_with_t_ci(avg_returns_Q_og_b2, "Q* (epsilon-greedy)", "blue", window_size, l, test_steps)
        plot_with_t_ci(avg_returns_Q_star_b2, "Q* (modified epsilon-greedy)", "green", window_size, l, test_steps)
        plot_with_t_ci(avg_returns_Q_mu_b2, "Q_mu (modified epsilon-greedy)", "red", window_size, l, test_steps)
        plt.xlabel("Steps", fontsize=16)
        plt.ylabel("Average Return", fontsize=16)
        plt.title("Behavior 2 (R2) Learning Curve", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='center right', fontsize=14, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{avg}/behavior_2_learning_curve_{nst}.png", bbox_inches='tight', dpi=300)
        plt.show()
        
        
