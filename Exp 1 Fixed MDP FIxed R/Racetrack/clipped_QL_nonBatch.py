import numpy as np
from scipy.sparse import csr_matrix
import time
import pandas as pd

# Environment setup (unchanged)
WORLD = np.array([
    ["G", "_", "_", "_", "_", "X", "X"],
    ["G", "_", "_", "_", "_", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["X", "X", "X", "X", "X", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_"],
    ["S", "_", "_", "_", "X", "X", "X"]
])

S = WORLD.size
A = 7
STATES = range(WORLD.size)
STATE2WORLD = {i: (i//7, i%7) for i in range(49)}
START = 42
GOALS = [0, 7]
WALLS = [5,6,14,15,21,22,23,24,25,28,29,46,47,48]

# Reward definitions (unchanged)
CRASH = -10.
WIN = 100.
STEP = -1.

ACTIONS = [
    (0, 0), 
    (-1, 0), (-2, 0),
    (0, 1), (0, 2),
    (0, -1), (0, -2),
]

# Initialize reward tensors (unchanged)
R = np.zeros((S,A,S))
R2 = np.zeros((S,A,S))
R3 = np.zeros((S,A,S))
for s in range(S):
    for a in range(A):
        for sdash in range(S):
            if sdash in GOALS:
                R3[s,a,sdash] = 2
            if sdash not in WALLS and sdash not in GOALS:
                R2[s,a,sdash] = 0.2
                R3[s,a,sdash] = -0.3
            if sdash in WALLS:
                R2[s,a,sdash] = -0.5
                R3[s,a,sdash] = 0.3
            if sdash == 42:
                R[s,a,sdash] = 3
                R3[s,a,sdash] = -4
r = R + R2 + R3

# Transition function (unchanged)
def transition(state, action):
    dx, dy = ACTIONS[action]
    next_state = (STATE2WORLD[state][0] + dx, STATE2WORLD[state][1] + dy)
    state_idx = next_state[0]*7 + next_state[1]
    if (state_idx in WALLS or 
        next_state[0] < 0 or next_state[0] >= 7 or 
        next_state[1] < 0 or next_state[1] >= 7):
        return state_idx, False
    return state_idx, True

def neighbours(s):
    n = []
    for a in range(7):
        ns = transition(s, a)
        if ns[1]:
            n.append(ns[0])
        else:
            if ns[0] in STATE2WORLD:
                n.append(ns[0])
            else:
                n.append(s)
    return list(set(n))

# Initialize transition matrix (unchanged)
T = np.zeros((49,7,49))
for s in range(49):
    if s in WALLS or s in GOALS:
        continue
    reachable = neighbours(s)
    ns = []
    for i in reachable:
        if i in STATE2WORLD:
            ns.append(i)
        else:
            ns.append(s)
    for a in range(7):
        nxt_s = list(transition(s, a))
        if nxt_s[0] not in STATE2WORLD:
            nxt_s[0] = s
        for sdash in range(49):
            if nxt_s[0] == sdash:
                T[s,a,sdash] = 0.88
            elif sdash in ns:
                T[s,a,sdash] = 0.12/(len(ns)-1)
            else:
                T[s,a,sdash] = 0
        if np.sum(T[s,a]) != 1:
            T[s,a,nxt_s[0]] += 1 - np.sum(T[s,a])

# New clipped Q-learning functions
def compute_bounds(Q, dynamics_table, rewards_table, gamma):
    """Model-free bound computation"""
    S, A = Q.shape
    
    # Compute V values (max Q)
    V = np.max(Q, axis=1)
    
    # Reshape dynamics and convert to sparse
    dyn_flat = dynamics_table.reshape(S*A, S)
    dyn_sparse = csr_matrix(dyn_flat.T)
    
    # Compute Qj = T*V
    Qj = dyn_sparse.T.dot(V).reshape(S, A)
    
    # Compute delta_rwd = r + γQj - Q
    delta_rwd = rewards_table + gamma * Qj - Q
    
    # Handle infinite/nan values
    finite_mask = np.isfinite(delta_rwd)
    if not np.all(finite_mask):
        finite_vals = delta_rwd[finite_mask]
        delta_rwd[~finite_mask] = np.mean(finite_vals) if len(finite_vals) > 0 else 0
    
    # Compute min/max delta
    delta_min = np.min(delta_rwd[finite_mask]) if np.any(finite_mask) else 0
    delta_max = np.max(delta_rwd[finite_mask]) if np.any(finite_mask) else 0
    
    # Compute bounds
    lb = Q + delta_rwd + gamma * delta_min / (1 - gamma)
    ub = Q + delta_rwd + gamma * delta_max / (1 - gamma)
    
    # Clip to reasonable values
    lb = np.maximum(lb, -10 / (1 - gamma))
    ub = np.minimum(ub, 100 / (1 - gamma))
    
    return lb, ub

def clipped_q_learning(N_steps, test_steps, T1, gamma):
    """Main training loop with clipped Q-updates"""
    global Q
    
    # Initialize tables for model-free learning
    rewards_table = np.zeros((S, A))
    dynamics_table = np.zeros((S, A, S))
    
    epsilon = epsilon_initial
    episode_rewards = []
    state = START
    
    # Initialize bounds
    L = np.full((S, A), -np.inf)
    U = np.full((S, A), np.inf)
    
    for step in range(1, N_steps + 1):
        if state in WALLS or state in GOALS:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = START
        
        action = epsilon_greedy_policy(state, epsilon)
        next_state = np.random.choice(S, p=T1[state, action, :])
        reward = r[state, action, next_state]
        
        # Update model-free tables
        rewards_table[state, action] = reward
        dynamics_table[state, action, next_state] += 1
        
        # Update bounds periodically
        if step % bound_update_freq == 0:
            # Normalize dynamics table
            dyn = dynamics_table.copy()
            for s in range(S):
                for a in range(A):
                    if np.sum(dyn[s,a]) > 0:
                        dyn[s,a] /= np.sum(dyn[s,a])
            L, U = compute_bounds(Q, dyn, rewards_table, gamma)
        
        # Standard Q-update
        target = reward + gamma * np.max(Q[next_state])
        td_error = target - Q[state, action]
        new_q = Q[state, action] + learning_rate * td_error
        
        # Clip to bounds if they exist
        if L[state, action] > -np.inf and U[state, action] < np.inf:
            Q[state, action] = np.clip(new_q, L[state, action], U[state, action])
        else:
            Q[state, action] = new_q
        
        state = next_state
        
        if step % test_steps == 0:
            episode_rewards.append(test_q())
    
    return episode_rewards

# Helper functions (unchanged)
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(A)
    return np.argmax(Q[state])

def test_q(e=30):
    episode_rewards = []
    for _ in range(e):
        state = START
        total_reward = 0
        step = 0
        while state not in WALLS and state not in GOALS and step < max_steps:
            step += 1
            action = np.argmax(Q[state])
            next_state = np.random.choice(S, p=T1[state, action, :])
            total_reward += r[state, action, next_state]
            state = next_state
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards)

# Parameters (unchanged)
learning_rate = 0.1
discount_factor = 0.88
epsilon_initial = 1.0
epsilon_decay = 0.998
epsilon_min = 0.01
max_steps = 30
bound_update_freq = 100  # How often to update bounds

# Main execution
data = []
init = int(input("init: "))
last = int(input("end: "))

for avg in range(init, last):
    for nst in range(1, 8, 2):
        filename = f'{avg}//T_{nst}.npy'
        T1 = np.load(filename)
        
        start_time = time.time()
        Q = np.ones((S, A)) * -0.5  # Initial Q-values
        
        N_steps = 28000
        test_steps = 4
        
        # Run clipped Q-learning
        rewards_run = np.zeros((1, N_steps // test_steps))
        rewards_run[0] = clipped_q_learning(N_steps, test_steps, T1, discount_factor)
        
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//ClippedQL_0_{nst}.csv')
        data.append((avg, f'0_{nst}', end_time - start_time))

pd.DataFrame(data, columns=["Run", "Domain info", "ClippedQL"]).to_csv(f"Data_ClippedQL_{last}.csv")