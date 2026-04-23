import numpy as np
from scipy.sparse import csr_matrix
import time
import pandas as pd
from collections import deque, namedtuple
import numpy as np
from scipy.sparse import csr_matrix

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.memory)

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
def compute_bounds_from_batch(Q, batch, gamma):
    """Compute sample-based bounds from transitions"""
    delta_rwd_list = []

    for trans in batch:
        s, a, r, s_next = trans.state, trans.action, trans.reward, trans.next_state
        V = np.max(Q[s_next])
        Q_val = Q[s, a]
        delta = (r + gamma * V - Q_val)
        delta_rwd_list.append((s, a, delta))

    # Extract deltas
    deltas = np.array([d for (_, _, d) in delta_rwd_list])
    delta_min = np.min(deltas)
    delta_max = np.max(deltas)

    # Initialize bounds
    lb = np.full_like(Q, -np.inf)
    ub = np.full_like(Q, np.inf)

    for (s, a, delta) in delta_rwd_list:
        lb[s, a] = Q[s, a] + delta + gamma * delta_min / (1 - gamma)
        ub[s, a] = Q[s, a] + delta + gamma * delta_max / (1 - gamma)

    # Clip to global min/max
    lb = np.maximum(lb, -10 / (1 - gamma))
    ub = np.minimum(ub, 100 / (1 - gamma))

    return lb, ub

def clipped_q_learning(N_steps, test_steps, T1, gamma):
    global Q

    # Initialize Q and replay buffer
    buffer = ReplayBuffer(capacity=10000)
    Q = np.zeros((S, A))
    L = np.full((S, A), -np.inf)
    U = np.full((S, A), np.inf)

    epsilon = epsilon_initial
    episode_rewards = []
    state = START

    for step in range(1, N_steps + 1):
        if state in WALLS or state in GOALS:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = START

        action = epsilon_greedy_policy(state, epsilon)
        next_state = np.random.choice(S, p=T1[state, action, :])
        reward = r[state, action, next_state]

        # Only store and update from valid (non-terminal) states
        if state not in WALLS and state not in GOALS:
            buffer.push(state, action, reward, next_state)

            # Q-learning update
            target = reward + gamma * np.max(Q[next_state])
            td_error = target - Q[state, action]
            new_q = Q[state, action] + learning_rate * td_error

            # Clamp to bounds
            if L[state, action] > -np.inf and U[state, action] < np.inf:
                Q[state, action] = np.clip(new_q, L[state, action], U[state, action])
            else:
                Q[state, action] = new_q
        else:
            target = reward 
            td_error = target - Q[state, action]
            new_q = Q[state, action] + learning_rate * td_error
            Q[state, action] = new_q
            

        state = next_state

        # Periodically compute new bounds
        if step % bound_update_freq == 0 and len(buffer) > 10:
            batch = buffer.sample(min(len(buffer), 256))
            # Filter out terminal samples
            batch = [b for b in batch if b.state not in WALLS and b.state not in GOALS]
            if batch:
                L, U = compute_bounds_from_batch(Q, batch, gamma)

        # Test agent
        if step % test_steps == 0:
            episode_rewards.append(test_q())

    return episode_rewards

# def clipped_q_learning(N_steps, test_steps, T1, gamma):
#     global Q

#     # Initialize Q and replay buffer
#     buffer = ReplayBuffer(capacity=10000)
#     Q = np.zeros((S, A))
#     L = np.full((S, A), -np.inf)
#     U = np.full((S, A), np.inf)

#     epsilon = epsilon_initial
#     episode_rewards = []
#     state = START

#     for step in range(1, N_steps + 1):
#         if state in WALLS or state in GOALS:
#             epsilon = max(epsilon * epsilon_decay, epsilon_min)
#             state = START

#         action = epsilon_greedy_policy(state, epsilon)
#         next_state = np.random.choice(S, p=T1[state, action, :])
#         reward = r[state, action, next_state]

#         # Save transition
#         buffer.push(state, action, reward, next_state)

#         # Update bounds using recent transitions
#         if step % bound_update_freq == 0 and len(buffer) > 10:
#             batch = buffer.sample(min(len(buffer), 256))
#             L, U = compute_bounds_from_batch(Q, batch, gamma)

#         # Standard Q-update
        
#         if(next_state in WALLS or next_state in GOALS):
#             target = reward
#         else:
#             target = reward + gamma * np.max(Q[next_state])
#         td_error = target - Q[state, action]
#         new_q = Q[state, action] + learning_rate * td_error

#         # Clamp to bounds
#         if L[state, action] > -np.inf and U[state, action] < np.inf:
#             Q[state, action] = np.clip(new_q, L[state, action], U[state, action])
#         else:
#             Q[state, action] = new_q

#         state = next_state

#         if step % test_steps == 0:
#             episode_rewards.append(test_q())

#     return episode_rewards

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