import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
#np.random.seed(1)


class FrozenLake:
    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.start = (0, 0)
        self.goal = (size - 1, 1)
        self.hole_prob = 0.1
        self.holes = []
        self.generate_holes()

    def generate_holes(self, num_pairs=4):
        pairs = set()
        while len(pairs) < num_pairs:
          pair = (np.random.randint(0, self.size), np.random.randint(0, self.size))
          if pair not in pairs and pair != self.goal and pair != self.start:
            pairs.add(pair)
        self.holes = list(pairs)
        for x,y in self.holes:
            self.grid[x, y] = -1

    def is_valid_position(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_reward1(self, state):
        x, y = state
        if (x, y) == self.holes[0] or (x, y) == self.holes[2]:
            return 1
        elif(x, y) == self.holes[1] or (x, y) == self.holes[3]:
            return -1
        elif (x, y) == self.goal:
            return 0.6
        else:
            return -0.01  # For all other states

    def get_reward2(self, state):
        x, y = state
        if (x, y) == self.holes[1] or (x, y) == self.holes[3]:
            return 1
        elif(x, y) == self.holes[0] or (x, y) == self.holes[2]:
            return -1
        elif (x, y) == self.goal:
            return 0.6
        else:
            return -0.01  # For all other states

    def get_reward(self, state):
        x, y = state
        if (x, y) == self.goal:
            return 0.6*2  # Goal reached
        elif self.grid[x, y] == -1:
            return 0  # Fell into a hole
        else:
            return -0.02  # For all other states

    def get_neighboring_states(self, state):
        x, y = state
        neighbors = []
        for action in ['up', 'down', 'left', 'right']:
            new_x, new_y = self.transition(state, action)
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        return neighbors

    def transition(self, state, action):
        x, y = state
        if action == 'up':
            x -= 1
        elif action == 'down':
            x += 1
        elif action == 'left':
            y -= 1
        elif action == 'right':
            y += 1
        if self.is_valid_position(x, y):
            return x, y
        return state

def display_state(env, agent_location):
    state_grid = np.empty((env.size, env.size), dtype=object)

    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == agent_location:
                state_grid[i, j] = 'A'  # Agent's current position
            elif (i, j) == env.start:
                state_grid[i, j] = 'S'  # Agent's start position
            elif (i, j) == env.goal:
                state_grid[i, j] = 'G'  # Goal position
            elif env.grid[i, j] == -1:
                state_grid[i, j] = 'H'  # Hole
            else:
                state_grid[i, j] = '-'  # Empty tile

    return state_grid


env = FrozenLake(size=6)  # Change size as needed
total_states = env.size * env.size
T = np.zeros((total_states, 4, total_states))

def state_to_xy(state, size=env.size):
    x = state // size
    y = state % size
    return x, y



def get_env(w_filename, t_filename):
    gridworld = np.load(w_filename)
    env = FrozenLake(size=6)
    env.grid = gridworld
    env.size = gridworld.shape[0]
    env.start = (0, 0)
    env.goal = (env.size - 1, 1)
    a=np.where(env.grid==-1)
    env.holes = list(zip(a[0],a[1]))
    holes = env.holes
    terminal_state = holes + [env.goal]
    T = np.load(t_filename)
    return env,T, terminal_state

data = []
init=int(input("enter start index: "))
till=int(input("enter end index: "))
for avg in range(init,till): #for avg in range(30):
    for nst in range(1, 5):
        env, T1, terminal_state = get_env(f"{10}//env_{nst}.npy", f"{10}//T_{nst}.npy")
        ts = [state[0] * env.size + state[1] for state in terminal_state]
        x_percentage = 0
        number_stochastic_transitions = nst
        gamma = 0.9
        start_time = time.time()
        T1[ts, :, :] = 0
        gamma = 0.9
        start_time = time.time()
        Q1 = np.load(f'{avg}//Q1_{nst}.npy')
        Q2 = np.load(f'{avg}//Q2_{nst}.npy')

        def evaluate(env, q, t):
            Q1_e = q.copy()#np.zeros(q.shape)
            for i in range(1000):
                for s in range(total_states):
                    a =np.argmax(q[s])
                    p =t[s,a]
                    ns = np.ravel_multi_index(env.transition((s // env.size, s % env.size), ['up', 'down', 'left', 'right'][a]), (env.size, env.size))
                    r= [env.get_reward(state_to_xy(i))  for i in range(total_states)]#np.zeros(total_states)
                    # r[ns] = env.get_reward(state_to_xy(ns))
                    Q1_e[s,a] = np.sum(p *(r + gamma*np.max(Q1_e, axis=1)))
            return Q1_e
        Q1_e = evaluate(env, Q1, T1)
        Q2_e = evaluate(env, Q2, T1)
        # Initialize combined Q-table
        combined_Q = np.zeros(Q1.shape)


        # Iterate over each state-action pair
        for s in range(total_states):
            for a in range(4):
                # Find maximum Q-value across both Q-tables for state s and action a
                max_Q_value = max(Q1_e[s, a], Q2_e[s, a])

                # Assign the maximum Q-value to the combined Q-table
                combined_Q[s, a] = max_Q_value
        # print("###################################")
        # print(np.argmax(combined_Q,axis=1))
        # print(np.argmax(Qstar,axis=1))
        # print("###################################")


        def epsilon_greedy_policy(state, epsilon, Q):
            if np.random.rand() < epsilon:
                return np.random.choice(range(len(Q[state])))
            else:
                return np.argmax(Q[state])

        def test_q(Q,e=30):
            # global Q
            episode_rewards=[]
            for episode in range(e):
                state = 0
                total_reward = 0
                step = 0
                while state not in ts and step<max_steps:
                    step+=1
                    action = np.argmax(Q[state])
                    next_state = np.random.choice(total_states, p=T1[state, action, :])
                    reward = env.get_reward((next_state // env.size, next_state % env.size))
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
            return np.mean(episode_rewards)
        # def test_q(Q, e=1):
        #     # global Q
        #     for episode in range(e):
        #         state = 0
        #         total_reward = 0
        #         step = 0
        #         while state not in ts and step<max_steps:
        #             step+=1
        #             action = np.argmax(Q[state])
        #             next_state = np.random.choice(total_states, p=T1[state, action, :])
        #             reward = env.get_reward((next_state // env.size, next_state % env.size))
        #             total_reward += reward
        #             state = next_state
        #     return total_reward

        def q_learning(N_steps, test_steps, env, learning_rate, discount_factor, epsilon_initial, epsilon_decay, epsilon_min, num_episodes):
            num_actions = 4
            global combined_Q
            Q = combined_Q.copy()#np.zeros((total_states, num_actions))
            epsilon = epsilon_initial
            episode_rewards = []
            state = 0
            step = 1

            while step < N_steps+1:
                if state in ts:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    # Reset to the initial state if the agent reaches a terminal state
                    state = 0

                action = epsilon_greedy_policy(state, epsilon,Q)
                next_state = np.random.choice(total_states, p=T1[state, action, :])
                reward = env.get_reward((next_state // env.size, next_state % env.size))
                # Update Q-value
                Q[state, action] += learning_rate * (
                    reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
                )

                state = next_state
                step += 1

                if step % test_steps == 0:
                    tr = test_q(Q)
                    episode_rewards.append(tr)
            return episode_rewards



        # Define Q-learning parameters
        learning_rate = 0.1
        discount_factor = 0.9
        epsilon_initial = 1.0
        epsilon_decay = 0.9998
        epsilon_min = 0.01
        # num_episodes = 2000
        max_steps = 20

        N_steps=4800*5
        test_steps = 8 #12
        num_episodes = int(N_steps/test_steps)

        # Run multiple episodes and average results
        num_runs = 30
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))

        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):
            #np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps,env, learning_rate, discount_factor, epsilon_initial, epsilon_decay, epsilon_min, num_episodes)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)

        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//SFQL_{x_percentage}_{number_stochastic_transitions}.csv')
        # Plot average Q value per episode over 5 runs
        # window_size = 100
        # plt.plot(range(num_episodes - window_size + 1), np.convolve(average_rewards, np.ones(window_size), 'valid') / window_size)
        # plt.xlabel('Episode')
        # plt.ylabel('Average Reward')
        # plt.title(f'Average Reward per Episode over {num_runs} Runs')
        # plt.show()
        data.append((avg, f'{x_percentage}_{number_stochastic_transitions}', end_time - start_time))
pd.DataFrame(data, columns=["Run", "Domain info", "SFQL"]).to_csv(f"Data_SFQL_{till-1}.csv")
