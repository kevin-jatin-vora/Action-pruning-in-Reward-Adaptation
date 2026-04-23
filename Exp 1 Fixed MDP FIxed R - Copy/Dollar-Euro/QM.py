from scipy import stats
import random
import os
import numpy as np
#np.random.seed(6)
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time
import pickle
os.chdir(os.getcwd())

def read_mdp(mdp):

    """Function to read MDP file"""
    #mdp="mdp_new.txt"
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
            #print((s,a))
            for sPrime in range(S):
                #print(line[sPrime], end=" ")
                T[s][a][sPrime] = line[sPrime]

            #print()

    # Read the value of gamma
    gamma = float(f.readline().rstrip())
    #terminal_state=int((f.readline().rstrip()))

    f.close()

    return S, A, R, R2, T, gamma#,terminal_state

S, A, R, R2, T, gamma = read_mdp("gridworld_mdp(r1+r2).txt")
R[31,1,40] = 0.6
R2[31,1,40] = 0.6
R[41,2,40] = 0.6
R2[41,2,40] = 0.6
R[39,3,40] = 0.6
R2[39,3,40] = 0.6


def make_transitions_stochastic(T1, x_percentage):
    S, A, _ = T1.shape

    # Calculate the number of states to remain deterministic
    num_deterministic_states = int(S * x_percentage/100)

    # Randomly select which states will remain deterministic
    deterministic_states = np.random.choice(S, num_deterministic_states, replace=False)

    # Initialize modified transition probability matrix
    T_stochastic = np.zeros_like(T1)

    # Modify transition probabilities for deterministic states
    for s in deterministic_states:
        for a in range(A):
            max_prob_index = np.argmax(T1[s, a])
            T_stochastic[s, a, max_prob_index] = 1

    # Modify transition probabilities for stochastic states
    stochastic_states = np.setdiff1d(np.arange(S), deterministic_states)
    for s in stochastic_states:
        #print(s)
        for a in range(A):
            T_stochastic[s,a] = T1[s,a]
            # max_prob_index = np.argmax(T1[s, a])
            # probabilities = np.random.dirichlet(np.ones(S))
            # T_stochastic[s, a] = probabilities
            # if np.argmax(T_stochastic[s, a]) != max_prob_index:
            #     T_stochastic[s, a, max_prob_index] = np.max(T_stochastic[s, a])
            #     T_stochastic[s, a, np.argmax(T_stochastic[s, a])] = T_stochastic[s, a, max_prob_index]
    return T_stochastic


def restrict_transition(matrix, max_bf):
    S, A, S_prime = matrix.shape

    # Sample n uniformly from [1, max_bf] for each state
    n_values = np.random.randint(1, max_bf + 1, size=S)
    print(n_values.mean())
    # Identify top n states for each action
    # top_n_indices = np.argsort(matrix, axis=2)[:, :, -n_values[:, None]]
    sorted_matrix = np.argsort(matrix, axis=2)

    # Create a mask to zero out probabilities for states not in top n
    mask = np.zeros_like(matrix)
    for s in range(S):
        for a in range(A):
            mask[s, a, sorted_matrix[s, a, -n_values[s]:]] = 1

    # Apply mask
    restricted_matrix = matrix * mask

    # Normalize probabilities for the top n states for each action
    row_sums = restricted_matrix.sum(axis=2, keepdims=True)
    normalized_matrix = restricted_matrix / row_sums

    return normalized_matrix

combined_dict = {}
def load_and_combine_transitions(avg, nst):
    """Loads and combines the next state dictionaries from both behaviors."""
    folder_path_b1 = os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_1_{nst}')
    folder_path_b2 = os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_2_{nst}')

    dict_b1_file = os.path.join(folder_path_b1, 'next_states.pkl')
    dict_b2_file = os.path.join(folder_path_b2, 'next_states.pkl')

    with open(dict_b1_file, 'rb') as f:
        dict_b1 = pickle.load(f)
    with open(dict_b2_file, 'rb') as f:
        dict_b2 = pickle.load(f)

    global combined_dict
    for s in range(S):
        for a in range(A):
            key = (s, a)
            set1 = dict_b1.get(key, set())
            set2 = dict_b2.get(key, set())
            combined_dict[key] = list(set1.union(set2))
    # return combined_dict

def transition(current_state, action):
    """Returns a list of next states based on the combined dictionary."""
    global combined_dict
    key = (current_state, action)
    return combined_dict.get(key, [])


data=[]
init = int(input("start: "))
last = int(input("end: "))
for avg in range(init,last):
    # if(not os.path.isdir(str(avg))):
    #     os.mkdir(str(avg))
    for nst in range(1,5):
        # np.random.seed(avg)
        # T1 = T.copy()
        gamma = 0.939 #0.94 #0.97
        # T1[T1 == 0.7] = 0.856 #0.88
        # T1[T1 == 0.1] = 0.052 #0.04
        terminal_state=[0,8,40]
        x_percentage = 0#40 #80
        # T1 = make_transitions_stochastic(T1, x_percentage)
        number_stochastic_transitions = nst #A
        # T1 = restrict_transition(T1, number_stochastic_transitions)
        # T1[terminal_state,:, :] = 0

        filename = f'{avg}//T_{nst}.npy'
        T1=np.load(filename)
        load_and_combine_transitions(avg,nst)

        # Load the pre-computed Q-values
        q_p1 = np.load(os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_1_{nst}', 'Q_star.npy'))
        q_p2 = np.load(os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_2_{nst}', 'Q_star.npy'))
        q_m1 = np.load(os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_1_{nst}', 'Q_mu.npy'))
        q_m2 = np.load(os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_2_{nst}', 'Q_mu.npy'))

        # Load and sum the memorized rewards
        memorized_R1 = np.load(os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_1_{nst}', 'memorized_R.npy'))
        memorized_R2 = np.load(os.path.join(f'D:\\TMLR 2025\\Exp 1 Fixed MDP FIxed R - Copy\\Dollar-Euro\\{avg}\\behaviors_2_{nst}', 'memorized_R.npy'))
        real_r = memorized_R1 + memorized_R2



        start_time = time.time()


        Q = q_p1 + q_p2
        # Q_UB = Q.copy()
        Q[terminal_state,:] = 0
        for i in range(5000):
            if(i>0):
                U = Q_k.copy()
                Udash = Q.copy()
            Q_k=Q.copy()
            for s in range(S):
                if(s in terminal_state):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition(s,a):
                            temp.append(real_r[s,a,sdash] + gamma*np.max(Q_k[sdash]))
                        # if(temp!=[]):
                        Q[s,a] =min(Q_k[s,a],max(temp)) #max(temp)
            # if(i>0):
            #     if(np.round(np.max(np.abs(Q_k-Q)),7)> np.round(gamma*(np.max(np.abs(U-Udash))),7)):
            #         # print(np.max(np.abs(Q_k-Q)))
            #         # print(gamma*(np.max(np.abs(U-Udash))))
            #         print(i)
            #         # input()
            if(np.max(np.abs(Q-Q_k))<0.0000000001):
                print(i)
                print("------------")
                break

        Qm = np.zeros((S,A))
        o1 = q_p1 + q_m2
        o2 = q_p2 + q_m1

        for s in range(S):
            for a in range(A):
                Qm[s,a]= max(o1[s,a], o2[s,a])


        Qm[terminal_state,:] = 0
        for i in range(5000):
            if(i>0):
                U = Qm_k.copy()
                Udash = Qm.copy()
            Qm_k=Qm.copy()
            for s in range(S):
                if(s in terminal_state):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition(s,a):
                            temp.append(real_r[s,a,sdash] + gamma*np.max(Qm_k[sdash]))
                        # if(temp!=[]):
                        Qm[s,a] = max(Qm_k[s,a],min(temp)) #min(temp)
            # if(i>0):
            #     if(np.max(np.abs(Qm_k-Qm))> gamma*(np.max(np.abs(U-Udash)))):
            #         # print("lowerbound")
            #         print(i)
            #         # input()
            if(np.max(np.abs(Qm-Qm_k))<0.0000000001):
                print(i)
                print("------------")
                break

        # Qm = np.round(Qm,2)

        info=[]
        final_actions=set(list(range(A)))
        prune={}
        state_action={}


        for i in range(S):
            alist=[]
            for action_l in range(A):
                for action_u in range(A):
                    if(action_l==action_u):
                        continue
                    if( Qm[i, action_l]-Q[i,action_u] > 1e-12 ):
                        info.append((i,action_l, action_u))
                        alist.append(action_u)
            prune[i]= set(alist)
            state_action[i]= final_actions.difference(set(alist))
            if(state_action[i]==set()):
                state_action[i] = final_actions

        # print(S*A-sum([len(state_action[i]) for i in state_action.keys()]))

        # input()
        ########################################################################################
        rr=R+R2

        # Define environment parameters
        num_rows = 5
        num_cols = 9
        num_states = num_rows * num_cols
        num_actions = 4
        start_state = (0,4)
        goal_states = [(0, 0), (0, 8), (4, 4)]

        # Define rewards
        rewards = np.ones((num_rows, num_cols))*-0.0001
        rewards[0, 0] = 1.0
        rewards[0, 8] = 1.0
        rewards[4, 4] = 1.2

        # Define Q-learning parameters
        learning_rate = 0.1
        discount_factor = gamma #0.95
        epsilon_initial = 1.0
        epsilon_decay = 0.999 #[0.999,0.999, 0.9995, 0.999]
        epsilon_min = 0.01

        max_steps=30
        # Define actions: down, up, left, right
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Initialize Q-table
        Q = np.zeros((num_states, num_actions))

        # Convert state from (row, col) to index
        def state_to_index(state):
            row, col = state
            return row * num_cols + col

        # Convert index to state (row, col)
        def index_to_state(index):
            row = index // num_cols
            col = index % num_cols
            return row, col

        # Epsilon-greedy policy
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                # Exploration: Choose a random action from the available set
                return np.random.choice(list(state_action[state_to_index(state)]))
            else:
                # Exploitation: Choose a best action from the available set, breaking ties randomly
                state_index = state_to_index(state)
                q_values = Q[state_index, list(state_action[state_index])]
                best_action_indices = np.where(q_values == np.max(q_values))[0]
                best_action_index = np.random.choice(best_action_indices)
                bas = list(state_action[state_index])
                best_action = bas[best_action_index]
                return best_action

        def test_q(e=30):
            global Q
            episode_rewards = []
            for episode in range(e):
                state = start_state
                total_reward = 0
                step = 0
                while state not in goal_states and step<max_steps:
                    step+=1
                    action = epsilon_greedy_policy(state, 0)
                    # if(nst>=5):
                        # action = np.argmax(Q[state_to_index(state)])
                    # else:
                        # best_actions = np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                        # if(len(best_actions)>1):
                            # try:
                                # action = np.random.choice(list(set(state_action[state_to_index(state)]).intersection(set(best_actions))))
                            # except:
                                # action = np.random.choice(list(state_action[state_to_index(state)]))
                        # else:
                            # action=best_actions[0]
                    next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                    next_state = index_to_state(next_state_index)
                    reward = rewards[next_state[0], next_state[1]]
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)

            return np.mean(episode_rewards)

        def q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = start_state
            step = 1

            while step < N_steps+1:
                if state in goal_states:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)

                    # Reset to the initial state if the agent reaches a terminal state
                    state = start_state

                action = epsilon_greedy_policy(state, epsilon)
                next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                next_state = index_to_state(next_state_index)
                reward = rewards[next_state[0], next_state[1]]

                # Update Q-value
                Q[state_to_index(state), action] += learning_rate * (
                    reward + discount_factor * np.max(Q[state_to_index(next_state)]) - Q[state_to_index(state), action]
                )

                state = next_state
                step += 1

                # Track reward for each step
                # episode_rewards.append(reward)

                # Decay epsilon
                # epsilon = max(epsilon * epsilon_decay, epsilon_min)

                # Optional: Print or test the Q-values periodically
                if step % test_steps == 0:
                    tr = test_q()
                    # for k in tr:
                    episode_rewards.append(tr)

            return episode_rewards

        # Run multiple episodes and average results
        num_runs = 1
        N_steps=18000
        test_steps = 12
        num_episodes = int(N_steps/test_steps)
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs,num_episodes))
        for run in range(num_runs):
            Q = np.zeros((S,A))
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)

        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//ours_{x_percentage}_{number_stochastic_transitions}.csv')

        # w=150
        # # Plot average Q value per episode over 5 runs
        # plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
        # plt.xlabel('step')
        # plt.ylabel('Average Reward')
        # plt.title(f'Average Reward per 3000 steps over {num_runs} Runs')
        # plt.show()
        # # input()

        xy_s={}
        for i in range(5):
            for j in range(9):
                xy_s[i*9 + j]=(i,j)

        heat_map=np.zeros((5,9))
        for key in state_action.keys():
            heat_map[xy_s[key]]=len(state_action[key])


        cmap =  'Blues' #sns.cm.flare
        ax = sns.heatmap(heat_map, linewidth=0.5, linecolor='black', cmap=cmap, alpha=0.6)
        ax.invert_yaxis()
        plt.savefig(f"{avg}//heatmap_DE_{x_percentage}_{number_stochastic_transitions}.png",bbox_inches = 'tight', dpi=1000)
        plt.show()
        # input()
        data.append((avg, f"Dollar-Euro_{x_percentage}_{number_stochastic_transitions}", S, A, S*A-sum([len(state_action[i]) for i in state_action.keys()]), end_time-start_time))
pd.DataFrame(data, columns=['Run', 'Domain', '|S|', '|A|', 'Actions Pruned', 'QM']).to_csv(f"Data_RA_{last-1}.csv")
f = open("test_ep.txt", "w")
f.write(f"{test_steps}")
f.close()
