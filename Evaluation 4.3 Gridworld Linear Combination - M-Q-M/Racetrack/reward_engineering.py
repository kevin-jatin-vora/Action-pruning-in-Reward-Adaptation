# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:52:20 2024

@author: kvora1
"""
import pandas as pd
R1=np.zeros((S,A,S))
R2=np.zeros((S,A,S))
R3=np.zeros((S,A,S))
for s in range(S):
    for a in range(A):
        for sdash in range(S):
            if(sdash in GOALS):
                R3[s,a,sdash] = 2
            if(sdash not in WALLS and sdash not in GOALS):
                R2[s,a,sdash] = 0.2#0.9
                R3[s,a,sdash] = -0.3#-1
            if (sdash in WALLS):
                R2[s,a,sdash] = -0.5#-2.5
                R3[s,a,sdash] = 0.3#2.3
            if(sdash==42):
                R1[s,a,sdash] = 3
                R3[s,a,sdash] = -4
# for s in range(S):
#     for a in range(A):
#         for sdash in range(S):
#             if(sdash in GOALS):
#                 R3[s,a,sdash] = 2
#             if(sdash not in WALLS and sdash not in GOALS):
#                 R2[s,a,sdash] = 0.9
#                 R3[s,a,sdash] = -1
#             if (sdash in WALLS):
#                 R2[s,a,sdash] = -2.5
#                 R3[s,a,sdash] = 2.3
#             if(sdash==42):
#                 R1[s,a,sdash] = 20
#                 R3[s,a,sdash] = -21
R = R1+R2+R3    
q_p1 = compute_q_values(S, A, R1, T, 0.9)
q_p2 = compute_q_values(S, A, R2, T, 0.9)
q_p3 = compute_q_values(S, A, R3, T, 0.9)
q_p = compute_q_values(S, A, R, T, 0.9)

combined_Q = np.zeros((S, A))

# Iterate over each state-action pair
for s in range(S):
    for a in range(A):
        # Find maximum Q-value across both Q-tables for state s and action a
        max_Q_value = max(q_p1[s, a], q_p2[s, a], q_p3[s,a])
        
        # Assign the maximum Q-value to the combined Q-table
        combined_Q[s, a] = max_Q_value
        
df=pd.DataFrame([np.argmax(q_p, axis=1), np.argmax(combined_Q, axis=1)])
df=df.T
df["uncommon"] = df[0]==df[1]
df_d = df.drop(WALLS, axis=0)

print(len(df_d[df_d['uncommon']==False]))

# diff1 = df.iloc[0] - df.iloc[2]
# diff2 = df.iloc[1] - df.iloc[2]

# temp=pd.concat([diff1, diff2], axis=1, ignore_index=True)

# temp['new_col'] = (temp[0] == 0) | (temp[1] == 0)

# # Sum the 'new_col'
# print(len(temp[temp['new_col']==False]))