#!/usr/bin/env python3

import math
from math import gamma
import itertools
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
alpha = 0.5
gamma = 0.5


state_features_child_performance = ['none', 'correct', 'incorrect', 'no-respond']
state_features_child_engagement = ['none', 'on-task', 'off-task']
state_features_history_of_the_last_action = ['none', 'edible', 'social']

product_performance_engagement = list(itertools.product(state_features_child_performance, state_features_child_engagement))
all_feature_state = list(itertools.product(product_performance_engagement, state_features_history_of_the_last_action))

all_states = ['O','A','B','C','D','E','F','G','H','I','J','K','L','M','N','X','Y','Z']

# O = START, X = GOAL1, Y = GOAL2, Z = GOAL3


# tab = np.array(all_states)
# # reshaped_tab = tab.reshape(6,6)
# print(len(all_states), all_states,tab)


def mag(x): 
    return math.sqrt(sum(i**2 for i in x))

all_actions = ['none', 'e', 's'] # e: Edible, s: Social

episodes = [[['O', 'e', 'M', 0],
             ['M', 's', 'N', 0],
             ['N', 's', 'K', +2],
             ['K', 's', 'Z', +4]],
            [['O', 's', 'I', +2],
             ['I', 's', 'K', +2],
             ['K', 'e', 'E', 0],
             ['E', 'e', 'D', 0],
             ['D', 'e', 'G', -2],
             ['G', 's', 'E', 0],
             ['E', 's', 'B', +1],
             ['B', 's', 'K', +2],
             ['K', 's', 'Z', +4]],
            [['O', 'e', 'C', 0],
             ['C', 'e', 'G', -2],
             ['G', 's', 'D', 0],
             ['D', 's', 'K', +2],
             ['K', 'e', 'H', -2],
             ['H', 's', 'M', 0],
             ['M', 's', 'E', 0],
             ['E', 's', 'K', +2],
             ['K', 's', 'Z', +4]],
            [['O', 's', 'L', 0],
             ['L', 'e', 'E', 0],
             ['E', 's', 'A', +1],
             ['A', 'e', 'H', -2],
             ['H', 's', 'J', +2],
             ['J', 's', 'K', +2],
             ['K', 's', 'Z', +4]],
            [['O', 's', 'I', +2],
             ['I', 's', 'B', +1],
             ['B', 's', 'Z', +4]]]

Qmax = {}
Q = {}
for state in all_states:
    for action in all_actions:
        Qmax[state,action] = 0
        Q[state,action] = 0
        # print(state, action)

k = 0
while True:
    for episode in episodes:
        for samples in episode:
            Old_Q = Q.copy()
            for state in samples[0]:
                for action in samples[1]:
                    for successor in samples[2]:
                        reward = samples[3]
                        # print(reward)
                        Q[state,action] = (1-alpha)*Old_Q[state,action] + alpha*(reward + gamma * Qmax[successor,action])
                        for act in all_actions:
                          # print(act)
                          if Q[state,act] > Qmax[state,action]:
                              Qmax[state,action] = Q[state,act]
                    q_values = list(Q.values())
                    old_q_values = list(Old_Q.values())

                    difference  = list()
                    for values, old_values in zip(q_values,old_q_values):
                        difference.append(values-old_values)
                    
    k += 1
    if mag(difference) == 0:
        break
    print(k)
print(Q)
