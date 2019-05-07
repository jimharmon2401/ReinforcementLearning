#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:02:34 2019

@author: james
"""

## example 6.2, p. 125, Random Walk
from scipy.stats import bernoulli

alpha = 0.05
gamma = 1 ## undiscounted
states = [i for i in range(7)]
num_episodes = 1000
values = [0.5 for i in range(7)]
values[0] = 0
values[6] = 0

for i in range(num_episodes):
    state = 3
    reward = 0
    while state < 6 and state > 0:
        action = 2*bernoulli.rvs(0.5) - 1 # size = 1
        newstate = state + action
        if newstate == 6:
            reward = 1
        values[state] = values[state] + alpha*(
                reward + gamma*values[newstate] - values[state])
        state = newstate

print(values[1:6])