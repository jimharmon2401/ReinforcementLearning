break ## heh, heh

## problem 4.9 - gambler's problem

import numpy as np
import matplotlib.pyplot as plt

probH = 0.55

def compute_probs(newst, rew, oldst, act):
    if (newst > oldst):
        return probH
    elif (newst < oldst):
        return 1-probH
    else:
        return 1

states = [i for i in range(0,101)]
rewards = [0,1]
values = [0 for i in range(0, 101)]
gamma = 1 ## non-discounted
policies = [0 for i in range(0, 101)]

theta = 0.001
delta = 1 ## to start it off
while delta > theta:
##for i in range(50):
    delta = 0
    for state in states[1:100]:
        v = values[state]
        alist = []
        for a in range(1,min(state, 100-state) + 1):
            tempval = 0
            poss_states = [state - a, state + a]
            for newstate in poss_states:
                reward = int(newstate/100) ## 0/1
                tempval = tempval + \
                          compute_probs(newstate, reward, state, a) * \
                          (reward + gamma * values[newstate])
            alist.append(tempval)
        values[state] = np.max(alist)
        policies[state] = np.argmax(alist) + 1
        delta = max(delta, abs(v - values[state]))
    print(delta)

## plot policies
plt.plot(states, policies)

## p = 0.25 looks similar to final policy for 0.4 in book
## and it's pretty stable
## p = 0.55, the optimal policies start out more aggressive, but
## end up with "just bet 1" each time as theta goes to zero
