## reinforcement learning k-arm bandit

import numpy as np
import matplotlib.pyplot as plt
import time

## defind k-arm bandit function for returning
## number of steps, average estimated reward, and the actual rewards.
def karm(q_s, numsteps, epsilon):
    Q = np.zeros(len(q_s))
    n = np.zeros(len(q_s))
    rewards = np.zeros(numsteps)
    actions = np.zeros(numsteps)
    for step in range(numsteps):
        ## select max, splitting ties by random
        bestaction = np.argmax(q_s)
        prob = np.random.uniform(0.0, 1.0, 1)
        if prob < epsilon:
            A = np.random.choice([i for i in range(len(Q))])
        else:
            maxval = max(Q)
            maxinds = []
            for index in range(len(Q)):
                if maxval == Q[index]:
                    maxinds.append(index)
            ## select action A at random from maxinds
            A = np.random.choice(maxinds)
        ## generate random reward
        if bestaction == A:
            actions[step] = 1
        R = np.random.normal(q_s[A], 1.0, 1)
        n[A] = n[A] + 1
        Q[A] = Q[A] + (1.0 / n[A]) * (R - Q[A])
        rewards[step] = R
        q_s = np.array(q_s) + np.random.normal(0, 0.01, len(q_s))
    return (rewards, actions)


def karm_alpha(q_s, numsteps, epsilon, alpha):
    Q = np.zeros(len(q_s))
    n = np.zeros(len(q_s))
    rewards = np.zeros(numsteps)
    actions = np.zeros(numsteps)
    for step in range(numsteps):
        ## select max, splitting ties by random
        bestaction = np.argmax(q_s)
        prob = np.random.uniform(0.0, 1.0, 1)
        if prob < epsilon:
            A = np.random.choice([i for i in range(len(Q))])
        else:
            maxval = max(Q)
            maxinds = []
            for index in range(len(Q)):
                if maxval == Q[index]:
                    maxinds.append(index)
            ## select action A at random from maxinds
            A = np.random.choice(maxinds)
        ## generate random reward
        if bestaction == A:
            actions[step] = 1
        R = np.random.normal(q_s[A], 1.0, 1)
        n[A] = n[A] + 1
        Q[A] = Q[A] + alpha * (R - Q[A])
        rewards[step] = R
        q_s = np.array(q_s) + np.random.normal(0, 0.01, len(q_s))
    return (rewards, actions)




simlength = 500
steplen = 10000
avgreward0 = np.zeros(steplen)
avgreward1 = np.zeros(steplen)
avgreward01 = np.zeros(steplen)

avgaction0 = np.zeros(steplen)
avgaction1 = np.zeros(steplen)
avgaction01 = np.zeros(steplen)

t0 = time.time()
np.random.seed(343)
for simstep in range(simlength):
    q_stars = np.random.normal(0, 1, 10)
    
    reward0, action0 = karm(q_stars, steplen, 0.1)
    reward1, action1 = karm_alpha(q_stars, steplen, 0.1, 0.1)

    avgreward0 = np.array(avgreward0) + np.array(reward0)
    avgreward1 = np.array(avgreward1) + np.array(reward1)

    avgaction0 = np.array(avgaction0) + np.array(action0)
    avgaction1 = np.array(avgaction1) + np.array(action1)


t1 = time.time()
total = t1 - t0
print(total)

## avg over 2000 runs
avgreward0 = np.array(avgreward0) / simlength
avgreward1 = np.array(avgreward1) / simlength

avgaction0 = np.array(avgaction0) / simlength
avgaction1 = np.array(avgaction1) / simlength


## calculate cumulate rewards
avgR0 = avgreward0#np.cumsum(avgreward0) / [i for i in range(1,steplen + 1)]
avgR1 = avgreward1#np.cumsum(avgreward1) / [i for i in range(1,steplen + 1)]

avgA0 = avgaction0#np.cumsum(avgaction0) / [i for i in range(1,steplen + 1)]
avgA1 = avgaction1#np.cumsum(avgaction1) / [i for i in range(1,steplen + 1)]

## compute x-axis numbers
plotstep = [i for i in range(1, steplen + 1)]

## plot - matches plot on p. 29, fig 2.2, top plot
plt.plot(plotstep, avgR0, color="green")
plt.plot(plotstep, avgR1, color="blue")
plt.ylim(0, 2.0)
plt.show()

plt.plot(plotstep, avgA0, color="green")
plt.plot(plotstep, avgA1, color="blue")
plt.ylim(0, 1.0)
plt.show()
