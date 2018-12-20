## figure 2.6 p. 42, section 2.10

import numpy as np
import matplotlib.pyplot as plt
import time
import math

def karm_epsg(q_s, numsteps, epsilon):
    Q = np.zeros(len(q_s))
    n = np.zeros(len(q_s))
    rewards = np.zeros(numsteps)
    for step in range(numsteps):
        ## select max, splitting ties by random
        prob = np.random.uniform(0.0, 1.0, 1)
        if prob < epsilon:
            A = np.random.choice([i for i in range(len(Q))])
        elif np.max(Q) == 0:
            maxval = max(Q)
            maxinds = []
            for index in range(len(Q)):
                if maxval == Q[index]:
                    maxinds.append(index)
            ## select action A at random from maxinds
            A = np.random.choice(maxinds)
        else:
            A = np.argmax(Q)
        ## generate random reward
        R = np.random.normal(q_s[A], 1.0, 1)
        n[A] = n[A] + 1
        Q[A] = Q[A] + (1.0 / n[A]) * (R - Q[A])
        rewards[step] = R
    return rewards

def karm_oivg(q_s, numsteps, Q0, alpha = 0.1): ## epsilon == 0
    Q = np.zeros(len(q_s)) + Q0
    ##n = np.zeros(len(q_s))
    rewards = np.zeros(numsteps)
    for step in range(numsteps):
        maxval = max(Q)
        maxinds = []
        for index in range(len(Q)):
            if maxval == Q[index]:
                maxinds.append(index)
        ## select action A at random from maxinds
        A = np.random.choice(maxinds)
        ## generate random reward
        R = np.random.normal(q_s[A], 1.0, 1)
        ##n[A] = n[A] + 1
        Q[A] = Q[A] + alpha * (R - Q[A])
        rewards[step] = R
    return rewards

def karm_ucb(q_s, numsteps, c): ## epsilon == 0
    Q = np.zeros(len(q_s))
    n = np.zeros(len(q_s))
    rewards = np.zeros(numsteps)
    for step in range(len(q_s)):
        R = np.random.normal(q_s[step], 1.0, 1)
        n[step] = n[step] + 1
        Q[step] = Q[step] + (1.0 / n[step]) * (R - Q[step])
    for step in range(len(q_s), numsteps):
        ## select action A at random from maxinds
        A = np.argmax(np.array(Q) + c * np.sqrt(np.log(step)/np.array(n)))
        ## generate random reward
        R = np.random.normal(q_s[A], 1.0, 1)
        n[A] = n[A] + 1
        Q[A] = Q[A] + (1.0 / n[A]) * (R - Q[A])
        rewards[step] = R
    return rewards

def karm_gba(q_s, numsteps, alpha): ## epsilon == 0
    H = np.zeros(len(q_s))
    probs = np.exp(H) / np.sum(np.exp(H))
    rewards = np.zeros(numsteps)
    Rbar = 0
    for step in range(numsteps):
        ## select action A at random from maxinds
        A = np.argmax(np.random.multinomial(1, probs))
        ## generate random reward
        R = np.random.normal(q_s[A], 1.0, 1)
        ## update Rbar
        Rbar = Rbar + (1.0 / (step+1)) * (R - Rbar)
        ## update Hs
        temp = H[A] ## fixed!
        H = H - alpha * (R - Rbar) * probs
        H[A] = temp + alpha * (R - Rbar)
        ## update probs
        probs = np.exp(H) / np.sum(np.exp(H))
        ## record reward
        rewards[step] = R
    return rewards

simlength = 2000
steplen = 1000

t0 = time.time()

eps_g = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
avgRepsg = []
np.random.seed(343)

for eps in eps_g:
    avgrewepsg = np.zeros(simlength)
    for simstep in range(simlength):
        q_stars = np.random.normal(0, 1, 10)
        avgrewepsg[simstep] = np.mean(karm_epsg(q_stars, steplen, eps))
    avgRepsg.append(np.mean(avgrewepsg))

t1 = time.time()
print(t1-t0)
print("First one done")
avgRepsg ## 591 seconds, 5:51


simlength = 2000
steplen = 1000

t0 = time.time()
cs = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
avgRucb = []
np.random.seed(343)

for cee in cs:
    avgrew = np.zeros(simlength)
    for simstep in range(simlength):
        q_stars = np.random.normal(0, 1, 10)
        avgrew[simstep] = np.mean(karm_ucb(q_stars, steplen, cee))
    avgRucb.append(np.mean(avgrew))

t1 = time.time()
print(t1-t0)
print("Second one done")
avgRucb ## 


simlength = 2000
steplen = 1000

t0 = time.time()
alphas = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
avgRgba = []
np.random.seed(343)

for alph in alphas:
    avgrew = np.zeros(simlength)
    for simstep in range(simlength):
        q_stars = np.random.normal(0, 1, 10)
        avgrew[simstep] = np.mean(karm_gba(q_stars, steplen, alph))
    avgRgba.append(np.mean(avgrew))

t1 = time.time()
print(t1-t0)
print("Third one done")
avgRgba ## 

simlength = 2000
steplen = 1000

t0 = time.time()
qs = [1/4, 1/2, 1, 2, 4]
avgRoiv = []
np.random.seed(343)

for q in qs:
    avgrew = np.zeros(simlength)
    for simstep in range(simlength):
        q_stars = np.random.normal(0, 1, 10)
        avgrew[simstep] = np.mean(karm_oivg(q_stars, steplen, q, 0.1))
    avgRoiv.append(np.mean(avgrew))

t1 = time.time()
print(t1-t0)
print("Fourth one done")
avgRoiv ## 


## plot so far

plotstep_espg = [-7, -6, -5, -4, -3, -2]
plotstep_ucb = [-4, -3, -2, -1, 0, 1, 2]
plotstep_gba = [-5, -4, -3, -2, -1, 0, 1, 2]
plotstep_oiv = [-2, -1, 0, 1, 2]

plt.plot(plotstep_espg, avgRepsg, color="red")
plt.plot(plotstep_ucb, avgRucb, color="blue")
plt.plot(plotstep_gba, avgRgba, color="green")
plt.plot(plotstep_oiv, avgRoiv, color="black")
plt.xlim(-7, 2)
plt.ylim(1.0, 1.5)
plt.show()

## figure 2.6 recreated!!!
## took 570.057 + 583.642 + 956.095 + 431.349 = 2541.143 seconds
## or 42.352 minutes.  
## if I want to code up the longer exercise, I should use F90






7.99241811*10**-2 + 8.92123580*10**-2 + 0.152211264 + 0.117816746 + \
8.12356174*10**-2 + 9.22161117*10**-2 + 7.07703084*10**-2 + 0.137211546 + \
8.04605708*10**-2 + 9.89413410*10**-2

7.99418315*10**-2 + 8.92236009*10**-2 + 0.152132660 + 0.117797211 + \
8.12524706*10**-2 + 9.22249109*10**-2 + 7.08437636*10**-2 + 0.137161657 + \
8.04779008*10**-2 + 9.89439860*10**-2

6.67471522*10**-6 + 7.02442776*10**-6 + 6.70086774*10**-6 + 0.999938250 + \
6.60645810*10**-6 + 7.03318528*10**-6 + 7.53177073*10**-6 + 6.65596917*10**-6 +\
5.90775608*10**-6 + 7.50738172*10**-6

6.67471522*10**-6 + 7.02442776*10**-6 + 6.70086774*10**-6 + 0.9999 + \
6.60645810*10**-6 + 7.03318528*10**-6 + 7.53177073*10**-6 + 6.65596917*10**-6 +\
5.90775608*10**-6 + 7.50738172*10**-6