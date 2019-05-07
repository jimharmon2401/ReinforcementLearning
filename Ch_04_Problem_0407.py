
## Jack's car rental problem
## Two location, 0-20 cars each
## rents cars at $10 each
## Can move up to five cars from one location to another at -$2 a car
## Rental and returns are Poisson RV's

## STATES: number of cars at each location at end of day
## ACTIONS: net numbers of cars moved between two locations overnight
## REWARD: cost of moving cars PLUS rentals the next day

## Location 1: rent request at lambda = 3, return at lambda = 3
## Location 2: rent request at Lambda = 4, return at lambda = 2
## gamma = 0.9

## state is number of cars at each location at end of day
## actions are net numbers of cars moved overnight 
## from location 1 to location 2
## location 1: new state = old state - carsreq + carsret - action
## location 2: new state = old state - carsreq + carsret + action

import time
import numpy as np
from scipy.stats import poisson

## set up parameters of problem
gamma = 0.9
maxcars = 20
maxmove = 5
numstates = maxcars + 1

probsLreq1 = poisson.pmf(range(numstates),3)
probsLreq2 = poisson.pmf(range(numstates),4)
probsLret1 = poisson.pmf(range(numstates),3)
probsLret2 = poisson.pmf(range(numstates),2)

## revise last probabilities to reflect sum? 
probsLreq1[-1] = 1 - sum(probsLreq1[:maxcars])
probsLreq2[-1] = 1 - sum(probsLreq2[:maxcars])
probsLret1[-1] = 1 - sum(probsLret1[:maxcars])
probsLret2[-1] = 1 - sum(probsLret2[:maxcars])

## tolerance for policy evaluation step
theta = 0.001

## initialize policy and value matrices
## index represents number of cars at fist and second locations
## index runs 0 - 20 = maxcars; 21 = numstates
policy = np.zeros((numstates, numstates), dtype = np.int8)
values = np.zeros((numstates, numstates))
policyrange = [i for i in range(-5,6)]


def getP(probvec, index, ismax):
    if ismax:
        return np.log(sum(probvec[index:]))
    else:
        return np.log(probvec[index])


#######################
## Policy Evaluation
#######################
## assume policies is constant
delta = 1 ## to kick off while loop
print(time.asctime(time.localtime()))
time0 = time.time()

while  delta > theta:
    delta = 0
    for cars1 in range(numstates):
        for cars2 in range(numstates):
            v = values[cars1, cars2]
            newval = 0
            ## set action based on policy
            action = policy[cars1, cars2]
            ## compute next morning "state"
            cars1nm = cars1 - action
            cars2nm = cars2 + action
            ## can only grant requests to as many cars as are there
            for carsreq1 in range(0, (cars1nm + 1)):
                cars1eod = cars1nm - carsreq1
                ## if at max, then set value to True for getP
                cars1ismax = (carsreq1 == cars1nm)
                carsreq1p = getP(probsLreq1, carsreq1, cars1ismax)
                ## maximum number of returns is 20 - the number of cars left
                maxret1 = maxcars - cars1eod
                ## can only grant requests to as many cars as are there
                for carsreq2 in range(0, (cars2nm + 1)):
                    cars2eod = cars2nm - carsreq2
                    cars2ismax = (carsreq2 == cars2nm)
                    carsreq2p = getP(probsLreq2, carsreq2, cars2ismax)
                    maxret2 = maxcars - cars2eod
                    ## can compute reward now
                    totrew = 10 * (carsreq1 + carsreq2) - 2 * abs(action)
                    ## loop through returns
                    for carsret1 in range(maxret1 + 1):
                        newcars1 = cars1eod + carsret1
                        cars1ismax = (carsret1 == maxret1)
                        carsret1p = getP(probsLret1, 
                                         carsret1, cars1ismax)
                        for carsret2 in range(maxret2 + 1):
                            newcars2 = cars2eod + carsret2
                            cars2ismax = (carsret2 == maxret2)
                            carsret2p = getP(probsLret2, 
                                             carsret2, cars2ismax)
                            newval = newval + \
                            np.exp(carsreq1p + carsreq2p + \
                                   carsret1p + carsret2p) * \
                                   (totrew + \
                                    gamma * policy[newcars1, newcars2])
            delta = max(delta, abs(v - newval))
            values[cars1, cars2] = newval
    print(delta)
    print(time.asctime(time.localtime()))

time1 = time.time()
print(time1-time0)
## fist run 455 seconds, 7.5 minutes

#######################
## Policy Improvement
#######################
## way too long
print(time.asctime(time.localtime()))
time0 = time.time()
policy_stable = True
for oldstate1 in range(numstates):
    for oldstate2 in range(numstates):
        old_action = policy[oldstate1, oldstate2]
        newpolicy = [0 for i in range(11)] ## temp holder for arg max
        maxaction = min(5, 20 - oldstate2, oldstate1)
        minaction = max(-5, -1*oldstate2, oldstate1 - 20)
        for newaction in range(minaction, maxaction + 1):
            temp = 0
            ## compute max and min rewards with this action/state
            maxreward = (oldstate1 + oldstate2) * 10 - 2 * abs(newaction)
            minreward = -2 * abs(newaction)
            cars1nm = oldstate1 - newaction
            cars2nm = oldstate2 + newaction
            for rewards in range(minreward, maxreward + 1, 10):
                ## need total cars requested and fulfilled for this reward
                carsreqd = int((rewards - minreward) / 10)
                ## carsreqd1 is at most cars1 or carsreqd
                for carsreqd1 in range(0, min(cars1nm,carsreqd)+1):
                    cars1max = (carsreqd1 == min(cars1nm, carsreqd))
                    carsreq1p = getP(probsLreq1, carsreqd1, cars1max)
                    ## num of cars reqd from 2, no checks yet
                    carsreqd2 = carsreqd - carsreqd1
                    carsleft1 = cars1nm - carsreqd1
                    carsleft2 = cars2nm - carsreqd2
                    if carsleft2 < 0:
                        continue ## carsreq2p = 0
                    elif carsleft2 == 0:
                        carsreq2p = getP(probsLreq2, carsreqd2, True)
                    else:
                        carsreq2p = getP(probsLreq2, carsreqd2, False)
                    for carsret1 in range(0, (numstates - carsleft1)):
                        cars1max = (carsret1 == (maxcars - carsleft1))
                        carsret1p = getP(probsLret1, carsret1, cars1max)
                        for carsret2 in range(0, (numstates - carsleft2)):
                            cars2max = (carsret2 == (maxcars - carsleft2))
                            carsret2p = getP(probsLret2, carsret2, cars2max)
                            newstate1 = carsleft1 + carsret1 
                            newstate2 = carsleft2 + carsret2
                            temp = temp + \
                            np.exp(carsreq1p + \
                            carsreq2p + \
                            carsret1p + \
                            carsret2p) * \
                            (rewards + gamma * values[newstate1, newstate2])
            newpolicy[newaction + 5] = temp
        policy[oldstate1, oldstate2] = np.argmax(newpolicy) - 5
        if old_action != policy[oldstate1, oldstate2]:
            policy_stable = False

print(policy_stable)

time1 = time.time()
print(time1 - time0) ## 772 seconds
policy

#######################################################
########################################################
########################################################
## CDOE GRAVEYARD

## compute probabilities
def compute_probs(loc1, loc2, action):
    '''
    given num cars at locations 1 and 2, and overnight action
    compute the probabilities of next states and rewards
    return all relevant values
    '''
    ## location 1, location 2, number of total rewards
    stateprobs = np.zeros((numstates, numstates, numstates + numstates - 1))
    loc1state = loc1 - action
    loc2state = loc2 - action
    for reward1 in range(loc1state + 1):
        if reward1 == loc1state:
            ## anything requests above what is there result in 
            ## no addditional reward, so sum probs
            prob1req = sum(probsLreq1[reward1:])
        else:
            prob1req = probsLreq1[reward1]
        loc1new = loc1state - reward1
        for reward2 in range(loc2state + 1):
            if reward2 == loc2state:
                prob2req = sum(probsLreq2[reward2:])
            else:
                prob2req = probsLreq2[reward2]
            loc2new = loc2state - reward2
            totreward = reward1 + reward2
            for state1 in range(loc1new,numstates):
                ## to achieve the state maxcars, you can 
                ## have up to possibly infinite returns
                if state1 == maxcars:
                    prob1ret = sum(probsLret1[(state1 - loc1new):]
                else: 
                    prob1ret = probsLret1[(state1 - loc1new)]
                for state2 in range(loc2new, numstates):
                    if state2 == maxcars:
                        prob2ret = sum(probsLret2[(state2 - loc2new):])
                    else:
                        prob2ret = probsLret2[(state2 - loc2new)]
                    stateprobs[state1, state2, totreward] +=  / 
                        np.exp(np.log(prob1req) + np.log(prob2req) + /
                        np.log(prob2ret) + np.log(prob2ret))
    return stateprobs





while  delta > theta:
    delta = 0
    for cars1 in range(numstates):
        for cars2 in range(numstates):
            v = values[cars1, cars2]
            newval = 0
            maxreward = (cars1 + cars2) * 10 + policy[cars1, cars2] * -2
            minreward = policy[cars1, cars2] * -2
            for rewards in range(minreward, maxreward+1, 10):
                carsreqd = int((rewards - minreward) / 10)
                for carsreqd1 in range(0, min(cars1,carsreqd)):
                    carsreqd2 = carsreqd - carsreqd1
                    carsleft1 = cars1 - carsreqd1
                    carsleft2 = cars2 - carsreqd2
                    if carsleft2 < 0:
                        continue
                    for carsret1 in range(0, (numstates - carsleft1)):
                        for carsret2 in range(0, (numstates - carsleft2)):
                            newstate1 = cars1 - carsreqd1 + carsret1 - \
                                policy[cars1, cars2]
                            newstate2 = cars2 - carsreqd2 + carsret2 + \
                                policy[cars1, cars2]
                            if (newstate1 < 0 or newstate2 < 0):
                                continue
                            elif(carsret1 == (maxcars - carsleft1) and
                                 carsret2 < (maxcars - carsleft2)):
                                newval = newval + \
                                np.exp(np.log(probsLreq1[carsreqd1]) + \
                                np.log(probsLreq2[carsreqd2]) + \
                                np.log(sum(probsLret1[carsret1:])) + \
                                np.log(probsLret2[carsret2]) ) * \
                                (rewards + gamma *  \
                                 values[min(20,newstate1), 
                                        min(20,newstate2)])
                            elif(carsret1 < (maxcars - carsleft1) and
                                 carsret2 == (maxcars - carsleft2)):
                                newval = newval + \
                                np.exp(np.log(probsLreq1[carsreqd1]) + \
                                np.log(probsLreq2[carsreqd2]) + \
                                np.log(probsLret1[carsret1]) + \
                                np.log(sum(probsLret2[carsret2:])) ) * \
                                (rewards + gamma *  \
                                 values[min(20,newstate1), 
                                        min(20,newstate2)])
                            elif(carsret1 == (maxcars - carsleft1) and
                                 carsret2 == (maxcars - carsleft2)):
                                newval = newval + \
                                np.exp(np.log(probsLreq1[carsreqd1]) + \
                                np.log(probsLreq2[carsreqd2]) + \
                                np.log(sum(probsLret1[carsret1:])) + \
                                np.log(sum(probsLret2[carsret2:])) ) * \
                                (rewards + gamma *  \
                                 values[min(20,newstate1), 
                                        min(20,newstate2)])
                            else:
                                newval = newval + \
                                np.exp(np.log(probsLreq1[carsreqd1]) + \
                                np.log(probsLreq2[carsreqd2]) + \
                                np.log(probsLret1[carsret1]) + \
                                np.log(probsLret2[carsret2]) ) * \
                                (rewards + gamma *  \
                                 values[min(20,newstate1), 
                                        min(20,newstate2)])
            values[cars1, cars2] = newval
            delta = max(delta, abs(v - newval))
    print(delta)
