## figure 2.6 p. 42, section 2.10

import matplotlib.pyplot as plt
import os

os.getcwd()
os.chdir('/home/james/CompSci/Projects/ReinforcementLearning')
os.listdir()

results = [[], [], [], []]
with open('output_rl.txt', 'r') as simfile:
    linecount = 0
    for line in simfile:
        data = line.split()
        for elt in data:
            results[linecount].append(float(elt))
        linecount += 1

results
## epsg, ucb, gba, oiv

## plot so far

def maxvalue(inputList):
    return max([max(sublist) for sublist in inputList])

def minvalue(inputList):
    return min([min(sublist) for sublist in inputList])

plotstep_espg = [-7, -6, -5, -4, -3, -2]
plotstep_ucb = [-4, -3, -2, -1, 0, 1, 2]
plotstep_gba = [-5, -4, -3, -2, -1, 0, 1, 2]
plotstep_oiv = [-2, -1, 0, 1, 2]

plt.plot(plotstep_espg, results[0], color="red")
plt.plot(plotstep_ucb, results[1], color="blue")
plt.plot(plotstep_gba, results[2], color="green")
plt.plot(plotstep_oiv, results[3], color="black")
plt.legend(('Epsilon Greedy', 'Upper Conf Bnd', 
            'Gradient Bandit', 'Optimistic Initial Value'))
plt.xlim(-7, 2)
plt.ylim(minvalue(results)-0.1, maxvalue(results)+0.1 )
plt.savefig('Problem0211.jpg')
##plt.show()

## figure 2.6 recreated!!!
## took 570.057 + 583.642 + 956.095 + 431.349 = 2541.143 seconds
## or 42.352 minutes.  
## if I want to code up the longer exercise, I should use F90
