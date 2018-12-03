import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
import scipy.linalg as lalg
###
# Excercise 3
###

p = np.linspace(0, 1, num=10**4)
allPassengerProb = p**100
passengerProb = np.exp(-2*(10000*(0.95-p))**(2) / 10000)
outcome = allPassengerProb*passengerProb
bound = max(outcome)
worstcase_p = p[np.argsort(outcome)[-1]]
print("Worst Case p:")
print(worstcase_p)
print("Bound")
print(bound)

f, (ax1) = plt.subplots(1,1)

ax1.plot(p, outcome)
ax1.axvline(worstcase_p, color='red')
ax1.set_xlabel('p')
ax1.set_ylabel('bound')

plt.savefig('ex3.png')
plt.close()
