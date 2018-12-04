import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
from sklearn.linear_model import LogisticRegression

###
# Exercise 1
###

# bias 1/2
bias = 0.5
# 20 coins
n = 20
# 1 million experiments
N = 1000000

# get randoms from interval [0,1], 1 million times
def simulate(bias, shape) :
    data = scp.random.binomial(1, bias, shape)
    return pd.DataFrame(data)

experiment = simulate(bias, (N, n))

experiment['mean'] = experiment.mean(axis = 1)

freqs = experiment[[0,'mean']].groupby('mean').count().reset_index().rename(columns = {0 : 'freq'})

vector_sum = []
vector_range_i = []

# range from 0.5 until 1
range = list(np.linspace(0.5,1.0,11))
for i in range:
    vector_sum.append((experiment['mean'] >= i).sum() / float(N))
    vector_range_i.append(i)

def hoeffding_bound(mu, alpha, n) :
    epsilon = alpha - mu
    return scp.exp((-2) * n * (epsilon ** 2))

def markov_bound(mu, alpha) :
    if (alpha == 0) : return scp.nan
    else : return mu / alpha

def chebychev_bound(mu, alpha, n) :
	# get rid of division by zero
	epsilon = alpha - mu
	if (epsilon == 0) : return scp.nan
	cheby = (1/n*0.25)/epsilon**2
	# skip outlayers
	if cheby > 1:
		return 1
	else:
		return cheby	

# Apply for each row a lambda and calculating each bound
freqs['hoeffding_bound'] = freqs['mean'].apply(lambda x : hoeffding_bound(bias, x, n))
freqs['markov_bound'] = freqs['mean'].apply(lambda x : markov_bound(bias, x))
freqs['chebychev_bound'] = freqs['mean'].apply(lambda x : chebychev_bound(bias, x, n))
freqs = freqs[freqs['mean']>=0.5]

print(freqs['hoeffding_bound'])
print(freqs['markov_bound'])
print(freqs['chebychev_bound'])

# Plot each bound + empirical frequency
plt.plot(vector_range_i, vector_sum, label="empirical frequency")
plt.plot(freqs['mean'], freqs['hoeffding_bound'], label = 'hoeffding bound')
plt.plot(freqs['mean'], freqs['markov_bound'], label = 'markov bound')
plt.plot(freqs['mean'], freqs['chebychev_bound'], label = 'chebyshev bound')
plt.xlabel('Treshold (alpha)', fontsize = 'large')
plt.ylabel('Fraction of row average', fontsize = 'large')
plt.legend()
plt.savefig('ex1.png')

# Excercise 1, Question 3
print("hoeffding 1")
print(hoeffding_bound(0.5,1,20))
print("hoeffding 0.95")
print(hoeffding_bound(0.5,0.95,20))

###
# Excercise 3
###

p = np.linspace(0, 1, num=10**4)
all_passenger_prob = p**100
passenger_prob = np.exp(-2*(10000*(0.95-p))**(2) / 10000)
outcome = all_passenger_prob*passenger_prob
bound = max(outcome)
worstcase_p = p[np.argsort(outcome)[-1]]
print("Worst Case p:")
print(worstcase_p)
print("Bound")
print(bound)

###
# Exercise 5
###

#load test and train data
train_set = pd.read_table('IrisTrainML.dt', header = None, sep = ' ')
test_set = pd.read_table('IrisTestML.dt', header = None, sep = ' ')

train_set.columns = ['length', 'width', 'label']
test_set.columns = ['length', 'width', 'label']

# remove class 2 from test and train set
train_set = train_set.loc[train_set.label != 2, :]
test_set = test_set.loc[test_set.label != 2, :]

# Size 62/ 26

