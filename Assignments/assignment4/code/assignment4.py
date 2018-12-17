import pandas as pd
import scipy as scp
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

parkinson_train = pd.read_table('parkinsonsTrainStatML.dt', header = None, sep = ' ')
parkinson_test = pd.read_table('parkinsonsTestStatML.dt', header = None, sep = ' ')

parkinson_train_labels = parkinson_train.iloc[:,-1].values
parkinson_train = parkinson_train.iloc[:,:-1].values
parkinson_test_labels = parkinson_test.iloc[:,-1].values
parkinson_test = parkinson_test.iloc[:,:-1].values

# normalize data with StandardScaler and calculate std deviations and means:
normalize = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(parkinson_train)

x_train = normalize.transform(parkinson_train)
y_train = parkinson_train_labels
x_test = normalize.transform(parkinson_test)
y_test = parkinson_test_labels


# create table with means and deviations for train Set and output is as latex
table = pd.DataFrame({'before mean' : scp.mean(parkinson_train, axis = 0),
                      'normalized mean' : scp.mean(x_train, axis = 0), 
					  'before std. deviation' : scp.std(parkinson_train, axis = 0),
                      'normalized std. deviation' : scp.std(x_train, axis = 0)},
                      index = range(1,23)) # 23 Data lines
table.index.name = 'Feature'
cols = table.columns
cols = scp.array([cols[0], cols[1], cols[2], cols[3]])
table = table[cols]
table = table.reset_index()
table = table.round(4)
table['normalized mean'] = abs(table['normalized mean'])
table.to_latex(buf='train_set_normalized_table.tex', index=False)

# Create table with means and eviations for test set and output it as latex
table = pd.DataFrame({'before mean' : scp.mean(parkinson_test, axis = 0),
                      'normalized mean' : scp.mean(x_test, axis = 0), 
					  'before std. deviation' : scp.std(parkinson_test, axis = 0),
                      'normalized std. deviation' : scp.std(x_test, axis = 0)},
                      index = range(1,23)) # 23 Data lines
table.index.name = 'Feature'
cols = table.columns
cols = scp.array([cols[0], cols[1], cols[2], cols[3]])
table = table[cols]
table = table.reset_index()
table = table.round(8)
table['normalized mean'] = abs(table['normalized mean'])
table.to_latex(buf='test_set_normalized_table.tex', index=False)

# set paramas gamma (y) and C
param_grid = {'C' : scp.logspace(-2,4,7, base=10), 'gamma' : scp.logspace(-4,2,7, base=10)}

# 5 fold cross validation
clf = GridSearchCV(SVC(kernel = 'rbf'), param_grid, iid=True,cv=5, return_train_score=True)

#Fit the model
clf.fit(x_train,y_train)

# Cross validation result
cross_validation_results = pd.DataFrame(clf.cv_results_)[['mean_test_score','params']]
cross_validation_results['C'] = cross_validation_results.params.apply(lambda d : d['C'])
cross_validation_results['Y'] = cross_validation_results.params.apply(lambda d : d['gamma'])
cross_validation_results = cross_validation_results.pivot(index = 'C', columns = 'Y', values = 'mean_test_score')

cross_validation_results.to_latex(buf='cross_validation_results_table.tex', index=False)

print("Best params:")
print(clf.best_params_)
print("Best Score:")
print(clf.best_score_)
print("Accuracy:")
print(1 - sum(abs(clf.predict(x_test) - y_test))/len(y_test))


param_grid = {'C' : scp.logspace(-2,4,7, base=10), 'gamma' : [0.01]}
for x in [0.1,10,100,1000,10000]:
	clf = SVC(kernel = 'rbf', C = x, gamma = 0.01,verbose=True)
	clf.fit(x_train,y_train)
	print(clf.score)

