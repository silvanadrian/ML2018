import matplotlib.pyplot as plt
import numpy as np

###
## Excercise 1 - Illustration of Markov's and Chebychev's Inequalities
###

# 1 Million experiments of 20 trials
vector_coins = np.random.randint(2, size=(10 ** 6, 20))
# Mean number of outcome (0/1) in each of the experiments
vector_means = np.mean(vector_coins, 1)

vector_count = []
vector_i = []
# Different Alpha-thresholds
for i in range(50, 105, 5):
    i = float(i) / 100  # Alpha

    count = (vector_means >= i).sum() / float(1000000)  # Fraction of experiments over the threshold
    vector_count.append(count)
    vector_i.append(i)

plt.plot(vector_i, vector_count, c="blue", alpha=0.5, label="Fraction of experiment's averages")
plt.ylabel("Fraction of experiment's averages above or equal to threshold")
plt.xlabel("Threshold")

m = 0.5  # The bias
vector_i = np.asarray(vector_i)

markov = m / vector_i
plt.plot(vector_i, markov, c="red", alpha=0.5, label="Markov upper bound")

v = 0.0125  # Change by theoretical variance

ebs = vector_i - m
ebs_sqr = np.square(ebs)
try:
    cheby = v / ebs_sqr
except:
    cheby = 1
np.asarray(cheby)[cheby > 1] = 1
vector_i = vector_i
plt.plot(vector_i, cheby, c="green", alpha=0.5, label="Chebyshev's upper bound")
plt.legend(loc='upper right')

plt.savefig('markov_chebychev.png')
plt.close()

###
## Excercise 2 - Digits Classification with Nearest Neighbours
###

def resize(name):
    name = np.reshape(name, (-1, 784))
    name = np.transpose(name)
    return name

def calc_distance(x1, x2):
    'Calculation of eucledian distance'
    dist = np.dot(np.transpose(x1 - x2), (x1 - x2))
    return dist


def loss(classification, label, comparison, input):
    'Loss function 0,1 for the predictions given by different K neighbors and the known label'
    x_plot = []
    y_plot = []
    for k in range(0, 33, 2):  # For each of the Ks specified in the exercise, i.e odd numbers from 1 to 34
        try:  # Exception handling if a small dataset is given
            knn = classification[:, k]  # Take the prediction for a certain K
        except:
            break
        knn[knn == 0] = np.random.choice([-1, 1])
        loss = 1 - (np.sum(knn == label) / knn.shape[0])

        x_plot.append(k + 1)
        y_plot.append(loss)
    plt.plot(x_plot, y_plot, alpha=0.5, label=input + "_" + str(comparison[0]) + "_" + str(comparison[1]))


def knn(train_data, test, comparison, input):
    a = comparison[0]
    b = comparison[1]

    train_i = train_data[(a == train_data[:, -1]) | (b == train_data[:, -1])]
    validation_index = int(0.8 * train_i.shape[0])  # Index below which the 80% of the data is present

    train_data = train_i[0:validation_index]  # select 80% of dataset for training

    # either validation or test
    if input == "validation":
        test = train_i[validation_index:]
    else:
        test = test[(a == test[:, -1]) | (b == test[:, -1])]

    validation_y = test[:, -1]
    validation_y[validation_y == b] = 1
    validation_y[validation_y == a] = -1
    validation_x = test[:, 0:-1]

    train_y = train_data[:, -1]
    train_y[train_y == b] = 1
    train_y[train_y == a] = -1
    train_x = train_data[:, 0:-1]

    classification_array = []  # predicated labels
    for l in validation_x:  # Iteration over each instance on the TEST/VALIDATION which label should be predicted
        distance_matrix = np.apply_along_axis(calc_distance, 1, train_x, x2=l)
        distance_matrix = np.column_stack((distance_matrix, train_y))
        distance_matrix = distance_matrix[distance_matrix[:, 0].argsort()]
        classification = np.sign(np.cumsum(distance_matrix, 0)[:, 1])
        classification_array.append(list(classification))

    classification_array = np.asarray(classification_array)
    loss(classification_array, validation_y, comparison, input)


# load data
dataTest = np.loadtxt('MNIST-Test-cropped.txt', delimiter=' ')
dataTrain = np.loadtxt('MNIST-Train-cropped.txt', delimiter=' ')

Y_Test = np.loadtxt('MNIST-Test-Labels-cropped.txt', delimiter=' ')
Y_Train = np.loadtxt('MNIST-Train-Labels-cropped.txt', delimiter=' ')

dataTrain = resize(dataTrain)
dataTest = resize(dataTest)

# Add Lables

train = np.column_stack((np.transpose(dataTrain), Y_Train))
test = np.column_stack((np.transpose(dataTest), Y_Test))

# Classification 0/1
knn(train, test, (0, 1), "validation")
# Classification 0/8
knn(train, test, (0, 8), "validation")
# Classification 5/6
knn(train, test, (5, 6), "validation")

knn(train, test, (0, 1), "test")
knn(train, test, (0, 8), "test")
knn(train, test, (5, 6), "test")

plt.ylabel("Average Loss given by 0-1 loss function")
plt.xlabel("Number of K neighbors")

plt.legend(loc='lower right')
plt.savefig("loss_knn_plot.png")
plt.close()

###
## Excercise 3 -- Linear Regression
###

def linear_regression(X, y):
    one_matrix = np.zeros((X.shape[0], 1)) + 1

    X = np.column_stack((one_matrix, X))
    inv = np.linalg.inv(np.dot(X.T, X))

    W = np.dot(np.dot(inv, X.T), y)
    predictions = np.dot(W.reshape(W.shape[0], 1).T, X.T)

    return W, predictions


def least_squares(Y, predicted):
    e = Y - predicted
    ls = np.mean(e ** 2)
    return ls

# load DanWood dataset
dataset = np.loadtxt("DanWood.dt")

weights, predictions = linear_regression(dataset[:, 0], dataset[:, 1])

error = least_squares(dataset[:, 1], predictions)

variance_data = np.var(dataset[:, 1])

x = dataset[:, 0]
y = dataset[:, 1]
plt.scatter(x, y)

f = lambda z: weights[1] * z + weights[0]
z = np.array([min(x), max(x) + 0.1])
plt.plot(z, f(z), c="orange", label="fit with affine linear model")

print("Weights: ")
print(weights)
print("Error and variance: ")
print(error, variance_data, error / variance_data)

# Non linear case:
# Transfromation of X

weights, predictions = linear_regression(dataset[:, 0] ** 3, dataset[:, 1])
error = least_squares(dataset[:, 1], predictions)

x = dataset[:, 0]
y = dataset[:, 1]
plt.scatter(x, y)

f = lambda z: weights[1] * z ** 3 + weights[0]
z = np.asarray(list(range(130, 180, 5))) / float(100)
plt.plot(z, f(z), c="blue", label="fir with non-linear model")

plt.legend(loc='upper left')
plt.xlabel("Units of 1000 kelvin")
plt.ylabel("Energy (radiation per cm^2/s)")
plt.savefig("linear_regression.png")
plt.close()

print("Weights: ")
print(weights)
print("Error: ")
print(error)
