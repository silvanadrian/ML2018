################################################EXERCISE 1 #############################
########################################################################################


import matplotlib.pyplot as plt
import numpy as np

#1,000,000 repetitions of the experiment of drawing 20 coins with bias 1/2

#Get matrix of 1 million experiments x outcome in each of the 20 trials
vector_outcome = np.random.randint(2,size=(1000000,20))
#Mean number of outcome (0/1) in each of the experimetns (row)
vector_mean = np.mean(vector_outcome,1)

vector_count = []
vector_i = []
#Different Alpha-thresholds
for i in range(50,105,5):
    i = float(i)/100 #Alpha

    count = (vector_mean>=i).sum()/float(1000000) #Fraction of experiments over the threshold
    vector_count.append(count)
    vector_i.append(i)

plt.plot(vector_i, vector_count,c="blue", alpha=0.5, label = "Fraction of experiment's averages")
plt.ylabel("Fraction of experiment's averages above or equal than threshold")
plt.xlabel("Threshold")


#3 :

#4 Because is a discrete distribution that can't produce those proability values (there are different combinations of 20 trials)
#Therefore above or equal to 0.51 is the same than 0.5 and so on until reaching 0.55

#Markov: E(x)/ebsilon ; mean/alpha --> This is the upper bound of your function

# By the Central Limit theorem, we know that the mean of a distribution of 20 bernuilli is equal to its mean (0.5), and its variance is the variance of one
#divided by the number, so 0.5*0.5/20
m = 0.5 #The bias
vector_i = np.asarray(vector_i)

markov = m/vector_i
plt.plot(vector_i, markov,c="red", alpha=0.5, label = "Markov upper bound")



# Chebyshev's bound; Alpha is taken as Ebsilon + Ebsilon[X] , as no values of alpha 0 can be taken, we remove the bound of 0.5

v = 0.0125 #Change by theretical variance

ebsilon = vector_i - m
ebsilon_squared = np.square(ebsilon)
try:
    chebyshev = v/ebsilon_squared
except: chebyshev = 1
np.asarray(chebyshev)[chebyshev>1] = 1
vector_i = vector_i
plt.plot(vector_i, chebyshev,c="green", alpha=0.5, label = "Chebyshev's upper bound")
plt.legend(loc='upper right')


plt.savefig('plot1.png')
plt.close()








################################################EXERCISE 2 #############################
########################################################################################





import matplotlib.pyplot as plt
import numpy as np

def resize(name):
    'Change the linear String to a Matrix of the dimensions specified on the exercise'
    name = np.reshape(name, (-1,784)) # The -1 index tells numpy to find the appropiated dimension
    name = np.transpose(name) # Get the matrix described by the exercise
    return name

#Get Number
def plot_number(col, name):
    'Generate image of the representation of the hand-written number'

    c = dataTest[:,col] #Choosing which number to represent
    c = c.reshape((28,28)) #Pixels
    c = np.transpose(c)
    c = np.real(c)
    plt.imshow(c, cmap='gray')
    plt.savefig(name+"_representation.png")
    plt.close()



def distance(x1,x2):
    'Calculation of Eucledian distance by matrix operations'
    dist = np.dot(np.transpose(x1-x2),(x1-x2))
    return dist


def loss(classification, label, comparison, input):
    'Loss function 0,1 for the predictions given by different K neighbors and the known label'
    x_plot = []
    y_plot = []
    for k in range(0,33,2): #For each of the Ks specified in the exercise, i.e odd numbers from 1 to 34
        try: #Exception handling code if a small dataset is given
            knn = classification[:,k] #Take the prediction for a certain K
        except: break
        knn[knn == 0] = np.random.choice([-1, 1]) #In case of Draw
        loss = 1 - (np.sum(knn == label)/knn.shape[0]) #The loss function is 1 - the matching fraction
        #X, Y values to be plot for each given K
        x_plot.append(k+1)
        y_plot.append(loss)
    #Generate a line plot for a given comparison ( for instance 0 - 1 ) and Input dataset (Test or Validation?)
    plt.plot(x_plot, y_plot, alpha=0.5, label = input + "_" + str(comparison[0])+"_"+ str(comparison[1]))


def knn_validation(train,test,comparison,input):
    #Which two things clasify
    a = comparison[0]
    b = comparison[1]

    #FILTERING
    train_i = train[(a == train[:,-1]) | (b == train[:,-1])]
    validation_index = int(0.8 * train_i.shape[0]) #Index below which the 80% of the data is present

    train = train_i[0:validation_index] # 80% of the data is Selected to be part of the Train dataset

    #Depending on whether we are using validation or test data
    if input == "val":

        test = train_i[validation_index:] #The 20% left is the validation, called Test

    else:
        test = test[(a == test[:,-1]) | (b == test[:,-1])] #if test data, filter

    #Remove of labels. They were included in order to allow the filtering
    validation_y = test[:,-1]
    validation_y[validation_y == b] = 1
    validation_y[validation_y == a] = -1
    validation_x = test[:,0:-1]

    train_y = train[:,-1]
    train_y[train_y == b] = 1
    train_y[train_y == a] = -1
    train_x = train[:,0:-1]


    classification_array = [] # List that will contain the predicted labels
    for i in validation_x: #Iteration over each instance on the TEST/VALIDATION which label should be predicted
        #Compute distances of Train and the instance
        distance_matrix = np.apply_along_axis(distance,1,train_x, x2= i)
        #Add the train labelsto the distance matrix
        distance_matrix = np.column_stack((distance_matrix, train_y))
        #Sort according to distance
        distance_matrix = distance_matrix[distance_matrix[:,0].argsort()]
        #Get classification -1/1, give the predictions to each possible K to a list
        classification = np.sign(np.cumsum(distance_matrix,0)[:,1])
        classification_array.append(list(classification))
    #Make a matrix of labels, each row is an entry, every column a K

    classification_array = np.asarray(classification_array)
    loss(classification_array, validation_y,comparison, input)



dataTest = np.loadtxt('MNIST-Test-cropped.txt', delimiter= ' ' )
dataTrain = np.loadtxt('MNIST-Train-cropped.txt', delimiter= ' ' )

Y_Test = np.loadtxt('MNIST-Test-Labels-cropped.txt', delimiter= ' ')
Y_Train = np.loadtxt('MNIST-Train-Labels-cropped.txt', delimiter= ' ')

dataTrain = resize(dataTrain)
dataTest = resize(dataTest)

#Add Lables

train = np.column_stack((np.transpose(dataTrain),Y_Train))
test = np.column_stack((np.transpose(dataTest),Y_Test))

#Call function for different classifications

knn_validation(train, test,(0,1),"val")
knn_validation(train, test,(0,8), "val")
knn_validation(train, test,(5,6), "val")


knn_validation(train, test,(0,1),"test")
knn_validation(train, test,(0,8), "test")
knn_validation(train, test,(5,6), "test")



plt.ylabel("Average Loss given by 0-1 loss function")
plt.xlabel("Number of K neighbors")

plt.legend(loc='lower right')
plt.savefig("Loss_Plot.png")
plt.close()



################################################EXERCISE 3 #############################
########################################################################################





import matplotlib.pyplot as plt
import numpy as np

def linreg(X,y):
    'Computes analytical W vector'
    one_matrix = np.zeros((X.shape[0],1)) + 1


    X = np.column_stack((one_matrix, X))
    inv = np.linalg.inv(np.dot(X.T,X))

    W = np.dot( np.dot(inv, X.T),y  )
    predictions = np.dot(W.reshape(W.shape[0],1).T,X.T)

    return W, predictions

def leastsquares(Y, predicted):
    'Compute the least of squares from a label Y and a predicted Y'
    e = Y-predicted
    #exit(np.linalg.norm(e))
    ls = np.mean(e**2)
    return ls
dataset = np.loadtxt("DanWood.dt") #  -units 1000 kelvin vs energy (radiation per cm 2 per second)

weights, predictions = linreg(dataset[:,0],dataset[:,1])

error = leastsquares(dataset[:,1],predictions)

variance_data = np.var(dataset[:,1])

x = dataset[:,0]
y = dataset[:,1]
plt.scatter(x, y)

f = lambda z: weights[1]*z + weights[0]
z = np.array([min(x),max(x)+0.1])
plt.plot(z,f(z), c="orange", label="fit with afine linear model")



print("Weights: ")
print(weights)
print("Error and variance: ")
print(error, variance_data, error/variance_data)

#Non linear case:
#Transfromation of X

weights, predictions = linreg(dataset[:,0]**3,dataset[:,1])
error = leastsquares(dataset[:,1],predictions)

x = dataset[:,0]
y = dataset[:,1]
plt.scatter(x, y)

f = lambda z: weights[1] * z**3 + weights[0]
z = np.asarray(list(range(130,180,5)))/float(100)
plt.plot(z,f(z), c="blue", label="fit with nin-linear model")


plt.legend(loc='upper left')
plt.xlabel("Units of 1000 kelvin")
plt.ylabel("Energy (radiation per cm^2/s)")
plt.savefig("Regression.png")
plt.close()

print("Weights: ")
print(weights)
print("Error: ")
print(error)