#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 06:15:30 2017

@author: brendontucker
"""
#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% LOAD TRAIN
orginal_train = pd.read_json("/Users/brendontucker/KaggleData/StatoilCCORE/data-1/processed/train.json")

#%% TEST TRAIN SPLIT
msk = np.random.rand(len(orginal_train)) < 0.8
train = orginal_train[msk]
test = orginal_train[~msk]

#%% LOAD TEST FILE 
#resutl of this is what I submit
submitTest = pd.read_json("/Users/brendontucker/KaggleData/StatoilCCORE/data/processed/test.json")
# wow, a 1.5 gb file loaded.... 

#%% WHAT IS THIS DATA?
# size of train is (1604, 5)
''' 
okay, train has 1604 images, each with five cols of data describing them
band_1: len of 5625 HH (transmit/receive horizontally)
band_2: len of 5625 HV (transmit horizontally and receive vertically)
    I believe this means we have 5625 X 5625 radar images
id: simple aplphanumeric identification label
inc_angle: surely I can use this normalize values in band_1 and band_2?
is_iceberg: binary classifier 
'''
#%% MORE EDA
# know this is ugly--will find a way to vectorize
#%% find info about inc_angles
inc_angles = []
for x in range(len(train)):
    if type(train.loc[x]['inc_angle']) == float:
        inc_angles.append(train.loc[x]['inc_angle'])
inc_angles_S = pd.Series(data=inc_angles)  
#%% 
#couldn't get statistics to work without skipping all 'na' in original
#creation... should find a way around this later
#have about 130 na values
inc_angles_S.describe(percentiles=[0.0001, 0.0005, 0.001, .25, .5,
                                   .75, 0.999, 0.9995, 0.9999])

#%% RETURN STATS
'''
count     1471.000000
mean        39.268707
std          3.839744
min         24.754600
0.01%       25.548988
0.05%       28.726540
0.1%        30.289589
25%         36.106100
50%         39.501500
75%         42.559100
99.9%       45.927318
99.95%      45.930223
99.99%      45.936045
max         45.937500
dtype: float64
'''
# has a fat left tail... easier to deal with than a fat right tail I suppose


#%% HH (band1 EDA)

testSeriesHH = pd.Series(train.iloc[0][0])
testSeriesHH.describe(percentiles=[0.0001, 0.0005, 0.001, .25, .5,
                                   .75, 0.999, 0.9995, 0.9999])


#%% RETURN STATS
#might need to make these positive to have Sigmoid predict accurately...
#as well as cost function

    
#%% HV EDA (band 2)

testSeriesHV = pd.Series(train.iloc[0][1])
testSeriesHV.describe(percentiles=[0.0001, 0.0005, 0.001, .25, .5,
                                   .75, 0.999, 0.9995, 0.9999])
  
#%% RETURN STATS
'''
Out[15]: 
count     5625.000000
mean       -29.910117
std          2.381496
min        -41.135918
0.01%      -41.135918
0.05%      -39.551050
0.1%       -38.716392
25%        -31.591387
50%        -30.007847
75%        -28.267622
99.9%      -14.742735
99.95%     -13.125043
99.99%     -12.068203
max        -11.252153
dtype: float64
'''
#again, not comparing to anything but other water-images...
# prob need to convert to positive... since all are negative it should be
#okay? again, not sure if this is needed or not

#%% CREATING DF OF TARGET VARIABLE
#need to take each list of train.iloc[x] of 5625 float, and add as col
#or row? Add as a row I believe. 
#for x in range(len(train)):
    

#%% CAN WE ADD SERIES TO CREATE A DF? 
testSeries = pd.Series(train.iloc[0][0])    
addSeries = pd.Series(train.iloc[1][0])
    
    
    
    
#%% NN SET UP 
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1 + np.exp(-z))
    ### END CODE HERE ###
    
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)#sigmoid(w.T*X +b) -> led to a broadcast error   # compute activation
    #print('test: A is;', A)
    cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) #Y*np.log(A) + (1-Y)*np.log(1-A)
    #print('test: cost is;', cost)
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dZ = A - Y #i added this to match format in videos
    #print('test: dZ is;', dZ)
    dw = (1/m)*np.dot(X, dZ.T) #(1/m)*X*dZ.T
    db = (1/m)*np.sum(dZ)
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###  
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate*dw
        b = b - learning_rate*db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X))  #[x = 0 if x<=.5 else x = 1 in A] 
    #print(A, A[0], A.shape[1], A[0][0], 'Y_pred:', Y_prediction)
    ### END CODE HERE ###
    
    #Y_predictionTest[0] = [x*0 if x<=.5 else math.ceil(x) for x in A[0]] #this is how to do it without a for loop?
    
    for i in range(A.shape[1]):
    
    # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[0][i] <= .5:
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


#%% X TARGET VARIABLE SET UP
    
XtargetTrain = np.zeros(shape=(len(train),5625))
for x in range(len(train)):
    XtargetTrain[x] = train.iloc[x][0]
XtargetTrain = XtargetTrain.T

#%% WHAT WE HAVE NOW

'''
XtargetTrain.shape
Out[64]: (5625, 1291)
'''

#%% Y TARGET VARIABLE SET UP

YtargetTrain = np.zeros(shape=(len(train),1))
for x in range(len(train)):
    YtargetTrain[x] = train.iloc[x][4]
YtargetTrain = YtargetTrain.T


#%% X TEST VAR SET UP
XtargetTest = np.zeros(shape=(len(test),5625))
for x in range(len(test)):
    XtargetTest[x] = test.iloc[x][0]
XtargetTest = XtargetTest.T

#%% Y TEST TARGET VARIABLE SET UP

YtargetTest = np.zeros(shape=(len(test),1))
for x in range(len(test)):
    YtargetTest[x] = test.iloc[x][4]
YtargetTest = YtargetTest.T

#%% BASIC PREPROCESSING VARS

mean = XtargetTrain.mean(axis=0)
std = XtargetTrain.std(axis=0)
std.shape

#%% SUPER BASIC PREPROCESSING

XtargetTrain = XtargetTrain/mean
XtargetTrain = XtargetTrain - std


# %% TESTSET PREPROCESSING 

mean1 = XtargetTest.mean(axis=0)
std1 = XtargetTest.std(axis=0)
XtargetTest = XtargetTest/mean1
XtargetTest = XtargetTest - std1

#%% RUN THE NN
'''
d = model(train_set_x, train_set_y, test_set_x, test_set_y, 
          num_iterations = 2000, learning_rate = 0.005, print_cost = True)
'''
d = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 4000, learning_rate = 0.0003, print_cost = True)


#%% playing with hyperparameters before moving on to a more complex model
'''
d = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 10000, learning_rate = 0.0001, print_cost = True)

Cost after iteration 9900: 0.608378
train accuracy: 66.69248644461658 %
test accuracy: 59.42492012779553 %

d = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 10000, learning_rate = 0.0005, print_cost = True)

Cost after iteration 9900: 0.674620
train accuracy: 64.44616576297443 %
test accuracy: 57.18849840255591 %

d = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 20000, learning_rate = 0.0001, print_cost = True)

Cost after iteration 19900: 0.572093
train accuracy: 71.3400464756003 %
test accuracy: 65.814696485623 %

d = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 40000, learning_rate = 0.0001, print_cost = True)

Cost after iteration 39900: 0.529754
train accuracy: 75.6003098373354 %
test accuracy: 68.05111821086263 %

d = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 100000, learning_rate = 0.0001, print_cost = True)

Cost after iteration 99900: 0.458904
train accuracy: 82.21191028615623 %
test accuracy: 65.27331189710611 %

d = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 4000, learning_rate = 0.0003, print_cost = True)

Cost after iteration 3900: 0.598017
train accuracy: 71.77107501933489 %
test accuracy: 58.842443729903536 %

'''

#%% MORE SOPHISTICATED EXPERIMENTATION
learning_rates = [0.0001, 0.00009, 0.00005, 0.00001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, 
          num_iterations = 1000, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#%% appears that our train accuracy is improving faster than the test accuracy
#would like to get > 90% train accuracy before making any judgements  

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()




#%% TEST ON TRUE TEST 
XsubmitTest = np.zeros(shape=(len(submitTest),5625))
for x in range(len(submitTest)):
    XsubmitTest[x] = test.iloc[x][0]
XsubmitTest = XsubmitTest.T


#%%
my_predicted_image = predict(d["w"], d["b"], my_image)
