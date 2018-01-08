#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 07:59:00 2018

@author: brendontucker
"""

#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% LOAD TEST FILE 
#resutl of this is what I submit
submitTest = pd.read_json("/Users/brendontucker/KaggleData/StatoilCCORE/data/processed/test.json")
# wow, a 1.5 gb file loaded.... 


#%% LOAD TRAIN
orginal_train = pd.read_json("/Users/brendontucker/KaggleData/StatoilCCORE/data-1/processed/train.json")

# TEST TRAIN SPLIT
msk = np.random.rand(len(orginal_train)) < 0.8 
train = orginal_train[msk]
test = orginal_train[~msk]

# X TARGET VARIABLE SET UP WITH BOTH RADAR TYPES
    
XtargetTrain = np.zeros(shape=(len(train)*2,5625))
for x in range(len(train)):
    XtargetTrain[x] = train.iloc[x][0]
    XtargetTrain[x+len(train)] = train.iloc[x][1]
XtargetTrain = XtargetTrain.T

# Y TARGET VARIABLE SET UP 

YtargetTrain = np.zeros(shape=(len(train)*2,1))
for x in range(len(train)):
    YtargetTrain[x] = train.iloc[x][4]
    YtargetTrain[x+len(train)] = train.iloc[x][4]
YtargetTrain = YtargetTrain.T

# X TEST VAR SET UP WITH BOTH RADAR TYPES
XtargetTest = np.zeros(shape=(len(test)*2,5625))
for x in range(len(test)):
    XtargetTest[x] = test.iloc[x][0]
    XtargetTest[x+len(test)] = test.iloc[x][1]
XtargetTest = XtargetTest.T

# Y TEST TARGET VARIABLE SET UP

YtargetTest = np.zeros(shape=(len(test)*2,1))
for x in range(len(test)):
    YtargetTest[x] = test.iloc[x][4]
    YtargetTest[x+len(test)] = test.iloc[x][4]
YtargetTest = YtargetTest.T

# PREPROCESSING (eventually this will have to be its own file)

# BASIC PREPROCESSING VARS

mean = XtargetTrain.mean(axis=0)
std = XtargetTrain.std(axis=0)
std.shape

# SUPER BASIC PREPROCESSING

XtargetTrain = XtargetTrain/mean
XtargetTrain = XtargetTrain - std


# TESTSET PREPROCESSING 

mean1 = XtargetTest.mean(axis=0)
std1 = XtargetTest.std(axis=0)
XtargetTest = XtargetTest/mean1
XtargetTest = XtargetTest - std1

#%% HELPER FUNCTIONS FOR DEEP LEARNING 

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * .01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * .01
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    ### END CODE HERE ###
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        ### END CODE HERE ###
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        caches.append(cache)
        ### END CODE HERE ###
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)

    ### END CODE HERE ###
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = (-1/m)*np.sum(np.multiply(Y,np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = (1/m)*np.dot(dZ, A_prev.T)
    #db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    #db = np.array([[db]])
    #print("db is:", db, "db shape is:", db.shape)
    #print("b is:", b, "b shape is:", b.shape)
    #db = np.array([[(1/m)*np.sum(dZ)]])
    #print('b:', b, 'shape of b:', b.shape, 'db:', db, 'shape of db:', db.shape)
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache[0])
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache[0])
        ### END CODE HERE ###
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL,current_cache[1]),current_cache[0])
    
    ### END CODE HERE ###
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)]  
        # ,grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        #print(l)
        current_cache = caches[l]
        #dZ = sigmoid_backward(dAL, caches[1][0][0])
        #print("dZ:", dZ)
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        #linear_backward(dZ, current_cache[0])
        #linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    ### END CODE HERE ###
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

#%% TWO LAYER MODEL
    
def two_layer_model(X, Y, layers_dims, learning_rate = 0.00285, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        ### END CODE HERE ###
        
        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        ### END CODE HERE ###
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.75, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    #np.random.seed(1) maybe this put me too close to the true minimized cost?
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
          
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    
    return parameters #trying to print costs also

#%% EXPERIMENT WITH TWO-LAYER MODEL 
    
n_x = 5625     # should be pixel count in HV image? + other radar dimension? 
n_h = 17        #what happens if we increase this? was 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

# was: train_x, train_y,
# is: XtargetTrain, YtargetTrain,

parameters = two_layer_model(XtargetTrain, YtargetTrain, 
                             layers_dims = (n_x, n_h, n_y),
                             learning_rate = 0.03,
                             num_iterations = 50000, print_cost=True)


#%% FINDING A BETTER LEARNING RATE for n_h=17
learning_rates = [0.02, 0.01, 0.002, 0.001, 0.0005]
parameters = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    parameters[str(i)] = two_layer_model(XtargetTrain, YtargetTrain, 
                             layers_dims = (n_x, n_h, n_y),
                             learning_rate = i,
                             num_iterations = 2000, print_cost=True)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(parameters[str(i)]["costs"]), 
             label= str(parameters[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations*100')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
#%% ACCURACY CHECK

predictions_train = predict(XtargetTrain, YtargetTrain, parameters)
predictions_test = predict(XtargetTest, YtargetTest, parameters)



#%% RESULTS LOG 
#would eventually like a way to plot accuracy predictions per 100. Maybe make
#that part of gradient checking
'''

n_h = 17
parameters = two_layer_model(XtargetTrain, YtargetTrain, 
                             layers_dims = (n_x, n_h, n_y),
                             learning_rate = 0.00285,
                             num_iterations = 10000, print_cost=True)
Cost after iteration 9900: 0.6121965393368791
Accuracy: 0.651813880126
Accuracy: 0.639880952381

parameters = two_layer_model(XtargetTrain, YtargetTrain, 
                             layers_dims = (n_x, n_h, n_y),
                             learning_rate = 0.00285,
                             num_iterations = 25000, print_cost=True)

Cost after iteration 24900: 0.5559457371623462
Accuracy: 0.688880126183
Accuracy: 0.665178571429


n_h = 17
parameters = two_layer_model(XtargetTrain, YtargetTrain, 
                             layers_dims = (n_x, n_h, n_y),
                             learning_rate = 0.00285,
                             num_iterations = 40000, print_cost=True)
Accuracy: 0.709266409266
Accuracy: 0.68932038835

n_h = 17
parameters = two_layer_model(XtargetTrain, YtargetTrain, 
                             layers_dims = (n_x, n_h, n_y),
                             learning_rate = 0.00285,
                             num_iterations = 45000, print_cost=True)
Cost after iteration 44900: 0.5123028478884648
Accuracy: 0.716602316602
Accuracy: 0.694174757282

n_h = 17
parameters = two_layer_model(XtargetTrain, YtargetTrain, 
                             layers_dims = (n_x, n_h, n_y),
                             learning_rate = 0.03,
                             num_iterations = 50000, print_cost=True)
Cost after iteration 49900: 0.5280477733712402


learning rate is: 0.285
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6910916423732921
Cost after iteration 200: 0.6910915055066513
Cost after iteration 300: 0.6910912781291456
Cost after iteration 400: 0.691090955934224
Cost after iteration 500: 0.691091828879263
Cost after iteration 600: 0.691091824718516
Cost after iteration 700: 0.6910918197480824
Cost after iteration 800: 0.6910918136428486
Cost after iteration 900: 0.691091805942465
Cost after iteration 1000: 0.691091796353015
Cost after iteration 1100: 0.6910917842668988
Cost after iteration 1200: 0.6910917688949045
Cost after iteration 1300: 0.6910917480676306
Cost after iteration 1400: 0.6910913361890751
Cost after iteration 1500: 0.6910906512002061
Cost after iteration 1600: 0.691091838262931
Cost after iteration 1700: 0.6910918372329317
Cost after iteration 1800: 0.6910918372329314
Cost after iteration 1900: 0.6910918372329314

-------------------------------------------------------

learning rate is: 0.0285
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6742351804889182
Cost after iteration 200: 0.6691008315401197
Cost after iteration 300: 0.6652674570048586
Cost after iteration 400: 0.6627233028494048
Cost after iteration 500: 0.6615328561052308
Cost after iteration 600: 0.6883745957432594
Cost after iteration 700: 0.6694019882118697
Cost after iteration 800: 0.6602101328607431
Cost after iteration 900: 0.6595229118379374
Cost after iteration 1000: 0.6590205027771038
Cost after iteration 1100: 0.6586116929473533
Cost after iteration 1200: 0.6583343092286582
Cost after iteration 1300: 0.6581138663658466
Cost after iteration 1400: 0.65799165578704
Cost after iteration 1500: 0.6576189667085518
Cost after iteration 1600: 0.6573304609195947
Cost after iteration 1700: 0.6569945843508901
Cost after iteration 1800: 0.656613848682086
Cost after iteration 1900: 0.656193424880438

-------------------------------------------------------

learning rate is: 0.00285
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6789104589011249
Cost after iteration 200: 0.678354863792582
Cost after iteration 300: 0.6778105706072725
Cost after iteration 400: 0.677273014107972
Cost after iteration 500: 0.6767469713610497
Cost after iteration 600: 0.6762292376030767
Cost after iteration 700: 0.6757187710956342
Cost after iteration 800: 0.6752146076386644
Cost after iteration 900: 0.6747155330994471
Cost after iteration 1000: 0.6742202632621919
Cost after iteration 1100: 0.6737274799069317
Cost after iteration 1200: 0.6732357593797926
Cost after iteration 1300: 0.672743971575757
Cost after iteration 1400: 0.6722500088926611
Cost after iteration 1500: 0.6717508626735565
Cost after iteration 1600: 0.6712437940361453
Cost after iteration 1700: 0.6707280440277581
Cost after iteration 1800: 0.6702008105770436
Cost after iteration 1900: 0.6696585911186352

learning rate is: 0.03
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6739760571279207
Cost after iteration 200: 0.6684863635149952
Cost after iteration 300: 0.6638341665505468
Cost after iteration 400: 0.6639272442078212
Cost after iteration 500: 0.6620048126770468
Cost after iteration 600: 0.6608005592881836
Cost after iteration 700: 0.6616349008803132
Cost after iteration 800: 0.6607987677057882
Cost after iteration 900: 0.6605045535558809
Cost after iteration 1000: 0.660331408133929
Cost after iteration 1100: 0.6602698044587846
Cost after iteration 1200: 0.6602247019261979
Cost after iteration 1300: 0.660047659435054
Cost after iteration 1400: 0.6598402946978986
Cost after iteration 1500: 0.6595586777685706
Cost after iteration 1600: 0.6591932612088411
Cost after iteration 1700: 0.658764913829915
Cost after iteration 1800: 0.6582865919745154
Cost after iteration 1900: 0.6577686761576642

-------------------------------------------------------

learning rate is: 0.04
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6722467839359155
Cost after iteration 200: 0.6672265716429222
Cost after iteration 300: 0.6664040574605576
Cost after iteration 400: 0.6660122710245562
Cost after iteration 500: 0.6661105673142648
Cost after iteration 600: 0.6660177961033965
Cost after iteration 700: 0.665669395218434
Cost after iteration 800: 0.6651615525115657
Cost after iteration 900: 0.6644913142864938
Cost after iteration 1000: 0.6638113489145564
Cost after iteration 1100: 0.6630627672304391
Cost after iteration 1200: 0.6623232189782766
Cost after iteration 1300: 0.6615456512220688
Cost after iteration 1400: 0.660749429541625
Cost after iteration 1500: 0.6600112876083787
Cost after iteration 1600: 0.6592085370042204
Cost after iteration 1700: 0.6584316126047565
Cost after iteration 1800: 0.6577305598060624
Cost after iteration 1900: 0.6566523327620639

-------------------------------------------------------

learning rate is: 0.002
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6790907528633829
Cost after iteration 200: 0.6786839882171811
Cost after iteration 300: 0.6782964296589792
Cost after iteration 400: 0.6779143517907158
Cost after iteration 500: 0.6775350499136987
Cost after iteration 600: 0.6771608911087572
Cost after iteration 700: 0.6767921861081613
Cost after iteration 800: 0.6764276231218348
Cost after iteration 900: 0.6760667808292171
Cost after iteration 1000: 0.6757093950217944
Cost after iteration 1100: 0.6753550978421162
Cost after iteration 1200: 0.6750035462100352
Cost after iteration 1300: 0.6746541051134755
Cost after iteration 1400: 0.6743066360695669
Cost after iteration 1500: 0.6739603132769066
Cost after iteration 1600: 0.6736149691218196
Cost after iteration 1700: 0.6732699746887183
Cost after iteration 1800: 0.6729250703967946
Cost after iteration 1900: 0.6725794365328946

-------------------------------------------------------

learning rate is: 0.001
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6804954042659552
Cost after iteration 200: 0.6790906069307653
Cost after iteration 300: 0.6788792235563258
Cost after iteration 400: 0.6786831443155711
Cost after iteration 500: 0.6784887820998639
Cost after iteration 600: 0.6782956182286465
Cost after iteration 700: 0.6781041040599862
Cost after iteration 800: 0.6779135713116089
Cost after iteration 900: 0.677723847295489
Cost after iteration 1000: 0.6775342963645022
Cost after iteration 1100: 0.6773462870377353
Cost after iteration 1200: 0.6771601745215491
Cost after iteration 1300: 0.6769752638839822
Cost after iteration 1400: 0.6767914993701443
Cost after iteration 1500: 0.6766087722244409
Cost after iteration 1600: 0.6764269680245293
Cost after iteration 1700: 0.6762461123568432
Cost after iteration 1800: 0.6760661563470979
Cost after iteration 1900: 0.6758870702679387

-------------------------------------------------------

learning rate is: 0.0005
Cost after iteration 0: 0.6904752463266352
Cost after iteration 100: 0.6846543864635856
Cost after iteration 200: 0.6804979097673449
Cost after iteration 300: 0.6793253269196675
Cost after iteration 400: 0.6790905460465373
Cost after iteration 500: 0.6789787719307535
Cost after iteration 600: 0.678878798731068
Cost after iteration 700: 0.6787804114297994
Cost after iteration 800: 0.6786827229114959
Cost after iteration 900: 0.6785854204719174
Cost after iteration 1000: 0.6784883692047496
Cost after iteration 1100: 0.6783915437298315
Cost after iteration 1200: 0.6782952118843935
Cost after iteration 1300: 0.6781992841436296
Cost after iteration 1400: 0.6781037054949026
Cost after iteration 1500: 0.6780083457021184
Cost after iteration 1600: 0.6779131783243407
Cost after iteration 1700: 0.6778182394921353
Cost after iteration 1800: 0.6777234521307376
Cost after iteration 1900: 0.6776285339460031


'''




