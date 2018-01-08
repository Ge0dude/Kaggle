#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:10:01 2018

@author: brendontucker
"""

#%% IMPORTS

import pandas as pd
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

#np.random.seed(1)

#%% DEF HOT ONES AND CONVERT
#used to transform the target variable



def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return one_hot

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y



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

#%% X TARGET VARIABLE SET UP WITH BOTH RADAR TYPES

'''
we want to match the general dimensions of the tensorflow example of 1080 images
each with 64*64*64*3 pixels
X_train shape: (12288, 1080)
and we do
XtargetTrain.shape
Out[6]: (5625, 2536)
'''
    
XtargetTrain = np.zeros(shape=(len(train),11250))
for x in range(len(train)):
    XtargetTrain[x] = train.iloc[x][0] + train.iloc[x][1]
XtargetTrain = XtargetTrain.T 


#%% Y TARGET VARIABLE SET UP 
'''
we want to match the example data for five hand options
Y_train shape: (6, 1080)
we aren't quite right with
YtargetTrain.shape
Out[13]: (1, 2536)
think we'll have to do a hot-one conversion ?
Y_train_orig.shape (1, 1080)
gets converted to
Y_train.shape (6, 1080)
through Y_train = convert_to_one_hot(Y_train_orig, 6)


'''

YtargetTrain = np.zeros(shape=(len(train),1))
for x in range(len(train)):
    YtargetTrain[x] = train.iloc[x][4]
#had to add .astype to fix convertohotones error 
YtargetTrain = YtargetTrain.astype('int64') # a row of truth values corresponding to training data
#%% converting to one-hot representation 

'''
Y_train = convert_to_one_hot(Y_train_orig, 6)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
    
IthinkIdentity
Out[38]: 
array([[ 1.,  0.],
       [ 0.,  1.]])
'''
#IthinkIdentity=np.eye(2)
YtargetTrain = convert_to_one_hot(YtargetTrain, 2)
'''
okay! now we have :
YtargetTrain[0].shape
Out[58]: (1282,)

YtargetTrain.shape[0]
Out[59]: 2

which is what we want!

'''


#%% X TEST VAR SET UP WITH BOTH RADAR TYPES
'''
    
XtargetTrain = np.zeros(shape=(len(train),11250))
for x in range(len(train)):
    XtargetTrain[x] = train.iloc[x][0] + train.iloc[x][1]
XtargetTrain = XtargetTrain.T 
'''

XtargetTest = np.zeros(shape=(len(test),11250))
for x in range(len(test)):
    XtargetTest[x] = test.iloc[x][0] + test.iloc[x][1]
    #XtargetTest[x+len(test)] = test.iloc[x][1]
XtargetTest = XtargetTest.T

'''
XtargetTest.shape
Out[61]: (11250, 322)
'''

#%% Y TEST TARGET VARIABLE SET UP

'''
YtargetTrain = np.zeros(shape=(len(train),1))
for x in range(len(train)):
    YtargetTrain[x] = train.iloc[x][4]
#had to add .astype to fix convertohotones error 
YtargetTrain = YtargetTrain.astype('int64')
'''

YtargetTest = np.zeros(shape=(len(test),1))
for x in range(len(test)):
    YtargetTest[x] = test.iloc[x][4]
YtargetTest = YtargetTest.astype('int64')


YtargetTest = convert_to_one_hot(YtargetTest, 2)
#%% PREPROCESSING (eventually this will have to be its own file)

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
#%% TENSORFLOW NN 

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction



def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    '''
    #what I did first, wrong!
    X = tf.constant(np.random.randn(n_x), name = "X")
    Y = tf.constant(np.random.randn(n_y), name = "Y")
    '''
    ### END CODE HERE ###
    
    return X, Y

#tried to modify this fit my data parameters, didn't change the amount of layers
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25, 11250], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [2, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [2, 1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3

def linear_function():
    """
    Implements a linear function: 
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """
    
    np.random.seed(1)
    
    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W,X), b)
    ### END CODE HERE ### 
    
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    
    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y)
    ### END CODE HERE ### 
    
    # close the session 
    sess.close()

    return result

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name = "x")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above. 
    # You should use a feed_dict to pass z's value to x. 
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict = {x: z})
    
    ### END CODE HERE ###
    
    return result

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    
    ### START CODE HERE ### 
    
    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(tf.float32, name = "z")
    y = tf.placeholder(tf.float32, name = "y")
    
    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,  labels = y)
    
    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()
    
    # Run the session (approx. 1 line).
    cost = sess.run(cost, feed_dict={z: logits, y: labels})
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return cost

def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
    ### START CODE HERE ###
    
    # Create "ones" tensor using tf.ones(...). (approx. 1 line)
    ones = tf.ones(shape)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session to compute 'ones' (approx. 1 line)
    ones = sess.run(ones)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    return ones

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost


#%% here is the meat of the program... going ot spend lots of time looking at 
    #this I believe 
    

def model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, learning_rate = 0.0001,
          num_epochs = 300, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = XtargetTrain.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = YtargetTrain.shape[0]                            # n_y : output size--after converting to one-hot representation 
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y) #don't think I need to change this
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters() #hope I changed those parameters correctly 
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters) #should be the same 
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y) #might have issues with int32 vs int64 but should otherwise be good
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(XtargetTrain, YtargetTrain, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: XtargetTrain, Y: YtargetTrain}))
        print("Test Accuracy:", accuracy.eval({X: XtargetTest, Y: YtargetTest}))
        
        return parameters




#%% EXECUTE TENSORFLOW NEURAL NETWORK 

# create_placeholders(11250, 2)
parameters = model (XtargetTrain, YtargetTrain, XtargetTest, YtargetTest)
#parameters = model(X_train, Y_train, X_test, Y_test)

#%% create submission
# predict(X, parameters) 

#%%
#first run, not too bad!

'''
parameters = model (XtargetTrain, YtargetTrain, XtargetTest, YtargetTest)
#parameters = model(X_train, Y_train, X_test, Y_test)
Cost after epoch 0: 0.811464
Cost after epoch 100: 0.725620
Cost after epoch 200: 0.681771
Cost after epoch 300: 0.645571
Cost after epoch 400: 0.611891
Cost after epoch 500: 0.548936
Cost after epoch 600: 0.511825
Cost after epoch 700: 0.433819
Cost after epoch 800: 0.404315
Cost after epoch 900: 0.390438
Cost after epoch 1000: 0.404510
Cost after epoch 1100: 0.412515
Cost after epoch 1200: 0.382937
Cost after epoch 1300: 0.297682
Cost after epoch 1400: 0.284559
Parameters have been trained!
Train Accuracy: 0.919657
Traceback (most recent call last):

  File "<ipython-input-66-07ec26d314e1>", line 1, in <module>
    parameters = model (XtargetTrain, YtargetTrain, XtargetTest, YtargetTest)

  File "<ipython-input-65-40e9df3264d6>", line 106, in model
    print("Test Accuracy:", accuracy.eval({X: XtargetTest, Y: YtargetTrain}))
    
    
    
wowowowowowowo :)
    
Cost after epoch 0: 0.811464
Cost after epoch 100: 0.725620
Cost after epoch 200: 0.681771
Cost after epoch 300: 0.645571
Cost after epoch 400: 0.611891
Cost after epoch 500: 0.548936
Cost after epoch 600: 0.511825
Cost after epoch 700: 0.433819
Cost after epoch 800: 0.404315
Cost after epoch 900: 0.390438
Cost after epoch 1000: 0.404510
Cost after epoch 1100: 0.412515
Cost after epoch 1200: 0.382937
Cost after epoch 1300: 0.297682
Cost after epoch 1400: 0.284559
Cost after epoch 1500: 0.313036
Cost after epoch 1600: 0.240628
Cost after epoch 1700: 0.233152
Cost after epoch 1800: 0.191060
Cost after epoch 1900: 0.209523
Parameters have been trained!
Train Accuracy: 0.978159
Test Accuracy: 0.754658

Cost after epoch 0: 0.811464
Cost after epoch 100: 0.725620
Cost after epoch 200: 0.681771
Cost after epoch 300: 0.645571
Cost after epoch 400: 0.611891
Cost after epoch 500: 0.548936
Cost after epoch 600: 0.511825
Cost after epoch 700: 0.433819
Cost after epoch 800: 0.404315
Cost after epoch 900: 0.390438
Cost after epoch 1000: 0.404510
Cost after epoch 1100: 0.412515
Cost after epoch 1200: 0.382937
Cost after epoch 1300: 0.297682
Cost after epoch 1400: 0.284559
Cost after epoch 1500: 0.313036
Cost after epoch 1600: 0.240628
Cost after epoch 1700: 0.233152
Cost after epoch 1800: 0.191060
Cost after epoch 1900: 0.209523
Cost after epoch 2000: 0.171558
Cost after epoch 2100: 0.134904
Cost after epoch 2200: 0.124109
Cost after epoch 2300: 0.103590
Cost after epoch 2400: 0.260602
Cost after epoch 2500: 0.076344
Cost after epoch 2600: 0.064412
Cost after epoch 2700: 0.059679
Cost after epoch 2800: 0.049298
Cost after epoch 2900: 0.044766
Parameters have been trained!
Train Accuracy: 0.98986
Test Accuracy: 0.71118

wow, guess early stoppage might be a good idea if regularization doesn't work out

parameters = model (XtargetTrain, YtargetTrain, XtargetTest, YtargetTest)
#parameters = model(X_train, Y_train, X_test, Y_test)
Cost after epoch 0: 0.811464
Cost after epoch 100: 0.725620
Cost after epoch 200: 0.681771
Cost after epoch 300: 0.645571
Cost after epoch 400: 0.611891
Cost after epoch 500: 0.548936
Cost after epoch 600: 0.511825
Cost after epoch 700: 0.433819
Cost after epoch 800: 0.404315
Cost after epoch 900: 0.390438
Cost after epoch 1000: 0.404510
Cost after epoch 1100: 0.412515
Cost after epoch 1200: 0.382937
Cost after epoch 1300: 0.297682
Parameters have been trained!
Train Accuracy: 0.924337
Test Accuracy: 0.763975

ran this one without preprocessing, just out of curiosity, wow, 
trains much faster

Cost after epoch 0: 2.600578
Cost after epoch 100: 0.378037
Cost after epoch 200: 0.290580
Cost after epoch 300: 0.083913
Cost after epoch 400: 0.118189
Cost after epoch 500: 0.087333
Cost after epoch 600: 0.018625
Cost after epoch 700: 0.009767
Cost after epoch 800: 0.059056
Cost after epoch 900: 0.010778
Cost after epoch 1000: 0.007017
Cost after epoch 1100: 0.009179
Cost after epoch 1200: 0.003659
Cost after epoch 1300: 0.003250
Parameters have been trained!
Train Accuracy: 1.0
Test Accuracy: 0.728614


    
'''
