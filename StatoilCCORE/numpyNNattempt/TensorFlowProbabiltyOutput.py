#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 08:43:29 2018

@author: brendontucker

trying to change this to 

1) produce an output file (csv: id, is_iceberg)


2) change loss function to tf.nn.log_loss
    -think I did this, only changed in the compute cost function...should be the only place I need
    it? 


3) have predictions be probability (skip step where probablities are converted
     to predictions )


first submission was on the 400 epoch cycle with no data preprocessing.
submitted 1 column and got .65 ... thinking maybe 1 is actually the not-an-iceberg prediction
going to try trainng 2000 epochs on preprocessed data and submit that version and see if 1 column
is still so bad... 

still not using the "right" cost function either... 
"""

#%% IMPORTS

import pandas as pd
import math
import numpy as np
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

#%% clean submission data

X_test = np.zeros(shape=(len(submitTest),11250))
for x in range(len(submitTest)):
    X_test[x] = submitTest.iloc[x][0] + submitTest.iloc[x][1]
X_test = X_test.T

#%% BASIC PREPROCESSING VARS

mean = X_test.mean(axis=0)
std = X_test.std(axis=0)

X_test = X_test/mean
X_test = X_test - std


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
#%% PREPROCESSING (eventually this will have to be its own function/file)

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
    #print('logits are:',logits)
    labels = tf.transpose(Y)
    #print('lables are:', labels)
    
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost

def compute_cost_regularized(Z3, Y, parameters, lambd):
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
    #print('logits are:',logits)
    labels = tf.transpose(Y)
    #print('lables are:', labels)
    
    #adding L2 Regularization 
    
    # Retrieve the parameters from the dictionary "parameters" 
    #W1 = parameters['W1']
    #W2 = parameters['W2']
    #W3 = parameters['W3']
    #m = Y.shape[1]
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0000005, scope=None)
    weights = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    
    #L2_regularization_cost = lambd*(tf.nn.l2_normalize(W1, dim=0)+tf.nn.l2_normalize(W2, dim=0)+tf.nn.l2_normalize(W3, dim=0)) / (m*2)
    #L2_regularization_cost = lambd*(tf.reduce_sum(tf.square(W1))+tf.reduce_sum(tf.square(W2))+tf.reduce_sum(tf.square(W3))) / (m*2)    
    
    #end L2
    
    ### START CODE HERE ### (1 line of code)
    softMax_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    cost = regularization_penalty + softMax_cost
    ### END CODE HERE ###
    
    return cost

    
    
#%% here is the meat of the program... going ot spend lots of time looking at 
    #this I believe 
# learning_rate = 0.0001    

def model(XtargetTrain, YtargetTrain, XtargetTest, YtargetTest, learning_rate = 0.0001,
          num_epochs = 2000, minibatch_size = 32, print_cost = True):
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
    #cost = compute_cost(Z3, Y)
    
    cost = compute_cost_regularized(Z3, Y, parameters, lambd=0.1)
    
    #print('cost is:', cost) #issue might be optimizer so this should help figure out where nan is coming from
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
        
        #trying to print outputs
        '''
        Xxx = tf.placeholder(shape=[X_test.shape[0],None],dtype=tf.float32,name="Xxx")
        #print(Xxx)
        predictTest = forward_propagation(Xxx,parameters)
        classify = tf.nn.softmax(tf.transpose(predictTest))
        hopefull = pd.DataFrame(data=sess.run(classify, feed_dict={Xxx: X_test}))
        '''
        #end of trying to print outputs
        
        
        
        #print(sess.run(x, feed_dict = {Z3 : x}))
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: XtargetTrain, Y: YtargetTrain}))
        print("Test Accuracy:", accuracy.eval({X: XtargetTest, Y: YtargetTest}))
        
        return parameters
        #return parameters, hopefull




#%% EXECUTE TENSORFLOW NEURAL NETWORK 

# create_placeholders(11250, 2)
parameters, answers = model (XtargetTrain, YtargetTrain, XtargetTest, YtargetTest)

#%% save parameters of model to a file 

#%% 
#delete column 0... pretty sure column 1 is the prediction of yes... pretty sure
answers = answers.drop([0], axis=1)

#%%
#use id a index
idList = []
for x in range(len(submitTest)):
    idList.append(submitTest.iloc[x][2])
idSeries = pd.Series(idList)
#%% add id values as index 
answers = answers.set_index(idSeries)
        

#%% rename columns
answers = answers.rename(index=str, columns={1 : "is_iceberg"})
#not quite right, need to rename index as "id",not column 1
#%% rename index
answers.index.name = "id"

#df.index.name = 'foo'
#%% export as csv

answers.to_csv('/Users/brendontucker/KaggleData/StatoilCCORE/submissions/subTwo.csv')


#%% create submission
# predict(X, parameters) 

'''

X = tf.placeholder(shape=[X_train.shape[0],None],dtype=tf.float32,name="X")

predict = forward_propagation(X,hyperparameters,parameters)

classify = tf.nn.softmax(tf.transpose(predict))
session = tf.Session()
session.run(init)

print(session.run(classify, feed_dict={X: X_test[:,3]}))

'''

'''

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss)) 

'''


#or maybe just do it in the loop itself? 

Xxx = tf.placeholder(shape=[X_test.shape[0],None],dtype=tf.float32,name="Xxx")
print(Xxx)
predictTest = forward_propagation(Xxx,parameters)
classify = tf.nn.softmax(tf.transpose(predict))

init = tf.global_variables_initializer()
with tf.Session() as session:                    # Create a session and print the output
    session.run(init) 
    hopefull = pd.DataFrame(data=session.run(classify, feed_dict={Xxx: X_test}))
#print(sess.run(classify, feed_dict={Xx: XtargetTest}))
print(hopefull[1])

#%%

'''
tyring to get new loss function to work

This is the output of the working loss function

cost is of type: <class 'tensorflow.python.framework.ops.Tensor'> 
cost is: Tensor("Mean:0", shape=(), dtype=float32)

and new version gets

cost is of type: <class 'tensorflow.python.framework.ops.Tensor'> 
cost is: Tensor("Mean:0", shape=(), dtype=float32)
    
which is the same... but obviously wrong because my cost value is nan... 
    I think most common problem that results in nan is too high a learning rate?
    so lets lower it and see
    
    
print statement for cost in body of model for working version
cost is: Tensor("Mean:0", shape=(), dtype=float32)

and for broken version 
cost is: Tensor("Mean:0", shape=(), dtype=float32)

so surely the issue must be further down?


maybe preprocessing will help? 
nope 

maybe I need to run for longer?






printing outputs : https://stackoverflow.com/questions/40430186/tensorflow-valueerror-cannot-feed-value-of-shape-64-64-3-for-tensor-uplace

from example
ValueError: Cannot feed value of shape (64, 64, 3) for Tensor u'Placeholder:0', 
which has shape '(?, 64, 64, 3)'


Cannot feed value of shape (11250,) for Tensor 'X_1:0',

=>X does not have the right dimensions to fit into

which has shape '(11250, ?)'




[[ 0.57323956  0.42676041]
 [ 0.55073351  0.44926649]
 [ 0.55944043  0.4405596 ]
 [ 0.73255342  0.26744655]
 [ 0.75022084  0.24977909]
 [ 0.5478847   0.45211524]
 [ 0.69621283  0.30378711]
 [ 0.61005563  0.38994434]
 [ 0.66413742  0.33586264]
 [ 0.58855838  0.41144171]
 [ 0.63553393  0.36446607]
 [ 0.57203412  0.42796594]
 [ 0.63672274  0.36327732]
 [ 0.66789401  0.33210596]
 [ 0.63979203  0.360208  ]
 [ 0.62979633  0.37020367]
 [ 0.57863921  0.42136082]
 [ 0.63418597  0.36581403]
 [ 0.56350785  0.4364922 ]
 [ 0.6790055   0.32099447]
 
 
 
'''


#%% lololol at how badly I undersand tensorflow

 
 