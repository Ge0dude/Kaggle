#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:06:08 2018

more or less a copy of TensorFlow1 but without most of the comments--code 
added to here only if it is working and matching the model form tutorial 

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
    
XtargetTrain = np.zeros(shape=(len(train),11250))
for x in range(len(train)):
    XtargetTrain[x] = train.iloc[x][0] + train.iloc[x][1]
XtargetTrain = XtargetTrain.T

#%% Y TARGET VARIABLE SET UP 

YtargetTrain = np.zeros(shape=(len(train),1))
for x in range(len(train)):
    YtargetTrain[x] = train.iloc[x][4]
#had to add .astype to fix convertohotones error 
YtargetTrain = YtargetTrain.astype('int64')

#%% converting to one-hot representation 

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

YtargetTrain = convert_to_one_hot(YtargetTrain, 2)

#%% samefor test set 

XtargetTest = np.zeros(shape=(len(test),11250))
for x in range(len(test)):
    XtargetTest[x] = test.iloc[x][0] + test.iloc[x][1]
XtargetTest = XtargetTest.T

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

