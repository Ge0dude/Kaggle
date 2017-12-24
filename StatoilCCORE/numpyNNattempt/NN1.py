#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 06:15:30 2017

@author: brendontucker
"""
#%% IMPORTS
import numpy as np
import pandas as pd

#%% LOAD TRAIN
train = pd.read_json("/Users/brendontucker/KaggleData/StatoilCCORE/data-1/processed/train.json")
#%% LOAD TEST FILE 
test = pd.read_json("/Users/brendontucker/KaggleData/StatoilCCORE/data/processed/test.json")
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

#%% TEST WITH HH ONLY 
