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
for x in range(len(train)):
    

#%% CAN WE ADD SERIES TO CREATE A DF? 
testSeries = pd.Series(train.iloc[0][0])    
addSeries = pd.Series(train.iloc[1][0])
    
    
    
    
#%% TEST WITH HH ONLY 

