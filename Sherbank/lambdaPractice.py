#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:56:09 2017

@author: brendontucker

file created to begin to play with lambdas and work through a section of 
SherbankExplore code taken from Kaggle
"""
# %%IMPORT
import pandas as pd
import numpy as np
# %% FUNCTION
lambda x: x[:4]+x[5:7]

# %% DATA 

'''from docs
np.random.randint(5, size=(2, 4))
'''
testRand = np.random.randint(100, high=1000, size=(5,5))
df2 =pd.DataFrame(testRand)
df3 =pd.DataFrame(np.random.randn(10, 5),
                 columns=['a', 'b', 'c', 'd', 'e'])
testCol = df2[0]

# %% TYPE ALTERATION 
testColString = testCol.apply(lambda z: str(z)) '''hey, I just made my first
lambda! Look at me go!'''



# %% IMPLEMENTATION
lambdaTest = testColString.apply(lambda x: x[:2]+x[2:3])

# %% ALTERNATE TESTING OUTSIDE OF DATAFRAME
tester = '1234567'
for string in map(lambda x: x[:4]+x[5:7], tester):
    print(string)
newTester = str(map(lambda x: x[:4]+x[5:7], tester))

