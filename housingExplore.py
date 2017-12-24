#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 07:15:47 2017

@author: brendontucker
"""
# %% PACKAGE IMPORTS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# %% DATA READ

df_train = pd.read_csv('/Users/brendontucker/KaggleData/HousePrices/train.csv')

# %% COLUMN NAMES

df_train.columns

# %% DESCRIPTION OF THE TARGET VARIABLE 

df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])

print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#seems to have a fat tail, maybe lognorm is neeeded?

# %% LOG OF TARGET VARIABLE
sns.distplot(np.log(df_train['SalePrice'])) #looks a lot better--more norm
np.log(df_train['SalePrice']).describe()

print("Kurtosis after lognorm: %f" % np.log(df_train['SalePrice']).skew())
print("Skew after lognorm: %f" % np.log(df_train['SalePrice']).kurt())