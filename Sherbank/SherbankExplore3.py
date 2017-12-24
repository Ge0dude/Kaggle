#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:56:49 2017

@author: brendontucker
"""

# %% LIBRARY IMPORTS

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
color = sns.color_palette()

# %% DATA INPUT
df = pd.read_csv('/Users/brendontucker/KaggleData/SherbankData/SherbankTrain.csv')

# %% COLUMN EXPLORATION
colsList = list(df.columns.values)
# %% TSCATTER PLOT

plt.figure(figsize=(8,6))
plt.scatter(range(df.shape[0]), np.sort(df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()

# %% DISTRIBUTION 
plt.figure(figsize=(12,8))
sns.distplot(df.price_doc.values, bins=20, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()

# %% LOGNORM PLOT

sns.distplot(np.log(df.price_doc.values), bins=50, kde=True)