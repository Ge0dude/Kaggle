#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:42:50 2017

@author: brendontucker
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
color = sns.color_palette()

train_df = pd.read_csv('/Users/brendontucker/KaggleData/SherbankData/SherbankTrain.csv')
