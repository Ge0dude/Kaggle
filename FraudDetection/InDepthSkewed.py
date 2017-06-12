#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:42:50 2017

@author: brendontucker

taken from https://www.kaggle.com/joparga3/in-depth-skewed\
-data-classif-93-recall-acc-now
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
color = sns.color_palette()

data = pd.read_csv('/Users/brendontucker/KaggleData/CreditCardFraud/creditcard.csv')

count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)