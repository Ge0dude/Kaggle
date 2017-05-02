# -*- coding: utf-8 -*-
"""
Spyder Editor


taken from https://www.kaggle.com/sudalairajkumar/sberbank-russian-housing-
market/simple-exploration-notebook-sberbank
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import model_selection, preprocessing
#import xgboost as xgb
color = sns.color_palette()

train_df = pd.read_csv('/Users/brendontucker/KaggleData/SherbankData/SherbankTrain.csv')

plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values)) 

'''need to understand how scatter is working better'''
#x = 4
#y = [10, 11, 12, 13]
#plt.scatter(range(x), y)
#okay, much clearer now 
sns.distplot(train_df.price_doc.values, kde=True)
sns.distplot(np.log(train_df.price_doc.values), kde=True, bins=50)

#sns.distplot(np.log(train_df.full_sq.values), kde=True)