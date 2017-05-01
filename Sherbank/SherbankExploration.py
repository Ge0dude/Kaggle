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
 