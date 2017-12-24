#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:42:50 2017

@author: brendontucker

taken from https://www.kaggle.com/joparga3/in-depth-skewed\
-data-classif-93-recall-acc-now

clearly has taken a lot, if not plagarized from 
http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes\
-in-your-machine-learning-dataset/
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
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)

#splitting the independent and dependent variables
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']

#counting the instances of fraud and finding their location in the dataframe
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

#find the locations of the non-fraud cases
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# creating a new dataset with the arrays we just formed joined
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

#add variables to new dataset 
under_sample_data = data.iloc[under_sample_indices,:]




