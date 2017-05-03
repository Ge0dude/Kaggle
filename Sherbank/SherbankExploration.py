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
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()

train_df = pd.read_csv('/Users/brendontucker/KaggleData/SherbankData/SherbankTrain.csv')

#plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values)) 

'''need to understand how scatter is working better'''
#x = 4
#y = [10, 11, 12, 13]
#plt.scatter(range(x), y)
#okay, much clearer now 
#sns.distplot(train_df.price_doc.values, kde=True)
#sns.distplot(np.log(train_df.price_doc.values), kde=True, bins=50)

#sns.distplot(np.log(train_df.full_sq.values), kde=True)

#train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
#
#for f in train_df.columns:
#    if train_df[f].dtype=='object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(list(train_df[f].values)) 
#        train_df[f] = lbl.transform(list(train_df[f].values))
'''        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

'''


''' #should take approx. one minute to run
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

xgb.plot_importance(model, max_num_features=100, height=0.8) #ax=ax)
'''


ulimit = np.percentile(train_df.price_doc.values, 99.5) #is this 3 sigma?
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit



col = "full_sq"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

sns.jointplot(x=np.log1p(train_df.full_sq.values), 
              y=np.log1p(train_df.price_doc.values), kind="hex")
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of Total area in square metre', fontsize=12)

#dig into a lowervariable, look for splits, split, retry xgboost

col = "life_sq"
train_df[col].fillna(0, inplace=True)
ulimit = np.percentile(train_df[col].values, 95)
llimit = np.percentile(train_df[col].values, 5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

sns.jointplot(x=np.log1p(train_df.life_sq.values), 
              y=np.log1p(train_df.price_doc.values), 
              kind='scatter', size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of living area in square metre', fontsize=12)

testCol = train_df['life_sq']

sns.distplot(testCol, kde=True) #this shows something like 16% have a value of 
#0. So I think it would be good to seperate that data into two groups, 0 and 
#then the rest














