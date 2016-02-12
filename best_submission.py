# Kaggle Home-Site
# Arun Kr. Khattri
# arun.kr.khattri@gmail.com

import pandas as pd
import numpy as np
import sys
sys.path.append("C:\\Anaconda3\\Lib\\site-packages\\xgboost\\wrapper")
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest

seed = 2140528
# train and test data 
train = pd.read_csv('C:/Users/nakedgun/Google Drive/python/Kaggle_Homesite_Quote_Conversion/train.csv')
test = pd.read_csv('C:/Users/nakedgun/Google Drive/python/Kaggle_Homesite_Quote_Conversion/test.csv')

# features & response
y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop(['QuoteNumber'], axis=1)

# Data manipulation
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek


test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

# replacing NAs with -1. 
train = train.fillna(-1)
test = test.fillna(-1)

# transform non-numerical labels  to numerical labels.
for f in train.columns:
    if train[f].dtype=='object':
        # print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

# train-test split
train_train,train_test,y_train,y_test = train_test_split(train,y,random_state=0)


clf = xgb.XGBClassifier(n_estimators=810, max_depth=5, subsample=.80,colsample_bytree=0.70,reg_alpha=0.0001)
clf.fit(train_train,y_train,early_stopping_rounds=20, eval_metric='auc', eval_set=[(train_test,y_test)])

# Prediction
preds = clf.predict_proba(test)[:,1]

# Submission
sample = pd.read_csv('C:/Users/nakedgun/Google Drive/python/Kaggle_Homesite_Quote_Conversion/Submissions/Python/submission_csv/sample_submission.csv')
# change values in original sample files with preds, save as new file.
sample.QuoteConversion_Flag = preds
sample.to_csv('bestSubmission_9.csv', index=False)
