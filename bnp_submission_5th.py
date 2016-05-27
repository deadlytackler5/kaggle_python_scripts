#!usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
# from sklearn.decomposition import PCA
from time import time
print('imported all modules')
######################################################################


def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features

######################################################################

def MungeData(train, test):

    features = train.columns[2:]
    print(type(features))
    for col in features:
        if((train[col].dtype == 'object') and (col!="v22")):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, train.target.values)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)
    return train, test

######################################################################
### Data loading
t0 = time()
train = pd.read_csv("D:/kaggle/BNP/Data/train.csv")
test = pd.read_csv("D:/kaggle/BNP/Data/test.csv")

print('Data loaded in %0.3fs' % (time()-t0))

train, test = MungeData(train,test)

target = train['target'].values
id_test = test['ID'].values

train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53',
                    'v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105',
                    'v107','v108','v109','v110','v116','v117','v118','v119','v123',
                    'v124','v128'],axis=1)
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53',
                  'v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105',
                  'v107','v108','v109','v110','v116','v117','v118','v119','v123',
                  'v124','v128'],axis=1)

######################################################################
### Clearing the data
t0 = time()

for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

print("clearing done in %0.3fs" % (time()-t0))
######################################################################
### train test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=214)
print("Train Test split completed")

### standard scalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler complted")
######################################################################
### ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV

clf = ExtraTreesClassifier(n_jobs= 4, random_state=214, verbose=True)
param_grid = {'n_estimators':[750],
              'max_features':[60],
              'criterion':['entropy'],
              'min_samples_split':[4],
              'max_depth': [40],
              'min_samples_leaf': [2, 4, 6, 8]}  # try --> [2, 4, 6, 8]
grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=True)
start = time()
grid_search.fit(X_train_scaled, y_train)
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      # % (time()-start, len(grid_search.grid_scores_)))
# print(grid_search.grid_scores_)
print(grid_search.best_params_)
# best n_setimators for [550,650,750,850,950,1150]
# {'criterion': 'entropy',
#  'max_depth': 40,
#  'max_features': 60,
#  'min_samples_leaf': 2,
#  'min_samples_split': 4,
#  'n_estimators': 750}
# Log Loss: 0.462142066786

y_pred = grid_search.predict_proba(X_test_scaled)

### log loss
from sklearn.metrics import log_loss
print("Log Loss:", log_loss(y_test, y_pred))
######################################################################
### work on main data
# scaler = StandardScaler()
# scaler = scaler.fit(train)
# train_scaled = scaler.transform(train)
# test_scaled = scaler.transform(test)

# params = {'criterion': 'entropy',
#           'max_depth': 40,
#           'max_features': 60,
#           'min_samples_leaf': 2,
#           'min_samples_split': 4,
#           'n_estimators': 750}

# clf = ExtraTreesClassifier(n_estimators=750, max_features=60,
#                            criterion= 'entropy',min_samples_split= 4,
#                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1)
# clf.fit(train_scaled, target)
# y_pred = clf.predict_proba(test_scaled)
# pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('extra_trees1.csv',index=False)





