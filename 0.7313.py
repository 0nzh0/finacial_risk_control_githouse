# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from category_encoders import TargetEncoder

#读取数据
path = 'data/'

train = pd.read_csv(path + 'train.csv', index_col='id')
test = pd.read_csv(path + 'testA.csv', index_col='id')
# =============================================================================
# train.drop(['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine'], axis=1, inplace=True)
# =============================================================================
label = train.pop('isDefault')
test = test[train.columns]
s = train.apply(lambda x:x.dtype)

#数据预处理
for i in range(15):
    train['n'+str(i)].fillna(-1, inplace=True)
train['employmentLength'].fillna('unkown', inplace=True)
train.fillna(-1.0, inplace=True)

def model_lgb():
    lgb = LGBMRegressor(
                num_leaves=2**5-1, 
                reg_alpha=0.25,
                reg_lambda=0.25, 
                objective='mse',
                max_depth=-1, 
                learning_rate=0.05, 
                min_child_samples=5, 
                random_state=2019,
                n_estimators=1000,
                subsample=0.9, 
                colsample_bytree=0.7,
                )
    
# =============================================================================
#     lgb = LGBMRegressor(num_leaves=30,
#                         max_depth=5,
#                         learning_rate=.02,
#                         n_estimators=1100,
#                         subsample_for_bin=5000,
#                         min_child_samples=200,
#                         colsample_bytree=.2,
#                         reg_alpha=.1,
#                         reg_lambda=.1
#                         )
# =============================================================================
    return lgb

def model_xgb():
    xgb = XGBRegressor(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=500,
                objective='reg:linear',
                tree_method='hist',
                subsample=0.9, 
                colsample_bytree=0.7,
                min_child_weight=5,
                eval_metric='auc',
                )
    return xgb

#特征工程

#



kf = KFold(n_splits=5, shuffle=True, random_state=100)
devscore = []
for train_idx, valid_idx in kf.split(train.index):
    X_train = train.iloc[train_idx]
    X_valid = train.iloc[valid_idx]
    y_train = label.iloc[train_idx]
    y_valid = label.iloc[valid_idx]
    enc = TargetEncoder(cols=['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine'])
    X_train = enc.fit_transform(X_train, y_train)
    X_valid = enc.transform(X_valid)
    
    lgb = model_lgb()
    lgb.fit(X_train, y_train)
    pre_lgb = lgb.predict(X_valid)
    xgb = model_xgb()
    xgb.fit(X_train, y_train)
    pre_xgb = xgb.predict(X_valid)

    pre = pre_lgb * 0.5 + pre_xgb * 0.5
    fpr, tpr, thresholds = roc_curve(y_valid, pre)
    score = auc(fpr, tpr)
    devscore.append(score)
print(np.mean(devscore))

# =============================================================================
# X_train = train
# X_test = test
# y_train = label
# 
# enc = TargetEncoder(cols=['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine'])
# X_train = enc.fit_transform(X_train, y_train)
# X_test = enc.transform(X_test)
# 
# lgb = model_lgb()
# lgb.fit(X_train, y_train)
# lgb_pre = lgb.predict(X_test)
# 
# xgb = model_xgb()
# xgb.fit(X_train, y_train)
# xgb_pre = lgb.predict(X_test)
# 
# pre = 0.5 * lgb_pre + 0.5 * xgb_pre
# pd.Series(pre, name='isDefault', index=test.index).reset_index().to_csv('result.csv', index=False)
# =============================================================================

