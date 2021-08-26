# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:56:31 2019

@author: pc2
"""

#import package
import re
from pprint import pprint
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics         import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing   import StandardScaler
from imblearn.over_sampling  import SMOTE
from sklearn.linear_model    import LogisticRegressionCV
from sklearn.ensemble        import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier

#데이터 불러오기
cd C:\Users\pc2\Desktop\UNIST\졸업 논문
ls
data=pd.read_csv('dataset12.csv')

#결측치 제거
df=pd.DataFrame(data)
df=df.dropna(axis=0)

data_corr= df.drop(['direction','date','SPM'], axis=1)
corr=data_corr.corr()
%matplotlib inline   
import matplotlib.pyplot as plt 
import seaborn as sns    


plt.figure(figsize=(12,12))
sns.heatmap(data = data_corr.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')


#SPM Close (1) data 확인
dfN=df[df.SPM!=0]


###################################데이터 분석#################################################

#Sampling
dfX= df.drop(['direction','date','SPM'], axis=1)
dfy= df['SPM']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(dfX, dfy, test_size=0.3, random_state=0)

#OVERSampling
pip install -U imbalanced-learn
!pip install imblearn
from imblearn.over_sampling import *
from imblearn.under_sampling import *

X_samp_over, y_samp_over = RandomOverSampler(random_state=0).fit_sample(dfX, dfy)
X_samp_over_train,X_samp_over_test, y_samp_over_train, y_samp_over_test = train_test_split(X_samp_over, y_samp_over, test_size=0.3, random_state=0)

X_samp_under, y_samp_under = RandomUnderSampler(random_state=0).fit_sample(dfX, dfy)
X_samp_under_train,X_samp_under_test, y_samp_under_train, y_samp_under_test = train_test_split(X_samp_under, y_samp_under, test_size=0.3, random_state=0)

##SMOTE
X_samp_SMOTE, y_samp_SMOTE = SMOTE(random_state=4).fit_sample(dfX, dfy)
X_samp_SMOTE_train,X_samp_SMOTE_test, y_samp_SMOTE_train, y_samp_SMOTE_test = train_test_split(X_samp_SMOTE, y_samp_SMOTE, test_size=0.3, random_state=0)

##################################Linear Regression###########################################
LR = LogisticRegressionCV(cv=10)
LR.fit(X_train, y_train)
LR.pred = LR.predict(X_test)
accuracy = accuracy_score(y_test, LR.pred)
fpr, tpr, thresholds = roc_curve(y_test, LR.pred)
auc_score = auc(fpr, tpr)

print("Model    : Logistic Regression(CV=10)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, LR.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of Linear Regresion")
plt.show()


LR_over = LogisticRegressionCV(cv=10)
LR_over.fit(X_samp_over_train, y_samp_over_train)
LR_over.pred = LR_over.predict(X_samp_over_test)
accuracy = accuracy_score(y_samp_over_test, LR_over.pred)
fpr, tpr, thresholds = roc_curve(y_samp_over_test, LR_over.pred)
auc_score = auc(fpr, tpr)

print("Model    : Logistic Regression(Over-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_over_test, LR_over.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of Linear Regresion_over")
plt.show()

LR_under = LogisticRegressionCV(cv=10)
LR_under.fit(X_samp_under_train, y_samp_under_train)
LR_under.pred = LR_under.predict(X_samp_under_test)
accuracy = accuracy_score(y_samp_under_test, LR_under.pred)
fpr, tpr, thresholds = roc_curve(y_samp_under_test, LR_under.pred)
auc_score = auc(fpr, tpr)

print("Model    : Logistic Regression(under-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_under_test, LR_under.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of Linear Regresion_under")
plt.show()

LR_SMOTE = LogisticRegressionCV(cv=10)
LR_SMOTE.fit(X_samp_SMOTE_train, y_samp_SMOTE_train)
LR_SMOTE.pred = LR_over.predict(X_samp_SMOTE_test)
accuracy = accuracy_score(y_samp_SMOTE_test, LR_SMOTE.pred)
fpr, tpr, thresholds = roc_curve(y_samp_SMOTE_test, LR_SMOTE.pred)
auc_score = auc(fpr, tpr)

print("Model    : Logistic Regression(SMOTE-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_SMOTE_test, LR_SMOTE.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of Linear Regresion_SMOTE")
plt.show()

#####Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
##5-fold validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(LR, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))
##10-fold validation
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(LR, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))
    
##################################Random Forest 분석###########################################
from sklearn.ensemble import RandomForestClassifier
#Best Parameter
param_grid = {"n_estimators" :[25,50, 75, 100, 125, 150, 175, 200],
              "min_samples_split":[2, 3],
              "min_samples_leaf":[1, 2],
              "max_depth":[1,3, 5, 7, 10]}

rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                              param_grid=param_grid, 
                              cv=5, verbose=2, n_jobs=5)

rf_grid_search.fit(X_train, y_train)
rf_grid_search.best_params_

forest=RandomForestClassifier(n_estimators=200,
                                 min_samples_split=3,
                                 min_samples_leaf=2,
                                 max_depth=30,
                                 random_state=42)
forest.fit(X_train, y_train)
forest.pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, forest.pred)
fpr, tpr, thresholds = roc_curve(y_test, forest.pred)
auc_score = auc(fpr, tpr)

print("Model    : Random Forest(Grid Search Tuning)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)



fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, forest.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of RandomForest")
plt.show()

#oversamp
param_grid = {"n_estimators" :[25,50, 75, 100, 125, 150, 175, 200],
              "min_samples_split":[2, 3],
              "min_samples_leaf":[1, 2],
              "max_depth":[1,3, 5, 7, 10]}

rf_grid_search_over = GridSearchCV(estimator=RandomForestClassifier(),
                              param_grid=param_grid, 
                              cv=5, verbose=2, n_jobs=-1)

rf_grid_search_over.fit(X_samp_over_train, y_samp_over_train)
rf_grid_search_over.best_params_

RF_over=RandomForestClassifier(n_estimators=200,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 max_depth=40,
                                 random_state=42)

RF_over.fit(X_samp_over_train, y_samp_over_train)
RF_over.pred = RF_over.predict(X_samp_over_test)
accuracy = accuracy_score(y_samp_over_test, RF_over.pred)
fpr, tpr, thresholds = roc_curve(y_samp_over_test, RF_over.pred)
auc_score = auc(fpr, tpr)

print("Model    : Random Forest(Over-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_over_test, RF_over.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of RandomForest_over")
plt.show()
print("특성 중요도:\n{}".format(RF_over.feature_importances_))

RF_under=RandomForestClassifier(n_estimators=200,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 max_depth=40,
                                 random_state=42)

RF_under.fit(X_samp_under_train, y_samp_under_train)
RF_under.pred = RF_under.predict(X_samp_under_test)
accuracy = accuracy_score(y_samp_under_test, RF_under.pred)
fpr, tpr, thresholds = roc_curve(y_samp_under_test, RF_under.pred)
auc_score = auc(fpr, tpr)

print("Model    : Random Forest(under-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)


fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_under_test, RF_under.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of RandomForest_under")
plt.show()

#oversamp
param_grid = {"n_estimators" :[25,50, 75, 100, 125, 150, 175, 200],
              "min_samples_split":[2, 3],
              "min_samples_leaf":[1, 2],
              "max_depth":[1,3, 5, 7, 10]}

rf_grid_search_SMOTE = GridSearchCV(estimator=RandomForestClassifier(),
                              param_grid=param_grid, 
                              cv=5, verbose=2, n_jobs=5)

rf_grid_search_SMOTE.fit(X_samp_SMOTE_train, y_samp_SMOTE_train)
rf_grid_search_SMOTE.best_params_

RF_SMOTE=RandomForestClassifier(n_estimators=150,
                                 min_samples_split=3,
                                 min_samples_leaf=1,
                                 max_depth=20,
                                 random_state=42)

RF_SMOTE.fit(X_samp_SMOTE_train, y_samp_SMOTE_train)
RF_SMOTE.pred = RF_SMOTE.predict(X_samp_SMOTE_test)
accuracy = accuracy_score(y_samp_SMOTE_test, RF_SMOTE.pred)
fpr, tpr, thresholds = roc_curve(y_samp_SMOTE_test, RF_SMOTE.pred)
auc_score = auc(fpr, tpr)

print("Model    : Random Forest(SMOTE-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_SMOTE_test, RF_SMOTE.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of RandomForest_SMOTE")
plt.show()


#####Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
##5-fold validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(forest, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))

##10-fold validation
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(forest, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))


########################################XGBoost#########################################
param_grid = {"learning_rate": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75],
              "max_depth":   [2, 3, 5, 7, 10],
              "max_features":[2, 3],
              "n_estimators":[50 ,100, 150, 175, 200, 250, 300]}

xgb_grid_search = GridSearchCV(estimator=XGBClassifier(),
                                param_grid=param_grid, 
                                cv=5, verbose=2, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
xgb_grid_search.best_params_
XGB_tuned = XGBClassifier(n_estimators=200,
                              learning_rate=0.25,
                              max_depth=20,
                              max_features=2,
                              random_state=42)
XGB_tuned.fit(X_train, y_train)
XGB_tuned.pred = XGB_tuned.predict(X_test)
accuracy = accuracy_score(y_test, XGB_tuned.pred)
fpr, tpr, thresholds = roc_curve(y_test, XGB_tuned.pred)
auc_score = auc(fpr, tpr)

print("Model    : XGBoost (Grid Search Tuning)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, XGB_tuned.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of XGB")
plt.show()


xgb_grid_search_over = GridSearchCV(estimator=XGBClassifier(),
                                param_grid=param_grid, 
                                cv=5, verbose=2, n_jobs=-1)

xgb_grid_search_over.fit(X_samp_over_train, y_samp_over_train)
xgb_grid_search_over.best_params_



XGB_over = XGBClassifier(n_estimators=200,
                              learning_rate=0.25,
                              max_depth=30,
                              max_features=2,
                              random_state=42)

XGB_over.fit(X_samp_over_train, y_samp_over_train)
XGB_over.pred = XGB_over.predict(X_samp_over_test)
accuracy = accuracy_score(y_samp_over_test, XGB_over.pred)
fpr, tpr, thresholds = roc_curve(y_samp_over_test, XGB_over.pred)
auc_score = auc(fpr, tpr)

print("Model    : XGBoost (Over-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_over_test, XGB_over.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of XGB_Over")
plt.show()
print("특성 중요도:\n{}".format(XGB_over.feature_importances_))


XGB_under = XGBClassifier(n_estimators=200,
                              learning_rate=0.25,
                              max_depth=30,
                              max_features=2,
                              random_state=42)

XGB_under.fit(X_samp_under_train, y_samp_under_train)
XGB_under.pred = XGB_under.predict(X_samp_under_test)
accuracy = accuracy_score(y_samp_under_test, XGB_under.pred)
fpr, tpr, thresholds = roc_curve(y_samp_under_test, XGB_under.pred)
auc_score = auc(fpr, tpr)

print("Model    : XGBoost (under-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_under_test, XGB_under.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of XGB_under")
plt.show()


xgb_grid_search_SMOTE = GridSearchCV(estimator=XGBClassifier(),
                                param_grid=param_grid, 
                                cv=5, verbose=2, n_jobs=-1)


xgb_grid_search_SMOTE.fit(X_samp_SMOTE_train, y_samp_SMOTE_train)
xgb_grid_search_over.best_params_


XGB_SMOTE = XGBClassifier(n_estimators=200,
                              learning_rate=0.25,
                              max_depth=20,
                              max_features=2,
                              random_state=42)
XGB_SMOTE.fit(X_samp_SMOTE_train, y_samp_SMOTE_train)
XGB_SMOTE.pred = XGB_SMOTE.predict(X_samp_SMOTE_test)
accuracy = accuracy_score(y_samp_SMOTE_test, XGB_SMOTE.pred)
fpr, tpr, thresholds = roc_curve(y_samp_SMOTE_test, XGB_SMOTE.pred)
auc_score = auc(fpr, tpr)

print("Model    : XGBoost (SMOTE-sampling)")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_SMOTE_test, XGB_SMOTE.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of XGB_SMOTE")
plt.show()
##5-fold validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(XGB_tuned, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))

##10-fold validation
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(XGB_tuned, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))


        ####SVM

from sklearn.svm import SVC

svm=SVC(C=100,gamma=0.001).fit(X_train, y_train)

svm.pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, svm.pred)
fpr, tpr, thresholds = roc_curve(y_test, svm.pred)
auc_score = auc(fpr, tpr)

print("Model    : svm")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, svm.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of svm")
plt.show()

##SVM Over
svmover=SVC(kernel='rbf',C=100,gamma=0.001).fit(X_samp_over_train, y_samp_over_train)

svmover.pred = svmover.predict(X_samp_over_test)
accuracy = accuracy_score(y_samp_over_test, svmover.pred)
fpr, tpr, thresholds = roc_curve(y_samp_over_test, svmover.pred)
auc_score = auc(fpr, tpr)

print("Model    : svm")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_over_test, svmover.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of svmover")
plt.show()

##SVM SMOTE
svmSMOTE=SVC(kernel='rbf',C=100,gamma=0.001).fit(X_samp_SMOTE_train, y_samp_SMOTE_train)

svmSMOTE.pred = svmSMOTE.predict(X_samp_SMOTE_test)
accuracy = accuracy_score(y_samp_SMOTE_test, svmSMOTE.pred)
fpr, tpr, thresholds = roc_curve(y_samp_SMOTE_test, svmSMOTE.pred)
auc_score = auc(fpr, tpr)

print("Model    : svmSMOTE")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_over_test, svmSMOTE.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of svmSMOTE")
plt.show()
##5-fold validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(svm, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))

##10-fold validation
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(svm, dfX, dfy, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(kfold, scores))
    
        ####SVM

from sklearn.svm import SVC

svm=SVC(kernel='rbf',C=1000,gamma=0.001).fit(X_train, y_train)

svm.pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, svm.pred)
fpr, tpr, thresholds = roc_curve(y_test, svm.pred)
auc_score = auc(fpr, tpr)

print("Model    : svm")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, svm.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of svm")
plt.show()

##SVM Over
svmover=SVC(kernel='rbf',C=1000,gamma=0.001).fit(X_samp_over_train, y_samp_over_train)

svmover.pred = svmover.predict(X_samp_over_test)
accuracy = accuracy_score(y_samp_over_test, svmover.pred)
fpr, tpr, thresholds = roc_curve(y_samp_over_test, svmover.pred)
auc_score = auc(fpr, tpr)

print("Model    : svm")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_over_test, svmover.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of svmover")
plt.show()

svmunder=SVC(kernel='rbf',C=1000,gamma=0.001).fit(X_samp_under_train, y_samp_under_train)

svmunder.pred = svmunder.predict(X_samp_under_test)
accuracy = accuracy_score(y_samp_under_test, svmunder.pred)
fpr, tpr, thresholds = roc_curve(y_samp_under_test, svmunder.pred)
auc_score = auc(fpr, tpr)

print("Model    : svm")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_under_test, svmunder.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of svmover")
plt.show()

##SVM SMOTE
svmSMOTE=SVC(kernel='rbf',C=1000,gamma=0.001).fit(X_samp_SMOTE_train, y_samp_SMOTE_train)

svmSMOTE.pred = svmSMOTE.predict(X_samp_SMOTE_test)
accuracy = accuracy_score(y_samp_SMOTE_test, svmSMOTE.pred)
fpr, tpr, thresholds = roc_curve(y_samp_SMOTE_test, svmSMOTE.pred)
auc_score = auc(fpr, tpr)

print("Model    : svmSMOTE")
print("Accuracy :", accuracy)
print("AUC score:", auc_score)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_samp_over_test, svmSMOTE.pred),
                                figsize=(8, 8),
                                show_absolute=True,
                                show_normed=False,
                                colorbar=False, class_names=["Open", "Close"])
plt.title("Confusion Matrix of svmSMOTE")
plt.show()