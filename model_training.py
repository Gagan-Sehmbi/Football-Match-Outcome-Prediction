# %% IMPORT LIBRARIES
import time

import numpy as np
import pandas as pd
import seaborn as sns

import pymysql
from sqlalchemy import create_engine, inspect
from login import football_result_prediction_db_details

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support, classification_report


# %% IMPORT DATA FROM DATABASE
# CONNECT TO DATABASE
DATABASE_TYPE = 'mysql'
DBAPI = 'pymysql'
ENDPOINT = 'football-results-db-instance.cfvpwrpdvrbp.eu-west-2.rds.amazonaws.com'
USER = football_result_prediction_db_details()['User Name']
PASSWORD = football_result_prediction_db_details()['Password']
PORT = int(3306)
DATABASE = 'football_results_db'

engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
inspector = inspect(engine)

engine.connect()

# %% IMPORT DATA
X_train = pd.read_sql_table('Training_Inputs_Table', engine, index_col='Unnamed: 0')
X_train.reset_index(drop=True, inplace=True)

y_train = pd.read_sql_table('Training_Outputs_Table', engine, index_col='Unnamed: 0')
y_train.reset_index(drop=True, inplace=True)

X_test = pd.read_sql_table('Testing_Inputs_Table', engine, index_col='Unnamed: 0')
X_test.reset_index(drop=True, inplace=True)

y_test = pd.read_sql_table('Testing_Outputs_Table', engine, index_col='Unnamed: 0')
y_test.reset_index(drop=True, inplace=True)

classes = pd.read_sql_table('Classes_Table', engine, index_col='Unnamed: 0')
classes.reset_index(drop=True, inplace=True)
#classes = list(classes.values)

# %% BASELINE MODEL PERFORMANCE (LOGISTIC REGRESSION)
lr = LogisticRegression()

s = time.perf_counter()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('LR Complete')
f = time.perf_counter()
print(f-s)

print('Performance of Logistic Regression Classifier (Train)\n')
print(classification_report(y_true=y_train, y_pred=lr.predict(X_train), target_names=classes['0'].values))

print('Performance of Logistic Regression Classifier (Test)\n')
print(classification_report(y_true=y_test, y_pred=y_pred_lr, target_names=classes['0'].values))


# %% COMPARE MODEL PERFORMANCES
# linear models
lr = LogisticRegression()
sgd = SGDClassifier()

# SVM models
svm = SVC()

# NB models
gnb = GaussianNB()

# neighbors models
knn = KNeighborsClassifier()

# tree models
dt = DecisionTreeClassifier()

# ensemble models
rfc = RandomForestClassifier()

# TRAIN ALL MODELS
s = time.perf_counter()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('LR Complete')
f = time.perf_counter()
print(f-s)

s = time.perf_counter()
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print('SGD Complete')
f = time.perf_counter()
print(f-s)

s = time.perf_counter()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print('SVM Complete')
f = time.perf_counter()
print(f-s)

s = time.perf_counter()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print('GNB Complete')
f = time.perf_counter()
print(f-s)

s = time.perf_counter()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('KNN Complete')
f = time.perf_counter()
print(f-s)

s = time.perf_counter()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('DT Complete')
f = time.perf_counter()
print(f-s)

s = time.perf_counter()
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print('RFC Complete')
f = time.perf_counter()
print(f-s)

# %% TEST ALL MODELS
preds = [y_pred_lr, y_pred_sgd, y_pred_svm, y_pred_gnb, y_pred_knn, y_pred_dt, y_pred_rfc]

idx = ['Linear Regression', 'SGD', 'SVM', 'GNB', 'KNN', 'Decision Tree', 'Random Forest']
cols = ['Precision', 'Recall', 'F_Score', 'Support']

performance_df = pd.DataFrame(index=idx, columns=cols)

for pred, ids in zip(preds, idx):
    result = precision_recall_fscore_support(y_true=y_test, y_pred=pred)
    precision, recall, f_score, support = result
    performance_df.loc[ids, 'Precision'] = precision
    performance_df.loc[ids, 'Recall'] = recall
    performance_df.loc[ids, 'F_Score'] = f_score
    performance_df.loc[ids, 'Support'] = support

unnested_lst = []
for col in performance_df.columns:
    unnested_lst.append(performance_df[col].apply(pd.Series).stack())
performance_df = pd.concat(unnested_lst, axis=1, keys=performance_df.columns)

performance_df.reset_index(inplace=True)
performance_df.rename(columns={'level_0':'Model', 'level_1':'Class'}, inplace=True)
performance_df['Class'] = performance_df['Class'].apply(lambda x: classes['0'][x])

# %% VISUALISE RESULTS
sns.barplot(data=performance_df, x='Model', y='Precision', hue='Class')

sns.barplot(data=performance_df, x='Model', y='Recall', hue='Class')

sns.barplot(data=performance_df, x='Model', y='F_Score', hue='Class')


# %%

