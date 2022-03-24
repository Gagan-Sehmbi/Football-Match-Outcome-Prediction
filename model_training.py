# %% IMPORT LIBRARIES
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pymysql
from sqlalchemy import create_engine, inspect
from login import football_result_prediction_db_details

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
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
df = pd.read_sql_table('Clean_Dataset', engine, index_col=0)
features = pd.read_sql_table('Features_Table', engine, index_col=0)
classes = pd.read_sql_table('Classes_Table', engine, index_col=0)

# %% SPLIT AND SCALE DATASET
# SPLIT INPUT FEATURES FROM TARGET FEATURE
X = df.drop(columns=['Labels'])
y = df['Labels']

# SPLIT DATASET INTO TRAIN AND TEST SETS
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
y_train = y_train.to_numpy().flatten()
y_val = y_val.to_numpy().flatten()

# SCALE INPUT VARIABLES
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

# %% BASELINE MODEL PERFORMANCE (LOGISTIC REGRESSION)
lr = LogisticRegression()

s = time.perf_counter()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
print('LR Complete')
f = time.perf_counter()
print(f-s)

print('Performance of Logistic Regression Classifier (Train)\n')
print(classification_report(y_true=y_train, y_pred=lr.predict(X_train), target_names=classes['0'].values))

print('Performance of Logistic Regression Classifier (Test)\n')
print(classification_report(y_true=y_val, y_pred=y_pred_lr, target_names=classes['0'].values))


# %% COMPARE MODEL PERFORMANCES
# linear models
lr = LogisticRegression()
s = time.perf_counter()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
print('LR Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_val, y_pred=y_pred_lr, target_names=classes['0'].values))

# NB models
gnb = GaussianNB()
s = time.perf_counter()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_val)
print('GNB Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_val, y_pred=y_pred_gnb, target_names=classes['0'].values))

# neighbors models
knn = KNeighborsClassifier()
s = time.perf_counter()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_val)
print('KNN Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_val, y_pred=y_pred_knn, target_names=classes['0'].values))

# tree models
dt = DecisionTreeClassifier()
s = time.perf_counter()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)
print('DT Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_val, y_pred=y_pred_dt, target_names=classes['0'].values))

# ensemble models
rfc = RandomForestClassifier()
s = time.perf_counter()
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_val)
print('RFC Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_val, y_pred=y_pred_rfc, target_names=classes['0'].values))


# %% TEST ALL MODELS
preds = [y_pred_lr, y_pred_gnb, y_pred_knn, y_pred_dt, y_pred_rfc]
idx = ['Logistic Regression', 'GNB', 'KNN', 'Decision Tree', 'Random Forest']
cols = ['Precision', 'Recall', 'F_Score', 'Support']

performance_df = pd.DataFrame(index=idx, columns=cols)

for pred, ids in zip(preds, idx):
    result = precision_recall_fscore_support(y_true=y_val, y_pred=pred)
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

fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharey=True)
fig.suptitle('Comparing Model Performance on Validation Set')

sns.barplot(ax= axes[0], data=performance_df, x='Model', y='Precision', hue='Class')
axes[0].set_title('Precision')

sns.barplot(ax= axes[1], data=performance_df, x='Model', y='Recall', hue='Class')
axes[1].set_title('Recall')

sns.barplot(ax= axes[2], data=performance_df, x='Model', y='F_Score', hue='Class')
axes[2].set_title('F_Score')

plt.show()

# %%

