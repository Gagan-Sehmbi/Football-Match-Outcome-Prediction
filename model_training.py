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
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

from joblib import dump, load


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
# SQL
df = pd.read_sql_table('Clean_Dataset', engine, index_col=0)
features = pd.read_sql_table('Features_Table', engine, index_col=0)
classes = pd.read_sql_table('Classes_Table', engine, index_col=0)

# LOCAL
#df = pd.read_csv('cleaned_dataset.csv', index_col=0)
#features = pd.read_csv('features.csv', index_col=0)
#classes = pd.read_csv('classes.csv', index_col=0)


# %% SPLIT AND SCALE DATASET
# SPLIT INPUT FEATURES FROM TARGET FEATURE
X = df.drop(columns=['Labels'])
y = df['Labels']

# SPLIT DATASET INTO TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
y_train = y_train.to_numpy().flatten()
y_test = y_test.to_numpy().flatten()

# SCALE INPUT VARIABLES
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
s = time.perf_counter()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('LR Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_test, y_pred=y_pred_lr, target_names=classes['0'].values))

# NB models
gnb = GaussianNB()
s = time.perf_counter()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print('GNB Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_test, y_pred=y_pred_gnb, target_names=classes['0'].values))

# neighbors models
knn = KNeighborsClassifier()
s = time.perf_counter()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('KNN Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_test, y_pred=y_pred_knn, target_names=classes['0'].values))

# tree models
dt = DecisionTreeClassifier()
s = time.perf_counter()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('DT Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_test, y_pred=y_pred_dt, target_names=classes['0'].values))

# ensemble models
rfc = RandomForestClassifier()
s = time.perf_counter()
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print('RFC Complete')
f = time.perf_counter()
print(f-s)
print(classification_report(y_true=y_test, y_pred=y_pred_rfc, target_names=classes['0'].values))


# %% TEST ALL MODELS
preds = [y_pred_lr, y_pred_gnb, y_pred_knn, y_pred_dt, y_pred_rfc]
idx = ['Logistic Regression', 'GNB', 'KNN', 'Decision Tree', 'Random Forest']
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

fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharey=True)
fig.suptitle('Comparing Model Performance on Validation Set')

sns.barplot(ax= axes[0], data=performance_df, x='Model', y='Precision', hue='Class')
axes[0].set_title('Precision')

sns.barplot(ax= axes[1], data=performance_df, x='Model', y='Recall', hue='Class')
axes[1].set_title('Recall')

sns.barplot(ax= axes[2], data=performance_df, x='Model', y='F_Score', hue='Class')
axes[2].set_title('F_Score')

plt.show()

# %% CHECK FEATURE IMPORTANCE FOR RFC MODEL

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
forest_importances = pd.Series(importances, index=features['0'])

fig, ax = plt.subplots(figsize=(15, 5))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")

plt.show()

# %% HYPERPARAMETER TUNING + K-FOLD CROSS VALIDATION (GRIDSEARCHCV)

pipe = Pipeline([('classifier' , RandomForestClassifier())])
# pipe = Pipeline([('classifier', RandomForestClassifier())])

# Create param grid.

param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(4,len(features)+1,5))}
]

# Create grid search object

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

# Fit on data

best_clf = clf.fit(X_train, y_train)

# Persist the model
model = best_clf.best_params_['classifier']
model.fit(X_train, y_train)
dump(model, 'classifier.joblib')

# %% HYPERPARAMETER TUNING + K-FOLD CROSS VALIDATION (MANUAL)

'''
def k_fold(dataset, n_splits: int = 5):
    chunks = np.array_split(dataset, n_splits)
    for i in range(n_splits):
        training = chunks[:i] + chunks[i + 1 :]
        training = np.concatenate(training)
        validation = chunks[i]

        yield training, validation

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# K-Fold evaluation
best_hyperparams, best_accuracy = None, 0
n_splits = 5
# Grid search goes first
for hyperparams in grid_search(random_grid):
    acc = 0
    # Instead of validation we use K-Fold
    for (X_train, X_test), (y_train, y_test) in zip(k_fold(X, n_splits), k_fold(y, n_splits)):

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        model = RandomForestClassifier(**hyperparams)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        fold_acc = accuracy_score(y_test, y_test_pred)
        acc += fold_acc
    # Take mean from all folds as final validation score
    total_acc = acc / n_splits
    print(f"H-Params: {hyperparams} Accuracy: {total_acc}")
    if total_acc > best_accuracy:
        best_accuracy = total_acc
        best_hyperparams = hyperparams

# And see our final results
print(f"Best loss: {best_accuracy}")
print(f"Best hyperparameters: {best_hyperparams}")
'''
