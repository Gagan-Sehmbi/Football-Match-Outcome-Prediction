# %% IMPORT LIBRARIES
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pymysql
from sqlalchemy import create_engine, inspect
from login import football_result_prediction_db_details

from joblib import load


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
df = pd.read_sql_table('Clean_Test_Dataset', engine, index_col=0)
features = pd.read_sql_table('Features_Table', engine, index_col=0)
classes = pd.read_sql_table('Classes_Table', engine, index_col=0)
params = pd.read_sql_table('Parameters_Table', engine, index_col=0)

# IMPORT MODEL

classifier = load('classifier.joblib')

# %% DATA PREPROCESSING
df.drop(columns=['Labels'], inplace=True)
data = (df.values - params['Mean'].values)/params['STD'].values

results = classifier.predict(data)

df['Prediction'] = results

# %%

df['Home_Team_Win_Streak'].value_counts()

# %% IMPORT LIBRARIES
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pymysql
from sqlalchemy import create_engine, inspect
from login import football_result_prediction_db_details

from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

classifier = load('classifier.joblib')

df = pd.read_sql_table('Clean_Train_Dataset', engine, index_col=0)
data = (df.drop(columns=['Labels']).values - params['Mean'].values)/params['STD'].values
results = classifier.predict(data)
df['Prediction'] = results

print(confusion_matrix(y_true=df['Labels'].values, y_pred=df['Prediction']))
print(classification_report(y_true=df['Labels'].values, y_pred=df['Prediction']))

df
# %%
df['Home_Team_Win_Streak'].value_counts()
# %%
df.loc[df['Season'] ==2022, 'Home_Team_Win_Streak'].value_counts()
# %%