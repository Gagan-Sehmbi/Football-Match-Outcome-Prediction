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
frame = pd.read_sql_table('Clean_Test_Dataset', engine, index_col=0)
df = frame.drop(columns=['Home_Team', 'Away_Team'])
features = pd.read_sql_table('Features_Table', engine, index_col=0)
classes = pd.read_sql_table('Classes_Table', engine, index_col=0)
params = pd.read_sql_table('Parameters_Table', engine, index_col=0)

# IMPORT MODEL

classifier = load('classifier.joblib')

# %% DATA PREPROCESSING
df.drop(columns=['Labels'], inplace=True)
data = (df.values - params['Mean'].values)/params['STD'].values

results = classifier.predict(data)

results_df = frame[['Home_Team', 'Away_Team']]

results_df['Home_Team_Prediction'] = results
results_df['Home_Team_Prediction'] = results_df['Home_Team_Prediction'].apply(lambda x: classes.loc[x, '0'])

results_df

# %%

# %%
