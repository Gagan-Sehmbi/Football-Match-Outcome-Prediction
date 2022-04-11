# %% 
# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import pickle

import difflib

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

import pymysql
from sqlalchemy import create_engine, inspect
from login import football_result_prediction_db_details

# %% IMPORT DATA

# IMPORT LEAGUE INFO
df_raw = pd.read_csv('League_Info.csv', index_col=0)
df_raw.drop(columns='Unnamed: 0', inplace=True)

# IMPORT RESULTS (LATEST LEAGUE INFO)
df_results = pd.read_csv('Results.csv', index_col=0)

# IMPORT FIXTURES (LATEST LEAGUE INFO)
df_fixtures = pd.read_csv('Fixtures.csv', index_col=0)

# IMPORT MATCH INFO
df_match = pd.read_csv('Match_Info.csv', index_col=0)
df_match.reset_index(inplace=True)

# IMPORT TEAM INFO
df_team = pd.read_csv('Team_Info.csv', index_col=0)
df_team.reset_index(inplace=True)
df_team.rename(columns={'Team': 'Home_Team'}, inplace=True)

# IMPORT MATCH ELO INFO
elo_dict = pickle.load(open('elo_dict.pkl', 'rb'))
df_elo = pd.DataFrame.from_dict(elo_dict, orient='index')
df_elo.reset_index(inplace=True)
df_elo.rename(columns={'index': 'Link', 'Elo_home': 'Home_ELO', 'Elo_away': 'Away_ELO'}, inplace=True)

# %% DATA CLEANING
# STANDARDISE LINK COLUMN IN EACH DATASET 
def link_clean(x):
    i=x.rfind('/')
    return x[:i+5]

df_raw['Link'] = df_raw['Link'].apply(lambda x: link_clean(x))
df_match['Link'] = df_match['Link'].apply(lambda x: 'https://www.besoccer.com' + x)
df_elo['Link'] = df_elo['Link'].apply(lambda x: link_clean(x))

# STANDARDISE TEAM NAME COLUMNS IN EACH DATASET
ht = set(df_raw['Home_Team'].tolist())
at = set(df_raw['Away_Team'].tolist())
tt = set(df_team['Home_Team'].tolist())

all_teams = set.union(ht, at, tt)

teams_dict = {}
for team in all_teams:
    try:
        new_name = difflib.get_close_matches(team, tt)[0]
    except:
        new_name = team
    teams_dict[team] = new_name

df_raw['Home_Team'] = df_raw['Home_Team'].apply(lambda x: teams_dict[x])
df_raw['Away_Team'] = df_raw['Away_Team'].apply(lambda x: teams_dict[x])
df_team['Home_Team'] = df_team['Home_Team'].apply(lambda x: teams_dict[x])


# %% 
# COMBINE DATASETS
df = pd.merge(df_raw, df_match, on='Link', how='outer')
df = pd.merge(df, df_team, on='Home_Team', how='outer')
df = pd.merge(df, df_elo, on='Link', how='outer')

# DROP DUPLICATE ROWS
df.drop_duplicates(inplace=True)

# DROP NULL VALUES FROM TEAM NAME AND RESULTS COLUMNS
df.dropna(subset=['Home_Team', 'Away_Team', 'Result'], inplace=True)

# DROP INVALID RESULTS
df.drop(df[df['Result'].str.len() != 3].index, inplace=True)

# DROP UNUSED COLUMNS
df.drop(columns=['Link', 'Date_New', 'Referee', 'Home_Yellow', 'Home_Red', 'Away_Yellow', 'Away_Red', 'City', 'Country', 'Pitch'], inplace=True)

# STANDARDISE TEAM NAME COLUMNS IN DF

df = df.append(df_results)
df = df.append(df_fixtures)
df.drop(columns='Link', inplace=True)
df.reset_index(inplace=True, drop=True)

ht = set(df['Home_Team'].tolist())
at = set(df['Away_Team'].tolist())

all_teams = set.union(ht, at)

teams_dict = {}
for team in all_teams:
    try:
        new_name = difflib.get_close_matches(team, tt)[0]
    except:
        new_name = team
    teams_dict[team] = new_name

df['Home_Team'] = df['Home_Team'].apply(lambda x: teams_dict[x])
df['Away_Team'] = df['Away_Team'].apply(lambda x: teams_dict[x])

# %%
# FILL NAN VALUES IN RESULT COLUMN
df.loc[df['Result'].isna(), 'Result'] = '100-10'
df.drop(df.loc[df['Result'].str.contains('N')].index, inplace=True)

# %% 
# SPLIT RESULTS TO HOME TEAM AND AWAY TEAM SCORE FEATURES
df['Home_Team_Score'] = df['Result'].apply(lambda x: int(x.split('-')[0]))
df['Away_Team_Score'] = df['Result'].apply(lambda x: int(x.split('-')[1]))

# CREATE RESULT COLUMN (WIN/DRAW/LOSS)
def result(x):
    if x['Home_Team_Score'] > x['Away_Team_Score']:
        return 'Win'

    elif x['Home_Team_Score'] == x['Away_Team_Score']:
        return 'Draw'
    else:
        return 'Loss'

df['Home_Team_Result'] = df.apply(lambda x: result(x), axis=1)

def result(x):
    if x['Home_Team_Score'] < x['Away_Team_Score']:
        return 'Win'

    elif x['Home_Team_Score'] == x['Away_Team_Score']:
        return 'Draw'
    else:
        return 'Loss'

df['Away_Team_Result'] = df.apply(lambda x: result(x), axis=1)

df = df.join(pd.get_dummies(df['Home_Team_Result'], prefix='Home_Team'))
df = df.join(pd.get_dummies(df['Away_Team_Result'], prefix='Away_Team'))

# %%
# MODIFY CAPACITY FEATURE TO NUMERICAL VALUES
def num_filter(x):
    numeric_filter = filter(str.isdigit, str(x))
    numeric_string = "".join(numeric_filter)
    try:
        return int(numeric_string)
    except ValueError:
        return np.nan

df['Capacity'] = df['Capacity'].apply(lambda x: num_filter(x))

capacity_dict = dict()

for i, team in enumerate(df.loc[~df['Capacity'].isna(), 'Home_Team']):
    capacity_dict[team] = df.loc[i, 'Capacity']

for i in df.loc[df['Capacity'].isna()].index:
    try:
        df.loc[i, 'Capacity'] = capacity_dict[df.loc[i, 'Home_Team']]
    except KeyError:
        continue

df.drop(columns=['League','Stadium'], inplace=True)

# DROP NULL VALUES AND RESET INDEX
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)

# %%
# SORT DATAFRAME BY SEASON AND ROUND
df.sort_values(by=['Season', 'Round'], inplace=True)
df.reset_index(inplace=True, drop=True)

# CREATE CUMULATIVE GOALS/WINS/DRAWS/LOSSES FEATURES
df_rt = df.groupby(by=['Home_Team', 'Season'])['Home_Team_Score', 'Home_Team_Win', 'Home_Team_Draw', 'Home_Team_Loss'].cumsum()
df_rt.rename(columns={
    'Home_Team_Score': 'Home_Team_Score_Total',
    'Home_Team_Win': 'Home_Team_Win_Total',
    'Home_Team_Draw': 'Home_Team_Draw_Total',
    'Home_Team_Loss': 'Home_Team_Loss_Total'
    },
    inplace=True)

df = df.join(df_rt)

df_rt = df.groupby(by=['Away_Team', 'Season'])['Away_Team_Score', 'Away_Team_Win', 'Away_Team_Draw', 'Away_Team_Loss'].cumsum()
df_rt.rename(columns={
    'Away_Team_Score': 'Away_Team_Score_Total',
    'Away_Team_Win': 'Away_Team_Win_Total',
    'Away_Team_Draw': 'Away_Team_Draw_Total',
    'Away_Team_Loss': 'Away_Team_Loss_Total'
    },
    inplace=True)

df = df.join(df_rt)

# MODIFY CUMMULATIVE FEATURES TO EXCLUDE CURRENT FIXTURE
df['Home_Team_Score_Total'] = df['Home_Team_Score_Total'] - df['Home_Team_Score']
df['Home_Team_Win_Total'] = df['Home_Team_Win_Total'] - df['Home_Team_Win']
df['Home_Team_Draw_Total'] = df['Home_Team_Draw_Total'] - df['Home_Team_Draw']
df['Home_Team_Loss_Total'] = df['Home_Team_Loss_Total'] - df['Home_Team_Loss']

df['Away_Team_Score_Total'] = df['Away_Team_Score_Total'] - df['Away_Team_Score']
df['Away_Team_Win_Total'] = df['Away_Team_Win_Total'] - df['Away_Team_Win']
df['Away_Team_Draw_Total'] = df['Away_Team_Draw_Total'] - df['Away_Team_Draw']
df['Away_Team_Loss_Total'] = df['Away_Team_Loss_Total'] - df['Away_Team_Loss']

# CREATE WIN/LOSS RATIO FEATURES
df['Home_Win_Loss_Ratio'] = df['Home_Team_Win_Total']/df['Home_Team_Loss_Total']
df['Home_Win_Loss_Ratio'] = df['Home_Win_Loss_Ratio'].replace([np.inf, -np.inf], 100)
df['Home_Win_Loss_Ratio'] = df['Home_Win_Loss_Ratio'].replace(np.nan, 0)

df['Home_Win_Ratio'] = df['Home_Team_Win_Total']/(df['Home_Team_Draw_Total'] + df['Home_Team_Loss_Total'])
df['Home_Win_Ratio'] = df['Home_Win_Ratio'].replace([np.inf, -np.inf], 100)
df['Home_Win_Ratio'] = df['Home_Win_Ratio'].replace(np.nan, 0)

df['Home_Draw_Ratio'] = df['Home_Team_Draw_Total']/(df['Home_Team_Win_Total'] + df['Home_Team_Loss_Total'])
df['Home_Draw_Ratio'] = df['Home_Draw_Ratio'].replace([np.inf, -np.inf], 100)
df['Home_Draw_Ratio'] = df['Home_Draw_Ratio'].replace(np.nan, 0)

df['Home_Loss_Ratio'] = df['Home_Team_Loss_Total']/(df['Home_Team_Draw_Total'] + df['Home_Team_Win_Total'])
df['Home_Loss_Ratio'] = df['Home_Loss_Ratio'].replace([np.inf, -np.inf], 100)
df['Home_Loss_Ratio'] = df['Home_Loss_Ratio'].replace(np.nan, 0)


df['Away_Win_Loss_Ratio'] = df['Away_Team_Win_Total']/df['Away_Team_Loss_Total']
df['Away_Win_Loss_Ratio'] = df['Away_Win_Loss_Ratio'].replace([np.inf, -np.inf], 100)
df['Away_Win_Loss_Ratio'] = df['Away_Win_Loss_Ratio'].replace(np.nan, 0)

df['Away_Win_Ratio'] = df['Away_Team_Win_Total']/(df['Away_Team_Draw_Total'] + df['Away_Team_Loss_Total'])
df['Away_Win_Ratio'] = df['Away_Win_Ratio'].replace([np.inf, -np.inf], 100)
df['Away_Win_Ratio'] = df['Away_Win_Ratio'].replace(np.nan, 0)

df['Away_Draw_Ratio'] = df['Away_Team_Draw_Total']/(df['Away_Team_Win_Total'] + df['Away_Team_Loss_Total'])
df['Away_Draw_Ratio'] = df['Away_Draw_Ratio'].replace([np.inf, -np.inf], 100)
df['Away_Draw_Ratio'] = df['Away_Draw_Ratio'].replace(np.nan, 0)

df['Away_Loss_Ratio'] = df['Away_Team_Loss_Total']/(df['Away_Team_Draw_Total'] + df['Away_Team_Win_Total'])
df['Away_Loss_Ratio'] = df['Away_Loss_Ratio'].replace([np.inf, -np.inf], 100)
df['Away_Loss_Ratio'] = df['Away_Loss_Ratio'].replace(np.nan, 0)

# %%
# CREATE SCORE/WIN/DRAW/LOSS STREAK FEATURES

home_teams = list(set(df['Home_Team']))
away_teams = list(set(df['Away_Team']))
seasons = list(set(df['Season']))

home_col_list = ['Home_Team_Score', 'Home_Team_Win', 'Home_Team_Draw', 'Home_Team_Loss']
home_col_list_2 = [i + '_' + 'Streak' for i in home_col_list]
away_col_list = ['Away_Team_Score', 'Away_Team_Win', 'Away_Team_Draw', 'Away_Team_Loss']
away_col_list_2 = [i + '_' + 'Streak' for i in away_col_list]


for season in seasons:
    for home_team in home_teams:
        home_temp_df = df.loc[(df['Home_Team'] == home_team) & (df['Season'] == season), home_col_list]
        home_temp_df_2 = pd.DataFrame(((home_temp_df.values).cumsum(axis=0)) - np.maximum.accumulate(((home_temp_df.values).cumsum(axis=0))*((home_temp_df.values) == 0)), index=home_temp_df.index, columns=home_col_list_2)
        home_temp_df_2 = home_temp_df_2.shift(fill_value=0)
        df.loc[(df['Home_Team'] == home_team) & (df['Season'] == season), home_col_list_2] = home_temp_df_2
        
    for away_team in away_teams:
        away_temp_df = df.loc[(df['Away_Team'] == away_team) & (df['Season'] == season), away_col_list]
        away_temp_df_2 = pd.DataFrame(((away_temp_df.values).cumsum(axis=0)) - np.maximum.accumulate(((away_temp_df.values).cumsum(axis=0))*((away_temp_df.values) == 0)), index=away_temp_df.index, columns=away_col_list_2)
        away_temp_df_2 = away_temp_df_2.shift(fill_value=0)
        df.loc[(df['Away_Team'] == away_team) & (df['Season'] == season), away_col_list_2] = away_temp_df_2


# %% 
# ENCODE TARGET VARIABLE
enc = LabelEncoder()
df['Labels'] = enc.fit_transform(df['Home_Team_Result'])

# %% 
# DROP FEATURES DEEMED UNIMPORTANT
df.loc[df['Home_Team_Score'] == 100, 'Labels'] = np.nan

df.drop(columns=['Result'], inplace=True)
df.drop(columns=['Home_Team_Score', 'Away_Team_Score'], inplace=True)
df.drop(columns=['Home_Team_Result', 'Home_Team_Win', 'Home_Team_Draw', 'Home_Team_Loss'], inplace=True)
df.drop(columns=['Away_Team_Result', 'Away_Team_Win', 'Away_Team_Draw', 'Away_Team_Loss'], inplace=True)


# %% 
# UPLOAD DATA TO DATABASE
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

df_train = df.loc[~df['Labels'].isna()]
df_test = df.loc[df['Labels'].isna()]

df.to_csv('cleaned_dataset.csv')
df.to_sql(name='Clean_Dataset', con=engine, if_exists='replace', index=False)

df_train.to_csv('cleaned_train_dataset.csv')
df_train.to_sql(name='Clean_Train_Dataset', con=engine, if_exists='replace', index=False)

df_test.to_csv('cleaned_test_dataset.csv')
df_test.to_sql(name='Clean_Test_Dataset', con=engine, if_exists='replace', index=False)

features = df.drop(columns='Labels').columns
features_df = pd.DataFrame(features)
features_df.to_csv('features.csv')
features_df.to_sql(name='Features_Table', con=engine, if_exists='replace', index=False)

classes = enc.classes_
classes_df = pd.DataFrame(classes)
classes_df.to_csv('classes.csv')
classes_df.to_sql(name='Classes_Table', con=engine, if_exists='replace', index=False)


# %%
