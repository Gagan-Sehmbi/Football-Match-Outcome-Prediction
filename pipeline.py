# %% IMPORT LIBRARIES
import os
import glob

import numpy as np
import pandas as pd
import pickle

import difflib

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# %% IMPORT DATA
# CREATE LEAGUE INFO CSV

leagues = [x[0] for x in os.walk('Football-Dataset')]
for league in leagues[1:]:
    league_files = glob.glob(league + '/*.csv')
    league_name = league.split('/')[1]
    l = []
    for file in league_files:
        df = pd.read_csv(file, index_col=None, header=0)
        l.append(df)

    frame = pd.concat(l, axis=0, ignore_index=True)
    frame.to_csv(f'Football-Dataset/{league_name}.csv')

path = 'Football-Dataset'
all_files = glob.glob(path + '/*.csv')

l = []
for file in all_files:
    df = pd.read_csv(file, index_col=0, header=0)
    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv('League_Info.csv')

# IMPORT LEAGUE INFO
df_raw = pd.read_csv('League_Info.csv', index_col=0)

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

# SPLIT DATE FEATURE
def date_split(x, value):
    x = x.split(', ')
    if len(x) == 3:
        if value=='Time':
            return x[-1].strip()
        elif value=='Day':
            return x[0].strip()

        elif value=='Date':
            x = x[1]
            x = x.split(' ')
            return int(x[0].strip())

        elif value=='Month':
            x = x[1]
            x = x.split(' ')
            return x[1].strip()

        elif value=='Year':
            x = x[1]
            x = x.split(' ')
            return int(x[2].strip())

        else:
            return 'NA'

    else:
        if value=='Time':
            return 'NA'

        elif value=='Date':
            x = x[1]
            x = x.split(' ')
            return int(x[0].strip())

        elif value=='Month':
            x = x[1]
            x = x.split(' ')
            return x[1].strip()

        elif value=='Year':
            x = x[1]
            x = x.split(' ')
            return int(x[2].strip())

        else:
            return 'NA'

df['Time'] = df.loc[~df['Date_New'].isna(),'Date_New'].apply(lambda x: date_split(x, 'Time'))
df['Day'] = df.loc[~df['Date_New'].isna(),'Date_New'].apply(lambda x: date_split(x, 'Day'))
df['Date'] = df.loc[~df['Date_New'].isna(),'Date_New'].apply(lambda x: date_split(x, 'Date'))
df['Month'] = df.loc[~df['Date_New'].isna(),'Date_New'].apply(lambda x: date_split(x, 'Month'))
df['Year'] = df.loc[~df['Date_New'].isna(),'Date_New'].apply(lambda x: date_split(x, 'Year'))

df.drop(columns='Date_New', inplace=True)
df = df.loc[df['Day'] != 'NA']

# MODIFY PRECISE TIME FEATURE TO HOURLY
df['Time'] = df.loc[~df['Time'].isna(),'Time'].apply(lambda x: int(x.split(':')[0]))

# MODIFY TIME FEATURE TO BINS
df['Time'] = pd.cut(df['Time'], bins=[-1, 11, 16, 20, 24], labels=['Morning', 'Afternoon', 'Evening', 'Night'])

# IMPUTE YELLOW/ RED CARD VALUES WITH MEAN VALUES
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df['Home_Yellow'] = mean_imputer.fit_transform(df['Home_Yellow'].values.reshape(-1, 1))
df['Home_Red'] = mean_imputer.fit_transform(df['Home_Red'].values.reshape(-1, 1))
df['Away_Yellow'] = mean_imputer.fit_transform(df['Away_Yellow'].values.reshape(-1, 1))
df['Away_Red'] = mean_imputer.fit_transform(df['Away_Red'].values.reshape(-1, 1))

# CREATE REGION FEATURE FROM LEAGUE FEATURE AND DROP OTHER LOCATION FEATURES
df['Region'] = df['League']
values_to_update ={
    'Region': {
        'segunda_division': 'Spain',
        'primera_division': 'Spain',
        'serie_b': 'Italy',
        'serie_a': 'Italy',
        'premier_league': 'England',
        'championship': 'England',
        'ligue_1': 'France',
        'ligue_2': 'France',
        '2_liga': 'Germany',
        'bundesliga': 'Germany',
        'eredivisie': 'Netherlands',
        'eerste_divisie': 'Netherlands',
        'primeira_liga': 'Portugal',
        'segunda_liga': 'Portugal'        
        }
}

df = df.replace(values_to_update)
df.drop(columns='City', inplace=True)
df.drop(columns='Country', inplace=True)

# MODIFY CAPACITY FEATURE TO NUMERICAL VALUES
def num_filter(x):
    numeric_filter = filter(str.isdigit, str(x))
    numeric_string = "".join(numeric_filter)
    return numeric_string

df['Capacity'] = df['Capacity'].apply(lambda x: num_filter(x))
df['Capacity'].value_counts().sort_index()

# DROP NULL VALUES FROM ELO FEATURES
df.dropna(subset=['Home_ELO', 'Away_ELO'], inplace=True)

# DROP NULL VALUES
df.dropna(inplace=True)

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

# CREATE SCORE/WIN/DRAW/LOSS STREAK FEATURES

home_teams = list(set(df['Home_Team']))
away_teams = list(set(df['Away_Team']))
seasons = list(set(df['Season']))

home_col_list = ['Home_Team_Score', 'Home_Team_Win', 'Home_Team_Draw', 'Home_Team_Loss']
home_col_list_2 = [i + '_' + 'Streak' for i in home_col_list]
away_col_list = ['Away_Team_Score', 'Away_Team_Win', 'Away_Team_Draw', 'Away_Team_Loss']
away_col_list_2 = [i + '_' + 'Streak' for i in away_col_list]

for home_team, away_team in zip(home_teams, away_teams):
    for season in seasons:
        home_temp_df = df.loc[(df['Home_Team'] == home_team) & (df['Season'] == season), home_col_list]
        away_temp_df = df.loc[(df['Away_Team'] == away_team) & (df['Season'] == season), away_col_list]

        home_temp_df_2 = pd.DataFrame(((home_temp_df.values).cumsum(axis=0)) - np.maximum.accumulate(((home_temp_df.values).cumsum(axis=0))*((home_temp_df.values) == 0)), index=home_temp_df.index, columns=home_col_list_2)
        away_temp_df_2 = pd.DataFrame(((away_temp_df.values).cumsum(axis=0)) - np.maximum.accumulate(((away_temp_df.values).cumsum(axis=0))*((away_temp_df.values) == 0)), index=away_temp_df.index, columns=away_col_list_2)

        home_temp_df_2 = home_temp_df_2.shift(fill_value=0)
        away_temp_df_2 = away_temp_df_2.shift(fill_value=0)

        df.loc[(df['Home_Team'] == home_team) & (df['Season'] == season), home_col_list_2] = home_temp_df_2
        df.loc[(df['Away_Team'] == away_team) & (df['Season'] == season), away_col_list_2] = away_temp_df_2

# DROP FEATURES DEEMED UNIMPORTANT
df.drop(
    columns=[
        'Home_Team', 'Away_Team', 'Result', 'Link',
        'League', 'Referee',
        'Stadium', 'Pitch', 'Capacity',
        'Region'], 
        inplace=True)

df.drop(columns=['Time', 'Day', 'Date', 'Month', 'Year'], inplace=True)

df.drop(columns=['Away_Team_Result', 'Away_Team_Win', 'Away_Team_Draw', 'Away_Team_Loss'], inplace=True)

df.drop(columns=['Home_Team_Score', 'Away_Team_Score'], inplace=True)

df.drop(columns=['Home_Team_Win', 'Home_Team_Draw', 'Home_Team_Loss'], inplace=True)

# %% ENCODING, SCALING AND SPLITTING DATASET
# ENCODE TARGET VARIABLE
enc = LabelEncoder()

df['Labels'] = enc.fit_transform(df['Home_Team_Result'])
df.drop(columns=['Home_Team_Result'], inplace=True)

# SPLIT INPUT FEATURES FROM TARGET FEATURE
X = df.drop(columns=['Labels'])
y = df['Labels']

# SPLIT DATASET INTO TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# SCALE INPUT VARIABLES
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %% CREATE TRAIN, TEST, FEATURES AND CLASSES DATAFRAMES

features = X.columns
features_df = pd.DataFrame(features)
features_df.to_csv('features.csv')

classes = enc.classes_
classes_df = pd.DataFrame(classes)
classes_df.to_csv('classes.csv')

X_train_df = pd.DataFrame(X_train, columns=features)
X_train_df.to_csv('X_train.csv')

X_test_df = pd.DataFrame(X_test, columns=features)
X_test_df.to_csv('X_test.csv')

y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

# %% UPLOAD DATASETS TO DATABASE

