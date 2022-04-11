# %%
# IMPORT LIBRARIES

from operator import index
import os
import glob
import pandas as pd
import pickle

# %% 
# FOOTBALL-DATASET/*

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

# %%
# FOOTBALL-DATASET

path = 'Football-Dataset'
all_files = glob.glob(path + '/*.csv')

l = []
for file in all_files:
    df = pd.read_csv(file, index_col=0, header=0)
    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv('League_Info.csv')

# %% BACKUP
# DATA/RESULTS/*

leagues = [x[0] for x in os.walk('Data/Results')]
for league in leagues[1:]:
    league_files = glob.glob(league + '/*.csv')
    league_name = league.split('/')[-1]
    l = []
    for file in league_files:
        df = pd.read_csv(file, index_col=None, header=0)
        l.append(df)

    frame = pd.concat(l, axis=0, ignore_index=True)
    frame.to_csv(f'Data/Results/{league_name}.csv')

# %%
# DATA/RESULTS/*

leagues = [x[0] for x in os.walk('Data/Results')]

l = []

for league in leagues[1:]:
    df_league = pd.read_csv(glob.glob(league + '/*.csv')[0], index_col=0, header=0)

    elo_dict = pickle.load(open(glob.glob(league + '/*.pkl')[0], 'rb'))
    df_elo = pd.DataFrame.from_dict(elo_dict, orient='index')
    df_elo.reset_index(inplace=True)
    df_elo.rename(columns={'index': 'Link', 'Elo_home': 'Home_ELO', 'Elo_away': 'Away_ELO'}, inplace=True)
    
    df = pd.concat([df_league.set_index('Link'),df_elo.set_index('Link')], axis=1, join='inner').reset_index()

    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv(f'Results.csv')

# %%
# DATA/TO_PREDICT/*

leagues = [x[0] for x in os.walk('Data/To_Predict')]

l = []

for league in leagues[1:]:
    df_league = pd.read_csv(glob.glob(league + '/*.csv')[0], index_col=0, header=0)

    elo_dict = pickle.load(open(glob.glob(league + '/*.pkl')[0], 'rb'))
    df_elo = pd.DataFrame.from_dict(elo_dict, orient='index')
    df_elo.reset_index(inplace=True)
    df_elo.rename(columns={'index': 'Link', 'Elo_home': 'Home_ELO', 'Elo_away': 'Away_ELO'}, inplace=True)
    
    df = pd.concat([df_league.set_index('Link'),df_elo.set_index('Link')], axis=1, join='inner').reset_index()

    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv(f'Fixtures.csv')

# %%
# DATA/TO_PREDICT

path = 'Data/To_Predict'
all_files = glob.glob(path + '/*.csv')

l = []
for file in all_files:
    df = pd.read_csv(file, index_col=0, header=0)
    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv('Fixtures.csv')

# %%
