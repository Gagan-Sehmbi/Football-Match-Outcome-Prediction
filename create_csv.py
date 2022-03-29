# %%
# IMPORT LIBRARIES

import os
import glob
import pandas as pd

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

# %%
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
# DATA/RESULTS

path = 'Data/Results'
all_files = glob.glob(path + '/*.csv')

l = []
for file in all_files:
    df = pd.read_csv(file, index_col=0, header=0)
    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv('Results.csv')

# %%
# DATA/TO_PREDICT/*

leagues = [x[0] for x in os.walk('Data/To_Predict')]
for league in leagues[1:]:
    league_files = glob.glob(league + '/*.csv')
    league_name = league.split('/')[-1]
    l = []
    for file in league_files:
        df = pd.read_csv(file, index_col=None, header=0)
        l.append(df)

    frame = pd.concat(l, axis=0, ignore_index=True)
    frame.to_csv(f'Data/To_Predict/{league_name}.csv')

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
