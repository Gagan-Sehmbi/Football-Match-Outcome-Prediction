# %%
import os
import glob
import pandas as pd
# %% 

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
path = 'Football-Dataset'
all_files = glob.glob(path + '/*.csv')

l = []
for file in all_files:
    df = pd.read_csv(file, index_col=0, header=0)
    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv('all_leagues_data.csv')
# %%
