# %%
import pandas as pd
# %%
df = pd.read_csv('all_leagues_data.csv', index_col=0)
df

# %%
df.drop(columns=['Link'], inplace=True)
df.drop(df[df['Result'].str.len() != 3].index, inplace=True)
df

# %%
df['Home_Team_Score'] = df['Result'].apply(lambda x: x.split('-')[0])
df['Away_Team_Score'] = df['Result'].apply(lambda x: x.split('-')[1])
df

# %%
df.info()
# %%
