import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print("DATA EXPLORATION FOR GIULIETTA'S HOUSE\n")

dpi = 0
while dpi > 500 or dpi < 100:
    dpi = int(input(
        'DPI for saving the figures. Enter a number between 100 and 500\n'))
    if dpi > 500 or dpi < 100:
        print(f"You've entered \"{
              dpi}\", please enter a value between 100 and 500")

redir_stdout = input(
    'Save the output of the script in a txt file? [{y}es, {n}o]')
if redir_stdout == 'y' or redir_stdout == 'yes':
    sys.stdout = open('data_casa_di_giulietta_output.txt', 'w')
elif redir_stdout == 'n' or redir_stdout == 'no':
    pass

df = pd.read_csv('../data_casa_di_giulietta.csv',
                 index_col='date',
                 parse_dates=['date'])
df = df.drop('Unnamed: 0', axis=1)

print("Data's first 5 rows\n",
      df.head(), '\n--------------------------------------------')

print("Data info\n", df.describe(), '\n')
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
print(info_str, '\n--------------------------------------------')

print(f'Data has {df.shape[0]} rows and {
      df.shape[1]} columns', '\n--------------------------------------------')

fig, ax = plt.subplots(2, 2, figsize=(20, 20))
plt.suptitle("Giulietta's house variables", fontsize=40)
sns.lineplot(y=df['count'], x=df.index, linewidth=0.4,
             color='#1d4863', ax=ax[0][0])
sns.lineplot(y=df['temp'], x=df.index, linewidth=0.3,
             color='#e8ad3a', ax=ax[0][1])
sns.lineplot(y=df['rain'], x=df.index, linewidth=0.3,
             color='#ca3ae8', ax=ax[1][0])
sns.lineplot(y=df['popularity'], x=df.index,
             linewidth=0.3, color='#5cad52', ax=ax[1][1])
plt.savefig('img/data_analysis/giulietta_timeseries.png', dpi=dpi)

week_mask = (df.index > '2018-05-21') & (df.index <= '2018-05-28')
week_df = df[week_mask]

fig, ax = plt.subplots(2, 2, figsize=(20, 20))
plt.suptitle("Giulietta's house variables, week of 2018-05-28", fontsize=40)
sns.lineplot(y=week_df['count'], x=week_df.index, linewidth=1.5,
             color='#1d4863', ax=ax[0][0])
sns.lineplot(y=week_df['temp'], x=week_df.index, linewidth=3,
             color='#e8ad3a', ax=ax[0][1])
ax[0][1].set_ylim(0, 4.3)
sns.lineplot(y=week_df['rain'], x=week_df.index, linewidth=3,
             color='#ca3ae8', ax=ax[1][0])
sns.lineplot(y=week_df['popularity'], x=week_df.index,
             linewidth=3, color='#5cad52', ax=ax[1][1])
ax[1][1].set_ylim(0, 19.3)
plt.savefig('img/data_analysis/giulietta_timeseries_oneweek.png', dpi=dpi)

max_count_mask = df['count'] == df['count'].max()
max_count = df[max_count_mask]
print("The maximum number of tourists entering Giulietta's house was",
      int(max_count.iloc[0]['count']), 'on', max_count.index[0])

if redir_stdout == 'y' or redir_stdout == 'yes':
    sys.stdout.close()
