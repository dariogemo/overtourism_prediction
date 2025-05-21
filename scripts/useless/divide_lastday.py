import pandas as pd

# load datasets
df = pd.read_csv('/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/main_dataset/count_data/data_anfiteatro_arena.csv',
                 parse_dates=True, index_col=1)
df.drop('Unnamed: 0', axis=1, inplace=True)
meteo_df = pd.read_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/main_dataset/context_weather_all_processed.csv', parse_dates=True, index_col=0)

# filter the meteo dataframe
meteo_df1 = meteo_df.loc['2018']
meteo_df2 = meteo_df.loc['2019']
meteo_df = pd.concat([meteo_df1, meteo_df2])

# up-sample meteo dataframe
new_index = pd.date_range(
    start=meteo_df.index.min(),
    end=meteo_df.index.max() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
    freq='15min'
)
df_meteo = pd.DataFrame(index=new_index)
df_meteo = df_meteo.merge(meteo_df, left_index=True,
                          right_index=True, how='left')
df_meteo.ffill(inplace=True)

# merge the two dataframes
final_df = pd.merge(df, df_meteo, left_index=True,
                    right_index=True, how='inner')
final_df.reset_index(inplace=True)
final_df.rename(columns={'index': 'date'}, inplace=True)

# change rain features
final_df.replace({'no_rain': 0, 'rain': 1}, inplace=True)
final_df = final_df.iloc[:34510]

# divide into 29 days and 1 day
last_day = len(final_df)-145
df_29days = final_df.iloc[:last_day]
df_day30 = final_df.iloc[last_day:]

print(df_29days)

# save the two datasets
df_29days.to_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/Code/Pyraformer/data/main/df_29days.csv', index=True)
df_day30.to_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/Code/Pyraformer/data/main/day30.csv', index=True)
