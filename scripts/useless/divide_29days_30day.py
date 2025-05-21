import pandas as pd

# load datasets
df = pd.read_csv('/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/main_dataset/count_data/small_castelvecchio.csv',
                 parse_dates=True, index_col=0)
meteo_df = pd.read_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/main_dataset/weather_all_processed.csv', parse_dates=True, index_col=0)

# filter the meteo dataframe
meteo_df = meteo_df.loc['2020-01']

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

# divide into 29 days and 1 day
last_day = len(final_df)-96
df_29days = final_df.iloc[:last_day]
df_day30 = final_df.iloc[last_day:]

# save the two datasets
df_29days.to_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/Code/Informer2020/data/main/df_29days.csv', index=True)
df_day30.to_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/Code/Informer2020/data/main/day30.csv', index=True)
