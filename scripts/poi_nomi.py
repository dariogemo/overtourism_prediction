# create a csv associating each poi id with its name

import pandas as pd

df = pd.read_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/main_dataset/poi_all.csv')

poi_id = pd.Series(df['poi_id'].unique(), name='poi_id')
poi_name = pd.Series(df['poi_name'].unique(), name='poi_name')

poi_df = pd.merge(poi_name, poi_id, left_index=True, right_index=True)
poi_df.to_csv(
    '/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/main_dataset/poi_idx.csv', index=False)
