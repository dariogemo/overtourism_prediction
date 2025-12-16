# Create a csv file where the temperature data from 2014 to 2020 gets encoded from 0 to 4

import numpy as np
import pandas as pd

df: pd.DataFrame = pd.read_csv(
    '../../main_dataset/context_weather_all_processed.csv')

bins = [-np.inf, 5, 10, 15, 20, np.inf]
labels = [0, 1, 2, 3, 4]

# Create a new column with the binned values
df['temp'] = pd.cut(df['temp'], bins=bins, labels=labels, right=False)
df.set_index('data', inplace=True)

df.to_csv('../../main_dataset/context_weather_all_temp_discr.csv')
