import pandas as pd
from datetime import datetime
import numpy as np


fest_dic = {
    2014: [
        [1, 1],
        [1, 6],
        [4, 20],
        [4, 21],
        [4, 25],
        [5, 1],
        [6, 2],
        [8, 15],
        [11, 1],
        [12, 8],
        [12, 25],
        [12, 26],
        [12, 31],
    ],
    2015: [
        [1, 1],
        [1, 6],
        [4, 5],
        [4, 6],
        [4, 25],
        [5, 1],
        [6, 2],
        [8, 15],
        [11, 1],
        [12, 8],
        [12, 25],
        [12, 26],
        [12, 31],
    ],
    2016: [
        [1, 1],
        [1, 6],
        [3, 27],
        [3, 28],
        [4, 25],
        [5, 1],
        [6, 2],
        [8, 15],
        [11, 1],
        [12, 8],
        [12, 25],
        [12, 26],
        [12, 31],
    ],
    2017: [
        [1, 1],
        [1, 6],
        [4, 16],
        [4, 17],
        [4, 25],
        [5, 1],
        [6, 2],
        [8, 15],
        [11, 1],
        [12, 8],
        [12, 25],
        [12, 26],
        [12, 31],
    ],
    2018: [
        [1, 1],
        [1, 6],
        [4, 1],
        [4, 2],
        [4, 25],
        [5, 1],
        [6, 2],
        [8, 15],
        [11, 1],
        [12, 8],
        [12, 25],
        [12, 26],
        [12, 31],
    ],
    2019: [
        [1, 1],
        [1, 6],
        [4, 21],
        [4, 22],
        [4, 25],
        [5, 1],
        [6, 2],
        [8, 15],
        [11, 1],
        [12, 8],
        [12, 25],
        [12, 26],
        [12, 31],
    ],
}

fest_series = pd.Series([])
idx = 0
for year in fest_dic.keys():
    for i, date in enumerate(fest_dic[year]):
        fest_series[idx] = datetime(year, date[0], date[1])
        idx += 1

fest_df = pd.DataFrame(np.ones(len(fest_series)), index=fest_series, columns=["values"])

fest_df: pd.DataFrame = fest_df.resample("15min").ffill(limit=95)
dates = pd.date_range(start="2019-12-31", end="2020-01-01", freq="15min")
fest_df: pd.DataFrame = pd.concat(
    [fest_df, pd.DataFrame(np.ones(len(dates)), index=dates, columns=["values"])],
    axis=0,
)[:-1]

fest_df.fillna(0, inplace=True)

for date in [
    datetime(2014, 12, 25),
    datetime(2015, 12, 12),
    datetime(2015, 12, 13),
    datetime(2015, 12, 14),
    datetime(2015, 12, 25),
    datetime(2016, 12, 25),
    datetime(2017, 4, 18),
    datetime(2017, 10, 15),
    datetime(2017, 10, 16),
    datetime(2017, 12, 25),
    datetime(2018, 1, 29),
    datetime(2018, 2, 6),
    datetime(2018, 2, 7),
    datetime(2018, 10, 8),
    datetime(2018, 11, 12),
    datetime(2018, 11, 19),
    datetime(2018, 12, 25),
    datetime(2019, 5, 27),
    datetime(2019, 12, 25),
]:
    fest_df.loc[date, "values"] = 2

print(fest_df.value_counts())

fest_df.to_csv("./../../main_dataset/context_festivities.csv")
