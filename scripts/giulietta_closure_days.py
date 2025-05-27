import pandas as pd

df: pd.DataFrame = pd.read_csv(
    "./../../main_dataset/count_data/data_casa_di_giulietta.csv",
    index_col=0,
)[["date", "count"]]
df.set_index("date", inplace=True)
df.index = pd.to_datetime(df.index)
print(type(df.index))

daily_sum = df.resample("D").sum()

# Step 2: Filter days where the sum is zero
zero_sum_days = daily_sum[daily_sum["count"] == 0]

# Step 3 (optional): Get just the dates
zero_sum_dates = zero_sum_days.index.date

print(zero_sum_days)
print(zero_sum_dates)
