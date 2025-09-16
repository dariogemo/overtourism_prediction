import io
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("DATA EXPLORATION FOR GIULIETTA'S HOUSE\n")

run_figs: str = str(input("Do you want to display the figures? [{y}es, {n}o]\n"))

if run_figs in ["y", "yes"]:
    DPI = 0
    while DPI > 500 or DPI < 100:
        try:
            DPI = int(
                input(
                    "DPI for saving the figures. Enter a number between 100 and 500\n"
                )
            )
        except ValueError as e:
            print("Please insert a valid number")
        if DPI > 500 or DPI < 100:
            print("Please enter a value between 100 and 500")
else:
    DPI = None

redir_stdout: str = str(
    input(
        "Save the output of the script in a txt file\n"
        "(txt_files/data_casa_di_giulietta_output.txt)? [{y}es, {n}o]\n"
    )
)
if redir_stdout in ["y", "yes"]:
    sys.stdout = open(
        "txt_files/data_casa_di_giulietta_output.txt", "w", encoding="utf-8"
    )
else:
    pass

df: pd.DataFrame = pd.read_csv(
    "../../main_dataset/count_data/data_anfiteatro_arena.csv",
    index_col="date",
    parse_dates=["date"],
)
df: pd.DataFrame = df.drop("Unnamed: 0", axis=1)
df.fillna(0, inplace=True)

print(
    "Data's first 5 rows\n", df.head(), "\n--------------------------------------------"
)

print("Data info\n", df.describe(), "\n")
buffer: io.StringIO = io.StringIO()
df.info(buf=buffer)
info_str: str = buffer.getvalue()
print(info_str, "\n--------------------------------------------")

print(
    f"Data has {df.shape[0]} rows and {df.shape[1]} columns",
    "\n--------------------------------------------",
)

if run_figs in ["y", "yes"]:
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    plt.suptitle("Arena di Verona variables", fontsize=40)
    sns.lineplot(y=df["count"], x=df.index, linewidth=0.4, color="#1d4863", ax=ax[0][0])
    sns.lineplot(y=df["temp"], x=df.index, linewidth=0.3, color="#e8ad3a", ax=ax[0][1])
    sns.lineplot(y=df["rain"], x=df.index, linewidth=0.3, color="#ca3ae8", ax=ax[0][2])
    sns.lineplot(
        y=df["festivity"], x=df.index, linewidth=0.3, color="#2596be", ax=ax[1][0]
    )
    sns.lineplot(
        y=df["rolling_mean_30d"],
        x=df.index,
        linewidth=0.3,
        color="#78BE25",
        ax=ax[1][1],
    )
    sns.lineplot(
        y=df["above_avg"], x=df.index, linewidth=0.01, color="#BE8825", ax=ax[1][2]
    )
    for ax1 in ax.flatten():
        ax1.set_xlabel(None)
    plt.savefig("img/data_analysis/arena_timeseries.png", dpi=DPI)
    plt.show()
else:
    pass

week_mask = (df.index > "2018-05-21") & (df.index <= "2018-05-28")
week_df: pd.DataFrame = df.loc[week_mask]

if run_figs in ["y", "yes"]:
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    plt.suptitle("Arena di Verona variables, week of 2018-05-28", fontsize=40)
    sns.lineplot(
        y=week_df["count"], x=week_df.index, linewidth=1, color="#1d4863", ax=ax[0][0]
    )
    sns.lineplot(
        y=week_df["temp"], x=week_df.index, linewidth=1.5, color="#e8ad3a", ax=ax[0][1]
    )
    sns.lineplot(
        y=week_df["rain"], x=week_df.index, linewidth=1.5, color="#ca3ae8", ax=ax[0][2]
    )
    sns.lineplot(
        y=week_df["festivity"],
        x=week_df.index,
        linewidth=1.5,
        color="#2596be",
        ax=ax[1][0],
    )
    sns.lineplot(
        y=week_df["rolling_mean_30d"],
        x=week_df.index,
        linewidth=1.5,
        color="#78BE25",
        ax=ax[1][1],
    )
    sns.lineplot(
        y=week_df["above_avg"],
        x=week_df.index,
        linewidth=0.5,
        color="#BE8825",
        ax=ax[1][2],
    )
    for ax1 in ax.flatten():
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_xlabel(None)
    plt.savefig("img/data_analysis/arena_timeseries_oneweek.png", dpi=DPI)
    plt.show()
else:
    pass

max_count_mask: pd.Series = df["count"] == df["count"].max()
max_count: pd.DataFrame = df.loc[max_count_mask]
print(
    "The maximum number of tourists entering Giulietta's house was",
    int(max_count.iloc[0]["count"]),
    "on",
    max_count.index[0],
)

print(max_count.index[0])

if redir_stdout in ["y", "yes"]:
    sys.stdout.close()
