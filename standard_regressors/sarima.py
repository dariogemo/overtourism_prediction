import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from utils.stationarity_tests import adfuller_test, kpss_test

df_train: pd.DataFrame = pd.read_csv(
    "./../data_casa_di_giulietta_train.csv", index_col=0, parse_dates=True
)
df_train.set_index("date", inplace=True)
df_test = pd.read_csv("./../data_casa_di_giulietta_test.csv")

"""for col in df_train.columns:
    print("Processing", col)
    adfuller_test(df_train[col])
    kpss_test(df_train[col])
    print("------------------------------")
"""

col_to_dif = ["count", "temp", "rain", "rolling_mean_30d", "above_avg"]
for col in col_to_dif:
    print("Processing", col)
    df_train[col] = df_train[col].diff().bfill()

arima_model = auto_arima(
    df_train["count"],
    exogenous=df_train.drop("count", axis=1),
    start_p=0,
    d=0,
    start_q=0,
    max_p=6,
    max_q=6,
    m=1,
    seasonal=False,
    error_action="warn",
    with_intercept=True,
    trace=True,
    supress_warnings=True,
    stepwise=False,
    random_state=20,
    information_criterion="aic",
)
