import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from utils.timefeatures import time_features


df_train: pd.DataFrame = pd.read_csv(
    "./../data_casa_di_giulietta_train.csv", index_col=1
).drop("Unnamed: 0", axis=1)
df_test: pd.DataFrame = pd.read_csv(
    "./../data_casa_di_giulietta_test.csv", index_col=1
).drop("Unnamed: 0", axis=1)
df_train.index = pd.to_datetime(df_train.index)
df_test.index = pd.to_datetime(df_test.index)

X_train = df_train.drop("count", axis=1)
y_train = df_train.pop("count")

X_test = df_test.drop("count", axis=1)
y_test = df_test.pop("count")

regr = RandomForestRegressor(n_estimators=100, max_depth=10)
regr.fit(X_train, y_train)

y_pred = np.around(regr.predict(X_test))
y_pred = pd.DataFrame(y_pred, columns=["Predicted"], index=y_test.index)

print(mean_absolute_error(y_test, y_pred))

plt.figure()
plt.plot(y_test, label="True")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.show()
plt.savefig("./../scripts/img/random_forest_regressor.png")

