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

# Process date features directly
df_train = df_train.reset_index()
df_test = df_test.reset_index()

# Create simple date features manually
df_train["date"] = pd.to_datetime(df_train["date"])
df_test["date"] = pd.to_datetime(df_test["date"])

df_train["minute"] = df_train["date"].dt.minute / 59.0 - 0.5
df_train["hour"] = df_train["date"].dt.hour / 23.0 - 0.5
df_train["dayofweek"] = df_train["date"].dt.dayofweek / 6.0 - 0.5
df_train["dayofmonth"] = (df_train["date"].dt.day - 1) / 30.0 - 0.5
df_train["dayofyear"] = (df_train["date"].dt.dayofyear - 1) / 365.0 - 0.5

df_test["minute"] = df_test["date"].dt.minute / 59.0 - 0.5
df_test["hour"] = df_test["date"].dt.hour / 23.0 - 0.5
df_test["dayofweek"] = df_test["date"].dt.dayofweek / 6.0 - 0.5
df_test["dayofmonth"] = (df_test["date"].dt.day - 1) / 30.0 - 0.5
df_test["dayofyear"] = (df_test["date"].dt.dayofyear - 1) / 365.0 - 0.5

# Prepare features and target for model
X_train = df_train.drop(["count", "date"], axis=1)
y_train = df_train["count"]

X_test = df_test.drop(["count", "date"], axis=1)
y_test = df_test["count"]

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
# plt.savefig("./../scripts/img/random_forest_regressor.png")

importances = regr.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = list(X_test.columns)

# Plot
plt.figure(figsize=(6, 4))
plt.bar(range(X_test.shape[1]), importances[indices], align="center")
plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()
