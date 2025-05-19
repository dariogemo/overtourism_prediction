import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils.timefeatures import time_features


df = pd.read_csv('../Informer2020/data/main/df_29days.csv', index_col = 1).drop('Unnamed: 0', axis=1)
data_stamp = time_features(df.index, timeenc = 1, freq = 'T')
df_data_stamp = pd.DataFrame(data_stamp, index = df.index)
df = pd.merge(df, df_data_stamp, right_index=True, left_index=True)

train_df = df.loc[df.index < '2018-10-14 14:45:00']
X_train = train_df.drop('count', axis = 1)
y_train = train_df.pop('count')

test_df = df.loc[df.index > '2018-10-14 14:45:00']
test_df = test_df.loc[test_df.index < '2018-12-24 22:30:00']

X_test = test_df.drop('count', axis = 1)
y_test = test_df.pop('count')

regr = RandomForestRegressor(n_estimators = 100, max_depth = 10)
regr.fit(X_train, y_train)

y_pred = np.around(regr.predict(X_test))
y_pred = pd.DataFrame(y_pred, columns = ['Predicted'], index = y_test.index)

print(mean_squared_error(y_test, y_pred))

plt.figure()
plt.plot(y_test, label = 'True')
plt.plot(y_pred, label = 'Predicted')
plt.legend()
plt.savefig('img/random_forest_regressor.png')