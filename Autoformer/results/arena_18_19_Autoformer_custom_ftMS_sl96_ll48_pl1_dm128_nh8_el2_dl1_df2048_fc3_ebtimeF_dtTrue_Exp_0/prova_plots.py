import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import datetime

df = pd.read_csv('../../dataset/main/df_29days.csv', index_col = 1, parse_dates = ['date'])['count']
num_train = int(len(df)*0.7)
num_test = int(len(df)*0.2)
num_vali = len(df) - num_train - num_test
df = df[int(len(df)*0.7):]
mean = np.mean(df)
std = np.std(df)

true = np.load('true.npy')
pred = np.load('pred.npy')
dates = np.load('date.npy', allow_pickle = True)

very_dates = dates[:, -1, :]
very_true = true[0::1]
very_pred = pred[0::1]

very_true = np.concatenate(very_true, axis=0)
very_pred = np.concatenate(very_pred, axis=0)
minute_norm = very_dates[:, 0]
hour_norm = very_dates[:, 1]
weekday_norm = very_dates[:, 2]
day_norm = very_dates[:, 3]
day_year_norm = very_dates[:, 4]

day = np.round((day_norm + 0.5) * 30 + 1).astype(int)
weekday = np.round((weekday_norm + 0.5) * 6).astype(int)
hour = np.round((hour_norm + 0.5) * 23).astype(int)
minute = np.round((minute_norm + 0.5) * 59).astype(int)
day_year = np.round((day_year_norm + 0.5) * 365).astype(int) + 1
dates = pd.to_datetime('2018-01-01') + pd.to_timedelta(day_year - 1, unit='D')
months = dates.month

full_datetimes = []

for i in range(len(very_dates)):
	month_final = months[i]
	day_final = day[i]
	hour_final = hour[i]
	minute_final = minute[i]
	dt = datetime.datetime(2018, month_final, day_final, hour_final, minute_final)
	full_datetimes.append(dt)

full_datetimes = np.array(full_datetimes)

very_true = (very_true * (std + 1e-8)) + mean
very_pred = (very_pred * (std + 1e-8)) + mean
very_pred[very_pred < 0] = 0
very_pred = np.around(very_pred)

final_df = pd.DataFrame(very_pred, index = full_datetimes, columns = ['Predicted'])
final_df = pd.merge(final_df, df, left_index = True, right_index = True)

print('MAE for informer', mean_absolute_error(final_df['Predicted'], final_df['count']))

plt.figure()
plt.plot(final_df.iloc[:500]['count'], label = 'True')
plt.plot(final_df.iloc[:500]['Predicted'], label = 'Predicted')
plt.legend()
plt.savefig('../../img/pl_1_first500.png')

plt.figure()
plt.plot(final_df['count'], label='True')
plt.plot(final_df['Predicted'], label='Predicted')
plt.legend()
plt.savefig('../../img/pl_1_full.png')