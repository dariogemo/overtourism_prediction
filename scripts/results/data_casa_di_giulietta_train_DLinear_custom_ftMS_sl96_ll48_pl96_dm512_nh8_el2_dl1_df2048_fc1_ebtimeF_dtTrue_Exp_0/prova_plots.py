import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime

# load the dataframes
df: pd.Series = pd.read_csv(
    "../../../../main_dataset/count_data_train/data_casa_di_giulietta_train.csv",
    index_col=1,
    parse_dates=["date"],
)["count"]
df_2019: pd.Series = pd.read_csv(
    "../../../../main_dataset/count_data_test/data_casa_di_giulietta_test.csv",
    parse_dates=["date"],
    index_col="date",
)["count"]
mask = df_2019.index >= datetime(2019, 10, 19)
df_2019 = df_2019[mask]

# get the params for inverse scaling
num_train = int(len(df) * 0.7)
num_test = int(len(df) * 0.2)
num_vali = len(df) - num_train - num_test
df = df[int(len(df) * 0.7) :]
mean = np.mean(df)
std = np.std(df)

# load the predicted/true values and the dates
true = np.load("true.npy")
pred = np.load("pred.npy")
very_dates = np.load("very_dates.npy", allow_pickle=True)

very_true = true[95::96]
very_pred = pred[95::96]
very_true = np.concatenate(very_true, axis=0)
very_pred = np.concatenate(very_pred, axis=0)

# inverse scale the true/pred values
very_true = (very_true * (std + 1e-8)) + mean
very_pred = (very_pred * (std + 1e-8)) + mean
very_pred[very_pred < 0] = 0
very_pred = np.around(very_pred)

final_df = pd.DataFrame(
    very_pred, index=df_2019.index[: len(very_pred)], columns=["Predicted"]
)
final_df = pd.merge(final_df, df_2019, left_index=True, right_index=True)

print(final_df.head())

print(
    "MAPE for dlinear",
    round(
        float(
            mean_absolute_percentage_error(
                np.array(final_df["count"]), np.array(final_df["Predicted"])
            )
        )
        / 100,
        2,
    )
    * 100,
    "%",
)

plt.figure()
plt.plot(final_df.iloc[:500]["count"], label="True")
plt.plot(final_df.iloc[:500]["Predicted"], label="Predicted")
plt.legend()
plt.savefig("../../img/arena_pl_96_first500.png")

plt.figure()
plt.plot(final_df["count"], label="True")
plt.plot(final_df["Predicted"], label="Predicted")
plt.legend()
plt.savefig("../../img/arena_pl_96_full.png")

plt.show()

