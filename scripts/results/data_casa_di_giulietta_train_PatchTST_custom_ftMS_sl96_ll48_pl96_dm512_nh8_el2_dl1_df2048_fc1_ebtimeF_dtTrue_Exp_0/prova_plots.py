import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# load the dataframes
df: pd.Series = pd.read_csv(
    "../../../../main_dataset/count_data_train/data_casa_di_giulietta_train.csv",
    index_col=1,
    parse_dates=True,
)["count"]
df_2019: pd.Series = pd.read_csv(
    "../../../../main_dataset/count_data_test/data_casa_di_giulietta_test.csv",
    parse_dates=True,
    index_col="date",
)["count"]

# get the params for inverse scaling
num_train = int(len(df) * 0.7)
num_test = int(len(df) * 0.2)
num_vali = len(df) - num_train - num_test
df = df[int(len(df) * 0.7) :]
mean = np.mean(df)
std = np.std(df)

# DEBUG, NEW WAY TO INVERSE SCALE
mean = np.mean(df_2019)
std = np.std(df_2019)

# load the predicted/true values and the dates
true = np.load("true.npy")
pred = np.load("pred.npy")
very_dates = np.load("date.npy", allow_pickle=True)
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

# MAE and plots
mae: float = mean_absolute_error(
    np.array(final_df["count"]), np.array(final_df["Predicted"])
)
print("MAE for dlinear", round(mae, 2))

plt.figure()
plt.title("PatchTST predictions for Giulietta first 5 days", fontsize=40)
plt.plot(final_df.iloc[:500]["count"], label="True")
plt.plot(final_df.iloc[:500]["Predicted"], label="Predicted")
plt.legend()
# plt.savefig("../../img/predictions/giulietta_dlinear_first500.png")

plt.figure()
plt.title("PatchTST predictions for Giulietta 2019", fontsize=40)
plt.plot(final_df["count"], label="True")
plt.plot(final_df["Predicted"], label="Predicted")
plt.legend()
# plt.savefig("../../img/predictions/giulietta_dlinear_full.png", dpi=1000)

plt.show()
