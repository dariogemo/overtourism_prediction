import pandas as pd
import torch
import numpy as np
from models.Autoformer import Model
from torch.utils.data import DataLoader, TensorDataset

# ========== CONFIG ==========

seq_len = 96        # same as during training
label_len = 48
pred_len = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'checkpoints/aut_cast_Autoformer_custom_ftS_sl96_ll48_pl1_dm128_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
day30_path = './dataset/main/day30.csv'  # must have 96 rows (one full day)

# ========== LOAD DATA ==========

df = pd.read_csv(day30_path)
data = df['count'].values.astype(np.float32)  # shape: (96,)

#assert len(data) == 96, "Your day30.csv must contain exactly 96 rows."

# Normalize or scale if needed (must match training preprocessing)
# Here we assume no scaling was done
enc_input = torch.tensor(data[-(seq_len + pred_len):-pred_len]).unsqueeze(0).unsqueeze(-1).to(device)  # [1, 96, 1]
dec_input = torch.zeros((1, label_len + pred_len, 1), dtype=torch.float32).to(device)
dec_input[0, :label_len, 0] = torch.tensor(data[-(label_len + pred_len):-pred_len])  # last 48 values

# ========== LOAD MODEL ==========
from exp.exp_basic import Exp_Basic  # or wherever Exp_*** is defined
from exp.exp_main import Exp_Main    # this is where Autoformer is loaded

# Load args (either reuse from training or define manually)
class Args:
    model = 'Autoformer'
    enc_in = 1
    dec_in = 1
    c_out = 1
    seq_len = 96
    label_len = 48
    pred_len = 1
    d_model = 128
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    dropout = 0.05
    embed = 'timeF'
    freq = '15min'
    activation = 'gelu'
    output_attention = False
    moving_avg = 25  # required by Autoformer
    use_gpu = True
    use_multi_gpu = False
    gpu = 0
    factor = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Args()

# Initialize experiment (this builds the model correctly)
exp = Exp_Main(args)
model = exp.model
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========== PREDICT ==========
def generate_time_features(timestamps, freq='15min'):
    df_stamp = pd.DataFrame({'date': pd.to_datetime(timestamps)})
    df_stamp['month'] = df_stamp['date'].dt.month
    df_stamp['day'] = df_stamp['date'].dt.day
    df_stamp['weekday'] = df_stamp['date'].dt.weekday
    df_stamp['hour'] = df_stamp['date'].dt.hour
    df_stamp['minute'] = df_stamp['date'].dt.minute
    df_stamp['minute'] = df_stamp['minute'] // 15  # since freq is 15min
    data = df_stamp.drop(['date'], axis=1).values
    return torch.tensor(data, dtype=torch.float32)

timestamps = df.index.values  # assumes this exists
x_mark_enc = generate_time_features(timestamps[-(seq_len + pred_len):-pred_len]).unsqueeze(0).to(device)
x_mark_dec = generate_time_features(timestamps[-(label_len + pred_len):]).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(enc_input, x_mark_enc, dec_input, x_mark_dec)
    prediction = output[0, 0, 0].item()

# ========== COMPARE ==========

actual = data[-1]  # last value of the 30th day
print(f"🔮 Predicted: {prediction:.2f}")
print(f"✅ Actual:    {actual:.2f}")
