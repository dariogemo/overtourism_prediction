import os
import torch
from exp.exp_informer import Exp_Informer
from utils.tools import StandardScaler
import numpy as np
from utils.tools import dotdict
import pandas as pd

args = dotdict()
args.is_training = 0  # important for inference
args.model_id = 'test'
args.model = 'informer'

# Data
args.root_path = './data/main/'
args.data_path = 'df_29days.csv'
args.data = 'custom'
args.features = 'MS'
args.target = 'count'
args.freq = 't'
args.detail_freq = 't'

# Model & training
args.seq_len = 96
args.label_len = 48
args.pred_len = 24
args.enc_in = 4
args.dec_in = 4
args.c_out = 1
args.d_model = 128
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.d_ff = 2048
args.moving_avg = 25
args.factor = 1
args.dropout = 0.05
args.embed = 'timeF'
args.activation = 'tanh'
args.padding = 0
args.distil = True  # crucial to match training

# Training setup (used for loading checkpoint)
args.checkpoints = './checkpoints/'
args.batch_size = 64
args.learning_rate = 0.0001
args.loss = 'mse'
args.train_epochs = 10
args.patience = 3
args.itr = 1
args.modes = 64

# System
args.use_gpu = True
args.gpu = 0
args.use_multi_gpu = False
args.devices = '0'
args.num_workers = 10
args.des = 'Exp'

print('Args in experiment:')
print(args)

Exp = Exp_Informer(args)

setting = 'informer_custom_ftMS_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'
path = os.path.join(args.checkpoints,setting,'checkpoint.pth')

Exp.args.root_path = './data/main/'
Exp.args.data_path = 'day30.csv'

df = pd.read_csv(os.path.join(args.root_path, args.data_path))
df.drop('Unnamed: 0', axis=1, inplace=True)

args.do_predict = True
if args.do_predict:
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    prediction=Exp.predict(setting, True)
    torch.cuda.empty_cache()
else:
	prediction=None
	print(prediction)

train_df = pd.read_csv('data/main/df_29days.csv')['count']
len_train_data = int(len(train_df)*0.7)
train_df = train_df[:len_train_data]

scaler = StandardScaler()
scaler.fit(train_df.values)

prediction = np.load('/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/Code/Informer2020/results/informer_custom_ftMS_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0/real_prediction.npy')
prediction = np.around(scaler.inverse_transform_mio(prediction)).astype(int).reshape(-1)
print(prediction, 'pred')
index_true = len(df) - 96
print(np.array(df.iloc[121:]['count']), 'real')