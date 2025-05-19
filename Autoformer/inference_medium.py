import os
import torch
from exp.exp_main import Exp_Main
import numpy as np
from utils.tools import dotdict, StandardScaler
import pandas as pd

args = dotdict()
args.target = 'count'
args.des = 'test'
args.dropout = 0.05
args.num_workers = 10
args.gpu = 0
args.lradj = 'type1'
args.devices = '0'
args.use_gpu = True
args.use_multi_gpu = False
args.freq = '15min'
args.checkpoints = './checkpoints/'
args.bucket_size = 4
args.n_hashes = 4
#args.is_trainging = True
args.root_path = './dataset/main/'
args.data_path ='df_29days.csv'
args.model_id='arena_18_19'
args.model = 'Autoformer'
args.data = 'custom'
args.features = 'MS'
args.seq_len = 96
args.label_len = 48
args.pred_len = 12
args.e_layers = 2
args.d_layers = 1
args.n_heads = 8
args.factor = 1
args.enc_in = 4
args.dec_in =4
args.c_out = 1
args.d_model = 128
args.des = 'Exp'
args.itr = 1
args.d_ff = 2048
args.moving_avg = 25
args.factor = 1
args.distil = True
args.output_attention = False
args.patience= 3
args.learning_rate = 0.0001
args.batch_size = 64
args.embed = 'timeF'
args.activation = 'gelu'
#args.use_amp = True
args.loss = 'mse'
args.train_epochs = 10
print('Args in experiment:')
print(args)

Exp = Exp_Main(args)

setting = 'arena_18_19_Autoformer_custom_ftMS_sl96_ll48_pl12_dm128_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
path = os.path.join(args.checkpoints,setting,'checkpoint.pth')

root_path = './dataset/main/'
data_path = 'day30.csv'

Exp.args.root_path = './dataset/main/'
Exp.args.data_path = 'day30.csv'

df = pd.read_csv(os.path.join(args.root_path, args.data_path))

args.do_predict = True
if args.do_predict:
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    prediction=Exp.predict(setting, True)#data_factory做好了pred里面的batch_size=1的情况，是autoformer在informer基础之上做的
    torch.cuda.empty_cache()
    #print(prediction.shape)
else:
	prediction=None
	print(prediction)

train_df = pd.read_csv('dataset/main/df_29days.csv')

import numpy as np
import torch
import pandas as pd

def inverse_scale_prediction(pred, original_data):
    count_series = original_data['count']
    mean = count_series.mean()
    std = count_series.std()

    # Convert prediction to float
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().item()
    elif isinstance(pred, np.ndarray):
        pred = pred.reshape(-1).astype(float)

    # Inverse transform
    unscaled_pred = (pred * std) + mean
    return unscaled_pred


prediction = np.load('/home/dario/Università/Data_science/Secondo_anno-secondo_semestre/Stage/Code/Autoformer/results/arena_18_19_Autoformer_custom_ftMS_sl96_ll48_pl12_dm128_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/real_prediction.npy')
prediction = np.around(inverse_scale_prediction(prediction, train_df))
print(prediction, 'pred')
print(np.array(df.iloc[-12:]['count']), 'real')