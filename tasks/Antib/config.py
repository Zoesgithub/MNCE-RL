import sys
sys.path.append("../../")
import utils.datautils as dt
from rdkit import Chem
config={

        'data_path':'Data/AntiB_multi/Json/', 

        'log_path':'tasks/Antib/',

        "run_mode":5,

        "reward_botem":-1.0,
        "reward_fp":-1.0,

        "pretrain_path":"tasks/Antib/model/pretrain.model.ckpt",
        "crit_prestep":100,
        "train_epoch":50,
        "pretrain_epoch":10,


        "dis_layers":3,
        "gen_layers":3,

        'batchsize':5,
        "max_len":160,
        'dis_lr':1e-4,
        'gen_lr':1e-4,
        "pre_lr":1e-3,
        "samplenum":5000,
        "top_train":True,
        "keep_fake":True
        #"entropy_beta":0.1
        }

