import sys
sys.path.append("../../")
from model.opt_func import logp_get_score, logp_target_get_score, wrapper
import utils.datautils as dt
from rdkit import Chem
from loguru import logger
config={
        'predata_path':'Data/zinc/train.txt', 
        'pretest_path':'Data/zinc/valid.txt', 

        'data_path':'Data/zinc/Json/', 

        'log_path':'tasks/Optimize_logp/',

        "run_mode":2,

        "reward_botem":-1.0,
        "reward_fp":-0.0,
        "avg_moving_rate":0.0,
        "reward_target":6.0,

        "pretrain_path":"tasks/Pretrain_zinc/model/pretrain.model.ckpt",
        "crit_prestep":100,
        "train_epoch":50,
        "pretrain_epoch":20,

        "score_func":dt.reward_penalized_log_p,
        "reward_func":wrapper(logp_get_score),

        "dis_layers":3,
        "gen_layers":3,

        'batchsize':32,
        "max_len":51,
        'dis_lr':1e-4,
        'gen_lr':1e-4,
        "pre_lr":5e-4,
        "samplenum":3200,
        "top_train":True
        #"entropy_beta":0.1
        }

config["reward_avg"]=config["reward_target"]*2.0
