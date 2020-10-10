import sys
sys.path.append("../../")
from model.opt_func import logp_target_get_score
import utils.datautils as dt
config={
        'predata_path':'Data/zinc/train.txt', 
        'pretest_path':'Data/zinc/valid.txt', 

        'data_path':'Data/zinc/Json/', 

        'log_path':'tasks/Target_logp_525/',

        "run_mode":2,

        "reward_botem":-0.5,
        "reward_avg":10.0,
        "reward_fp":-1.5,
        "reward_target":5.25,
        "avg_moving_rate":1.0,

        "pretrain_path":"tasks/Pretrain_zinc/model/pretrain.model.ckpt",
        "crit_prestep":100,
        "train_epoch":100,
        "pretrain_epoch":20,

        "score_func":dt.reward_penalized_log_p,
        "reward_func":logp_target_get_score,
        
        "dis_layers":3,
        "gen_layers":3,

        'batchsize':32,
        "max_len":51,
        'dis_lr':1e-4,
        'gen_lr':1e-4,
        "samplenum":3200,
        "entropy_beta":0.2
        }

