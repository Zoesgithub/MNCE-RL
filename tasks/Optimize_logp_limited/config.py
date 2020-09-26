import sys
sys.path.append("../../")
from model.opt_func import logp_get_score, logp_target_get_score
import utils.datautils as dt
config={
        'predata_path':'Data/zinc/train.txt', 
        'pretest_path':'Data/zinc/valid.txt', 

        'data_path':'Data/zinc/Json/', 

        'log_path':'tasks/Optimize_logp_limited/',

        "run_mode":3,

        "reward_botem":-6.0,
        "reward_avg":6.0,
        "reward_fp":-1.0,
        "reward_target":3.0,
        "avg_moving_rate":0.0,

        "pretrain_path":"tasks/Pretrain_zinc_3l/model/pretrain.model.ckpt",
        "crit_prestep":0,
        "train_epoch":8,
        "pretrain_epoch":20,

        "score_func":dt.reward_penalized_log_p,
        "reward_func":logp_get_score,
        "query_size":500,

        "dis_layers":3,
        "gen_layers":3,

        'batchsize':8,
        "max_len":51,
        'dis_lr':1.0e-3,
        'gen_lr':2.0e-3,
        "seed":1008611
        }

