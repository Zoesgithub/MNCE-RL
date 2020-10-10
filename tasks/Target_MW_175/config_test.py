import sys
sys.path.append("../../")
from model.opt_func import MW_get_score, MW_score
import utils.datautils as dt
config={
        'predata_path':'Data/zinc/train.txt', 
        'pretest_path':'Data/zinc/valid.txt', 

        'data_path':'Data/zinc/Json/', 

        'log_path':'tasks/Target_MW_175/',

        "run_mode":0,

        "reward_botem":-0.8,
        "reward_avg":2.7,
        "reward_fp":-1.0,
        "reward_target":1.75,
        "avg_moving_rate":1.0,

        "pretrain_path":"tasks/Pretrain_zinc/model/pretrain.model.ckpt",
        "crit_prestep":100,
        "train_epoch":100,
        "pretrain_epoch":20,

        "score_func":MW_score,
        "reward_func":MW_get_score,
        
        "dis_layers":3,
        "gen_layers":3,
        "iters":200,

        'batchsize':50,
        "max_len":100,
        'dis_lr':1e-4,
        'gen_lr':1e-4,
        "samplenum":5000,
        "entropy_beta":1.0
        }

