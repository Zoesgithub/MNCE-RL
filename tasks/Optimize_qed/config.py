import sys
sys.path.append("../../")
from model.opt_func import qed, qed_get_score, wrapper
import utils.datautils as dt
config={
        'predata_path':'Data/zinc/train.txt', 
        'pretest_path':'Data/zinc/valid.txt', 

        'data_path':'Data/zinc/Json/', 

        'log_path':'tasks/Optimize_qed/',

        "run_mode":2,

        "reward_botem":-1.0,
        "reward_fp":-0.5,
        "reward_avg":0.7,
        #"reward_target":1.0,
        #"avg_moving_rate":0.0,

        "pretrain_path":"tasks/Pretrain_zinc_3l/model/pretrain.model.ckpt",
        "crit_prestep":100,
        "train_epoch":100,
        "pretrain_epoch":20,

        "score_func":qed,
        "reward_func":wrapper(qed_get_score),

        "dis_layers":3,
        "gen_layers":3,

        'batchsize':32,
        "max_len":51,
        'dis_lr':1e-4,
        'gen_lr':1e-4,
        "pre_lr":5e-4,
        "samplenum":3200,
        "top_train":True
        }

#config["reward_avg"]=config["reward_target"]*2.0
