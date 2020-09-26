import sys
sys.path.append("../../")
from model.opt_func import logp_get_score, logp_target_get_score, get_sim_score
import utils.datautils as dt
from rdkit import Chem
config={
        'predata_path':'Data/zinc/train.txt', 
        'pretest_path':'Data/zinc/valid.txt', 

        'data_path':'Data/zinc/Json/', 

        "run_mode":4,

        "reward_botem":-1.5,
        "reward_fp":-2.0,
        "reward_target":6.0,

        "pretrain_path":"tasks/Pretrain_zinc_3l/model/pretrain.model.ckpt",
        "crit_prestep":100,
        "train_epoch":10,
        "pretrain_epoch":0,

        "score_func":dt.reward_penalized_log_p,
        "reward_func":get_sim_score,

        "dis_layers":3,
        "gen_layers":3,

        'batchsize':32,
        "max_len":51,
        'dis_lr':1e-4,
        'gen_lr':2e-4,
        "samplenum":3200,
        "iters":50,

        }

config["reward_avg"]=config["reward_target"]*2.0
config["log_path"]='tasks/Constraint_opt/'
config["smiles_path"]="Data/zinc/Copt/opt.test.logP-SA"
