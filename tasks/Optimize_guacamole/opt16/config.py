import sys
sys.path.append("../../")
from guacamol_utils.utils import benchmark_funcwrapper, make_initpop
from guacamol.benchmark_suites import goal_directed_suite_v2
benchmarks=goal_directed_suite_v2()
benchmark=benchmarks[15]
score_func=benchmark_funcwrapper(benchmark, 3.0, 1.0)

config={
        'predata_path':'Data/guacamol/train.txt', 
        'pretest_path':'Data/guacamol/valid.txt', 

        'data_path':'Data/guacamol/Json/', 

        'log_path':'tasks/Optimize_guacamole/opt16',

        "run_mode":2,

        "reward_botem":-2.0,
        "reward_fp":-1.0,
        "reward_avg":0.0,

        "pretrain_path":"tasks/Pretrain_guacamol_3l/model/pretrain.model.ckpt",
        #"pretrain_path":"tasks/Optimize_guacamole/opt16/model/pretrain.model.ckpt",
        "crit_prestep":100,
        "train_epoch":100,
        "pretrain_epoch":100,

        "score_func":score_func.score,
        "reward_func":score_func,

        "dis_layers":3,
        "gen_layers":3,

        'batchsize':32,
        "max_len":60,
        'dis_lr':1e-5,
        'gen_lr':1e-5,
        "samplenum":3200,
        "top_mode": True,
        "entropy_beta":0.5,
        "keep_positive":True,
        "pre_lr":5e-4,
        #"init_population":make_initpop("tasks/Optimize_guacamole/opt16/topk.txt" , benchmark, 512),
        "top_train":True
        }

