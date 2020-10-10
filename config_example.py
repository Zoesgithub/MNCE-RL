from guacamol_utils.utils import benchmark_funcwrapper, make_initpop
from guacamol.benchmark_suites import goal_directed_suite_v2
benchmarks=goal_directed_suite_v2()
benchmark=benchmarks[0]
score_func=benchmark_funcwrapper(benchmark, 1.5, 1.0)

config={
        'data_path':'Data/guacamol/Json/',  # the path to the Json files 

        'log_path':'tasks/Optimize_guacamole/opt1', # the path to save the trained models and the log files

        "run_mode":2, # 0: evaluation; 1: run pretraining; 2: run optimization with unlimited property evaluations; 3: run property evaluations with limited property evaluations; 4: run constrained optimizations; 5: run the antibiotic experiment

        "reward_botem":-2.0, # the minimum of the reward value
        "reward_fp":-1.0, # the reward value when the generated molecule is incompleted.
        "reward_avg":0.0, # the average value used to centralize the reward values

        "pretrain_path":"tasks/Pretrain_guacamol/model/pretrain.model.ckpt", # the path to the pre-trained model
        "crit_prestep":100, # the number of iterations to pre-train the critic
        "train_epoch":100, # the number of epochs to train the model
        "pretrain_epoch":100, # when run_mode=1, the number of epochs to pre-train the model

        "score_func":score_func.score, # the score function to evaluate the properties of the generated molecules
        "reward_func":score_func, # the reward function

        "dis_layers":3, # the number of GCN layers in critic
        "gen_layers":3, # the number of GCN layers in actor

        'batchsize':32, # the batch size
        "max_len":55, # the maximum number of steps in each trace
        'dis_lr':1e-5, # the learning rate of the critic
        'gen_lr':1e-5, # the learning rate of the actor
        "samplenum":3200, # the number of the sampled molecules to evaluate the model in each epoch
        "top_mode": True, # whether use the top mode. If True, in training process, only the top 5 in each batch receive positive reward; otherwise the actual reward value is used.
        "entropy_beta":0.5, # the weight of the entropy loss
        "pre_lr":5e-4, # when run_mode=1, the learning rate of the actor when pretraining
        "init_population":make_initpop("tasks/Optimize_guacamole/opt1/topk.txt" , benchmark, 512), #when init_population is not None, use the molecules in init_population are used to pre-train the model
        "top_train":True # whether maximize the log-likelihood of the best model found before. Empirically the training process is more stable when using top_train
        }

