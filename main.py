import sys
from utils.datautils import treeDataset, collate_wrapper, hdf5Dataset, hdf5_jsonDataset, hdf5Sampler, reward_penalized_log_p, Tree, AntibDataset, AntibDataset_pos, AntibDataset_neg
from model import generator, criterial, opt_func
from rdkit import Chem
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch
from loguru import logger
import os
import json
from tensorboardX import SummaryWriter
import argparse
from rdkit.Chem import rdchem, Crippen
from rdkit.Chem.Descriptors import qed
import rdkit
import re
import numpy as np
import NCE.toolbox as tb
from Eval.mol_distance import mol_diversity
from functools import partial
torch.manual_seed(1008611)

def load_json(path):
    with open(path, 'r') as f:
        content=json.load(f)
    return content

def check_path(path, name=None):
    if not os.path.exists(path):
        os.mkdir(path)
        return None
    List=os.listdir(path)
    if List:
        if not name is None:
            for File in List:
                Name=File
                if Name==name:
                    return os.path.join(path, name)
        else:
            Files=[_ for _ in List if  not "best" in _ and not "now" in _ and "model.ckpt" in _ and not "pretrain" in _]
            if len(Files)>0:
                return os.path.join(path, sorted(Files, key=lambda x:int(x.split("-")[-1]))[-1])
            else:
                return None
    return None

def eval_func(gen, score_func, samplenum, batchsize, eval_mode=False, keep_fake=False):
    right=0
    complete=0
    unique=0
    scores=[]
    ret=[]

    Aldict={}
    mols=[]
    gen.eval()
    if eval_mode:
        while len(ret)<samplenum:
            with torch.no_grad():
                moles, prob, _, _, _, _, _ = gen.sample(batchsize)
            for i, mol in enumerate(moles):
                try:
                    mol=tb.nodegraph2mol(mol, c=True)
                except:
                    scores.append(0.0)
                    continue
                complete+=1
                if tb.check_chemical_validity(mol):
                    right+=1
                else:
                    assert False
                smiles=Chem.MolToSmiles(mol)
                mol=Chem.MolFromSmiles(smiles)
                if keep_fake:
                    score=score_func(moles[i])
                else:
                    score=score_func(mol)
                if not smiles in Aldict:
                    Aldict[smiles]=1
                    unique+=1
                ret.append([smiles, score])
                mols.append(smiles)
                scores.append(score)
        ret=ret[:samplenum]
    else:
        for _ in range(samplenum//batchsize):
            with torch.no_grad():
                moles, prob, _, _, _, _, _ = gen.sample(batchsize)
            for i, mol in enumerate(moles):
                try:
                    mol=tb.nodegraph2mol(mol, c=True)
                except:
                    scores.append(0.0)
                    continue
                complete+=1
                if tb.check_chemical_validity(mol):
                    right+=1
                else:
                    assert False
                smiles=Chem.MolToSmiles(mol)
                mol=Chem.MolFromSmiles(smiles)
                if keep_fake:
                    score=score_func(moles[i])
                else:
                    score=score_func(mol)
                mols.append(smiles)
                if not smiles in Aldict:
                    Aldict[smiles]=1
                    unique+=1
                    ret.append([smiles, score])
                scores.append(score)


    ret = sorted(ret, key=lambda x: x[1])
    gen.train()
    if "reward_target" in gen.config:
        target=gen.config["reward_target"]
        target_write=[x for x in scores if x<target+0.25 and x>target-0.25]
        logger.info("Target right is {}/{}".format(len(target_write), len(scores)))
    logger.info("Diversity is {}".format(mol_diversity(mols)))
    return ret, scores, right, complete, unique

def allocate_memory():
    total, used= os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).read().split('\n')[0].split(',')
    total = int(total)
    total = int(total * 0.75)
    n=torch.cuda.device_count()
    for _ in range(n):
        x = torch.rand((256, 1024, total)).cuda(_)
        del x

def pretrain(config, gen, traindata=None, epochs=None):
    if traindata is None:
        treedataset = treeDataset(os.path.join(config["data_path"], "train.json"))
        traindata = DataLoader(treedataset, shuffle=True, batch_size=config['batchsize'], collate_fn=collate_wrapper,
                           pin_memory=True)
    logger.info("Train data size {}".format(len(traindata)))
    if epochs is None:
        epochs=config["pretrain_epoch"]
    if "writer" in config:
        writer=config["writer"]
    else:
        writer=None

    step=0
    model_path=os.path.join(config["log_path"], "model/")
    for epoch in range(epochs):
        for i, data in enumerate(traindata):
            loss, _, _ = gen.pretrain_onestep(data)
            if i % 500 == 0:
                step += 1
                if config["run_mode"]!=4:
                    if writer:
                        writer.add_scalar("Gen/Pretrain", loss, step)
                    torch.save({"gen": gen.state_dict(), "epoch": 0},
                           os.path.join(model_path, "pretrain.model.ckpt"))
                    logger.info("Epoch {} Loss {}".format(epoch, loss))
        if config["run_mode"]!=4:
            torch.save({"gen": gen.state_dict(), "epoch": 0},
                   os.path.join(model_path, "pretrain.model.ckpt-{}".format(epoch)))

        if epoch == 15 and epochs==20:
            for g in gen.optimizer.param_groups:
                g['lr'] = 1e-4
            logger.info("Adjust lr to 1e-4 in epoch 20")
    if config["run_mode"]!=4:
        logger.info("Finish pretrain...")

def train_steps(loss_func, steps, optimizers, step_fn=None, return_mols=False):
    losses=[]
    cm=[]
    for _ in range(steps):
        if isinstance(optimizers, list):
            for opt in optimizers:
                opt.zero_grad()
        else:
            optimizers.zero_grad()
        if return_mols:
            loss, complete_mols=loss_func(return_mols)
            cm.extend(complete_mols)
        else:
            loss=loss_func()
        loss.backward()
        if isinstance(optimizers, list):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()
        if step_fn:
            step_fn()
        losses.append(loss)
    if return_mols:
        return losses, cm
    else:
        return losses

def main(config):
    grammar=load_json(os.path.join(config['data_path'], "grammar.json"))
    atoms=load_json(os.path.join(config["data_path"], "atom.json"))
    bonds=load_json(os.path.join(config["data_path"], "bond.json"))
    for x in bonds:
        bonds[x]+=1
    for x in atoms:
        atoms[x]+=1
    mask=load_json(os.path.join(config["data_path"], "mask.json"))
    mask["none"]=np.zeros(len(grammar))


    logger.info("GRAMMAR SIZE {}".format(len(grammar)))
    logger.info("ATOMS SIZE {}".format(len(atoms)))
    logger.info("BONDS SIZE {}".format(len(bonds)))
    logger.info("MASK SIZE {}".format(len(mask)))

    ##construct models
    config['atom_size']=max([atoms[x] for x in atoms])+1
    config['bond_size']=max([bonds[x] for x in bonds])+1
    config["grammar_size"]=len(grammar)
    config["bonds"]=bonds
    config["atoms"]=atoms
    for key in config:
        logger.info("{}:{}".format(key, config[key]))

    config["grammar"]=grammar
    if config["run_mode"]!=3:
        allocate_memory()
    if config["run_mode"]==1: # Pretrain
        writer=SummaryWriter(os.path.join(config['log_path'], 'board'))
        config['writer']=writer
        logger.add (os.path.join(config['log_path'], 'pretrain.log'))
        check_path(os.path.join(config["log_path"], "model/"))
        gen=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        pretrain(config, gen)

    elif config["run_mode"]==2: #Train
        logger.add (os.path.join(config['log_path'], 'train.log'))
        writer=SummaryWriter(os.path.join(config['log_path'], 'board'))
        config['writer']=writer
        gen=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        old=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        if torch.cuda.device_count()>1:
            device=1
        else:
            device=0
        crit=criterial.Crit(config, sn=False, outlayer=False).cuda(device)

        config['gen'] = gen
        config["crit"]=crit
        config["old"]=old

        gen._get_score=config["reward_func"]
        generator.Generator.BOTEM=config["reward_botem"]
        generator.Generator.AVERAGE=config["reward_avg"]
        if "reward_target" in config:
            generator.Generator.TARGET=config["reward_target"]
        gen.fail_penalty=config["reward_fp"]

        save_path=os.path.join(config["log_path"], "model/")
        model_path=check_path(save_path)
        if model_path is None:
            model_path=config["pretrain_path"]

        if model_path is None:
            logger.info("No pretrain model found, start pretraining...")
            pretrain(config, gen)
            train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)
        else:
            logger.info("load model from {}".format(model_path))
            checkpoint=torch.load(model_path)
            logger.info("Loading gen ...")
            gen.load_state_dict(checkpoint["gen"])
            if "crit" in checkpoint:
                logger.info("Loading crit ...")
                crit.load_state_dict(checkpoint["crit"])
            else:
                gen.eval()
                #print(gen.training)
                train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)
                gen.train()

        if "init_population" in config:
            list_moleculars = []
            ori_grammar = load_json(os.path.join(config["data_path"], "grammar.jsonori"))
            orilength = len(ori_grammar)
            for smile in config["init_population"]:
                print(smile)
                molecular = Chem.MolFromSmiles(smile)
                start = 0
                Chem.Kekulize(molecular)
                while start < len(molecular.GetAtoms()):
                    tmpori = ori_grammar.copy()
                    try:
                        list_molecular, mol = tb.parsemol(molecular, tmpori, start, build_grammar=False)
                        print(tb.reward_target_molecule_similarity(mol, Chem.MolFromSmiles(smile)))
                    except:
                        start += 1
                        continue
                    if len(tmpori) == orilength:
                        list_moleculars.append(Tree(list_molecular))
                        print(len(Tree(list_molecular)))
                    start += 1
                    print(start, len(tmpori), orilength)
            traindata = DataLoader(list_moleculars, shuffle=True, batch_size=config['batchsize'], collate_fn=collate_wrapper,
                                   pin_memory=True)
            for g in gen.optimizer.param_groups:
                g['lr'] = config["pre_lr"]
            logger.info("Adjust lr to {}".format(config["pre_lr"]))
            pretrain(config, gen, traindata)

            for g in gen.optimizer.param_groups:
                g['lr'] = config["gen_lr"]
            logger.info("Adjust lr to {}".format(config["gen_lr"]))


        old.load_state_dict(gen.state_dict())
        old.eval()

        writer=config["writer"]
        score_func=config["score_func"]
        logger.info("Start training ...")

        maxv=-1008611
        ITERS=200
        if "iters" in config:
            ITERS=config["iters"]
        for epoch in range(config["train_epoch"]):
            #print(gen.training)
            losses=train_steps(gen.get_ppoloss, ITERS, [crit.optimizer, gen.optimizer], gen.update_old)
            if "top_train" in config:
                if epoch%5==4:
                    moles=[x[0] for x in gen._get_score.smilelist]
                    list_moleculars = []
                    ori_grammar = load_json(os.path.join(config["data_path"], "grammar.jsonori"))
                    orilength = len(ori_grammar)
                    for smile in moles:
                        print(smile)
                        molecular = Chem.MolFromSmiles(smile)
                        start = 0
                        Chem.Kekulize(molecular)
                        while start < len(molecular.GetAtoms()):
                            tmpori = ori_grammar.copy()
                            try:
                                list_molecular, mol = tb.parsemol(molecular, tmpori, start, build_grammar=False)
                            except:
                                start += 1
                                continue
                            if len(tmpori) == orilength:
                                list_moleculars.append(Tree(list_molecular))
                            start += 1
                    traindata = DataLoader(list_moleculars, shuffle=True, batch_size=config['batchsize'], collate_fn=collate_wrapper,
                                           pin_memory=True)
                    for g in gen.optimizer.param_groups:
                        g['lr'] = config["pre_lr"]
                    logger.info("Adjust lr to {}".format(config["pre_lr"]))
                    if "top_train_epoch" in config:
                        config["pretrain_epoch"]=config["top_train_epoch"]
                    else:
                        config["pretrain_epoch"]=10
                    pretrain(config, gen, traindata)

                    for g in gen.optimizer.param_groups:
                        g['lr'] = config["gen_lr"]
                    logger.info("Adjust lr to {}".format(config["gen_lr"]))

            ret, scores, right, complete, unique=eval_func(gen, score_func, config["samplenum"], config["batchsize"])
            if "avg_moving_rate" in config:
                if "reward_target" in config:
                    avg=[abs(x[-1]-config["reward_target"]) for x in ret]
                    avg=np.mean(avg)+config["reward_target"]
                else:
                    avg=np.mean([x[-1] for x in ret])
                generator.Generator.AVERAGE=generator.Generator.AVERAGE*(1.0-config["avg_moving_rate"])+avg*config["avg_moving_rate"]
                logger.info("Set avg as {}".format(generator.Generator.AVERAGE))
            writer.add_scalar("Gen/TrainLoss", sum(losses)/len(losses), epoch)
            torch.save({'gen': gen.state_dict(), "crit": crit.state_dict(), 'epoch': epoch, 'loss': sum(losses)/len(losses)},
                    os.path.join(save_path, 'model.ckpt-{}'.format(epoch)))


            writer.add_scalar("Eval/top1 score", ret[-1][1], epoch)
            writer.add_scalar("Eval/top2 score", ret[-2][1], epoch)
            writer.add_scalar("Eval/top3 score", ret[-3][1], epoch)
            writer.add_scalar("Eval/mean score", np.mean([x[1] for x in ret]), epoch)
            logger.info("Epoch {}: Loss {}, Right {}, Complete {}, Unique {}, Top1 {}, Top2 {}, Top3 {}".
                        format(epoch, sum(losses)/len(losses), right, complete, unique, ret[-1][1], ret[-2][1], ret[-3][1]))
            logger.info("Gen.random_wieght {}".format(gen.random_weight))
            if ret[-1][1]>maxv:
                maxv=ret[-1][1]
                torch.save({'gen': gen.state_dict(), "crit": crit.state_dict(), 'epoch': epoch, 'loss': sum(losses)/len(losses)},
                           os.path.join(save_path, 'model.ckpt-{}'.format("best")))
                logger.info("Saving best in epoch {} ...".format(epoch))
                logger.info("Top 3 molecules: {} {} {}".format(ret[-1][0], ret[-2][0], ret[-3][0]))
        logger.info("Finish training")

    elif config["run_mode"]==3: #limited optimization
        torch.manual_seed(config["seed"])
        logger.add (os.path.join(config['log_path'], 'train.log'))
        gen=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        old=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        if torch.cuda.device_count()>1:
            device=1
        else:
            device=0
        crit=criterial.Crit(config, sn=False, outlayer=False).cuda(device)

        config['gen'] = gen
        config["crit"]=crit
        config["old"]=old

        gen._get_score=config["reward_func"]
        generator.Generator.BOTEM=config["reward_botem"]
        generator.Generator.AVERAGE=config["reward_avg"]
        if "reward_target" in config:
            generator.Generator.TARGET=config["reward_target"]
        gen.fail_penalty=config["reward_fp"]

        save_path=os.path.join(config["log_path"], "model/")
        model_path=check_path(save_path)
        if model_path is None:
            model_path=config["pretrain_path"]

        if model_path is None:
            logger.info("No pretrain model found, start pretraining...")
            pretrain(config, gen)
            train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)
        else:
            logger.info("load model from {}".format(model_path))
            checkpoint=torch.load(model_path)
            logger.info("Loading gen ...")
            gen.load_state_dict(checkpoint["gen"])
            if "crit" in checkpoint:
                logger.info("Loading crit ...")
                crit.load_state_dict(checkpoint["crit"])
            else:
                train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)



        old.load_state_dict(gen.state_dict())
        old.eval()

        score_func=config["score_func"]
        logger.info("Start training ...")

        maxv=-1008611
        cm=[]
        while len(cm)<config["query_size"]:
            losses, ret=train_steps(gen.get_ppoloss, 1, [crit.optimizer, gen.optimizer], gen.update_old, return_mols=True)
            cm.extend(ret)

            ret=sorted(cm, key=lambda x: x[-1])
            if len(ret)>2:
                logger.info("CM size {} Loss {}, Top1 {}, Top2 {}, Top3 {}".
                        format(len(cm), sum(losses)/len(losses), ret[-1][1], ret[-2][1], ret[-3][1]))
        cm=cm[:config["query_size"]]
        #ret=sorted(cm, key=lambda x: x[-1])
        f=open(os.path.join(config["log_path"], "smiles.txt"), "w")
        for x in cm:
            f.writelines("{}\n".format(Chem.MolToSmiles(x[0], isomericSmiles=True)))
        logger.info("Finish training")
    elif config["run_mode"]==4: #constraint opt
        ori_grammar=load_json(os.path.join(config["data_path"], "grammar.jsonori"))
        orilength=len(ori_grammar)
        logger.add (os.path.join(config['log_path'], 'train.log'))
        with open(config["smiles_path"], "r") as f:
            smiles=f.readlines()
        for smile in smiles:
            print(smile)
            molecular=Chem.MolFromSmiles(smile)
            start=0
            list_moleculars=[]
            Chem.Kekulize(molecular)
            while start<len(molecular.GetAtoms()):
                tmpori=ori_grammar.copy()
                try:
                    list_molecular, mol=tb.parsemol(molecular, tmpori, start, build_grammar=False)
                    print(tb.reward_target_molecule_similarity(mol, Chem.MolFromSmiles(smile)))
                except:
                    start+=1
                    continue
                if len(tmpori)==orilength:
                    list_moleculars.append(Tree(list_molecular))
                    print(len(Tree(list_molecular)))
                start+=1
                print(start, len(tmpori), orilength)
            molecular=Chem.MolFromSmiles(smile)
            logger.info("Parse size {}".format(len(list_moleculars)))
            gen=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
            old=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
            if torch.cuda.device_count()>1:
                device=1
            else:
                device=0
            crit=criterial.Crit(config, sn=False, outlayer=False).cuda(device)

            config['gen'] = gen
            config["crit"]=crit
            config["old"]=old

            gen._get_score=partial(config["reward_func"], refmol=molecular)
            generator.Generator.BOTEM=config["reward_botem"]
            generator.Generator.AVERAGE=config["reward_avg"]-config["score_func"](molecular)
            generator.Generator.TARGET=config["reward_target"]
            totalresult={}
            totalresult[0.4]=[-111]
            totalresult[0.6]=[-111]
            gen.fail_penalty=config["reward_fp"]

            save_path=os.path.join(config["log_path"], "model/")
            model_path=check_path(save_path)
            if model_path is None:
                model_path=config["pretrain_path"]

            if model_path is None:
                logger.info("No pretrain model found, start pretraining...")
                pretrain(config, gen)
                train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)
            else:
                logger.info("load model from {}".format(model_path))
                checkpoint=torch.load(model_path)
                logger.info("Loading gen ...")
                gen.load_state_dict(checkpoint["gen"])
                if "crit" in checkpoint:
                    logger.info("Loading crit ...")
                    crit.load_state_dict(checkpoint["crit"])
                else:
                    train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)



            old.load_state_dict(gen.state_dict())
            old.eval()

            score_func=config["score_func"]
            logger.info("Start training ...")

            maxv=-1008611
            cm=[]
            for _ in range(500):
                pretrain(config, gen, [list_moleculars], 1)
                [x.re_init() for x in list_moleculars]

            ITERS=config["iters"]
            for epoch in range(config["train_epoch"]):
                losses=train_steps(gen.get_ppoloss, ITERS, [crit.optimizer, gen.optimizer], gen.update_old)
                ret, scores, right, complete, unique=eval_func(gen, score_func, config["samplenum"], config["batchsize"])
                if "avg_moving_rate" in config:
                    assert "reward_target" in config
                    avg=[abs(x[-1]-config["reward_target"]) for x in ret]
                    avg=np.mean(avg)+config["reward_target"]
                    generator.Generator.AVERAGE=generator.Generator.AVERAGE*(1.0-config["avg_moving_rate"])+avg*config["avg_moving_rate"]
                    logger.info("Set avg as {}".format(generator.Generator.AVERAGE))


                logger.info("Epoch {}: Loss {}, Right {}, Complete {}, Unique {}, Top1 {}, Top2 {}, Top3 {}".
                            format(epoch, sum(losses)/len(losses), right, complete, unique, ret[-1][1], ret[-2][1], ret[-3][1]))
                sim_score=[tb.reward_target_molecule_similarity(Chem.MolFromSmiles(x[0]), molecular) for x in ret]
                ret_4=[[x, y] for x,y in zip(ret, sim_score) if y<1.0 and y>0.4]
                ret_6=[[x, y] for x,y in zip(ret, sim_score) if y<1.0 and y>0.6]
                ret_4=sorted(ret_4, key=lambda x: x[0][1])
                ret_6=sorted(ret_6, key=lambda x: x[0][1])
                if len(ret_4)>0:
                    if ret_4[-1][0][1]>totalresult[0.4][0]:
                        totalresult[0.4]=[ret_4[-1][0][1], ret_4[-1][1], ret_4[-1][0][0]]
                if len(ret_6)>0:
                    if ret_6[-1][0][1]>totalresult[0.6][0]:
                        totalresult[0.6]=[ret_6[-1][0][1], ret_6[-1][1], ret_6[-1][0][0]]
                logger.info("{}".format(totalresult))


            logger.info("Finish training")
            logger.info("Final Results:#{}\t{}\t{}\t{}\t{}\t{}\t{}#".format(totalresult[0.4][0], totalresult[0.4][1], totalresult[0.4][2], totalresult[0.6][0], totalresult[0.6][1], totalresult[0.6][2], config["score_func"](molecular)))
    elif config["run_mode"]==5: #Train
        logger.add (os.path.join(config['log_path'], 'train.log'))
        writer=SummaryWriter(os.path.join(config['log_path'], 'board'))
        config['writer']=writer
        gen=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        old=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        if torch.cuda.device_count()>1:
            device=1
        else:
            device=0
        crit=criterial.Crit(config, sn=False, outlayer=False).cuda(device)
        Score_func=criterial.Crit(config, sn=False, outlayer=True).cuda(device)


        config['gen'] = gen
        config["crit"]=crit
        config["old"]=old
        config["score_func"]=Score_func.get_single_score
        antib_data=AntibDataset(os.path.join(config["data_path"], "train.hdf5"))
        antib_train_data=DataLoader(antib_data, shuffle=True, batch_size=config["batchsize"])
        config["raw_atd_pos"] = DataLoader(AntibDataset_pos(os.path.join(config["data_path"], "train.hdf5")), shuffle=True,
                                           batch_size=config["batchsize"])
        config["antib_train_data_pos"] = iter(config["raw_atd_pos"])
        config["raw_atd_neg"] = DataLoader(AntibDataset_neg(os.path.join(config["data_path"], "train.hdf5")), shuffle=True,
                                           batch_size=config["batchsize"])
        config["antib_train_data_neg"] = iter(config["raw_atd_neg"])

        gen._get_score=Score_func.get_score
        generator.Generator.BOTEM=config["reward_botem"]
        if "reward_target" in config:
            generator.Generator.TARGET=config["reward_target"]
        gen.fail_penalty=config["reward_fp"]

        save_path=os.path.join(config["log_path"], "model/")
        model_path=check_path(save_path)
        if model_path is None:
            model_path=config["pretrain_path"]
        SITERS=1000

        if model_path is None:
            logger.info("No pretrain model found, start pretraining...")
            for g in gen.optimizer.param_groups:
                g['lr'] = 1e-3
            pretrain(config, gen)
            for g in gen.optimizer.param_groups:
                g['lr'] = config["gen_lr"]
            train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)
            for g in Score_func.optimizer.param_groups:
                g['lr'] = 1e-3
            for _ in range(SITERS):
                for i, data in enumerate(antib_train_data):
                    Score_func.reg_onestep(data)
                torch.save({"score_func": Score_func.state_dict()},
                           os.path.join(save_path, "scorefunc_pretrain.model.ckpt"))
                Score_func.reg_eval(antib_train_data)
        else:
            logger.info("load model from {}".format(model_path))
            checkpoint=torch.load(model_path)
            logger.info("Loading gen ...")
            gen.load_state_dict(checkpoint["gen"])
            if "score_func" in checkpoint:
                logger.info("Loading score func ...")
                Score_func.load_state_dict(checkpoint["score_func"])
            else:
                if check_path(config["log_path"], "scorefunc_pretrain.model.ckpt"):
                    checkpoint=torch.load(os.path.join(config["log_path"], "scorefunc_pretrain.model.ckpt"))
                    Score_func.load_state_dict(checkpoint["score_func"])
                else:
                    for g in Score_func.optimizer.param_groups:
                        g['lr'] = 1e-3
                    for _ in range(SITERS):
                        for i, data in enumerate(antib_train_data):
                            Score_func.reg_onestep(data)
                        torch.save({"score_func":Score_func.state_dict()}, os.path.join(save_path, "scorefunc_pretrain.model.ckpt"))
                        Score_func.reg_eval(antib_train_data)
            if "crit" in checkpoint:
                logger.info("Loading crit ...")
                crit.load_state_dict(checkpoint["crit"])
            else:
                gen.eval()
                #print(gen.training)
                train_steps(gen.get_ppoloss, config["crit_prestep"], crit.optimizer)
                gen.train()

        for g in Score_func.optimizer.param_groups:
            g['lr'] =5e-5

        old.load_state_dict(gen.state_dict())
        old.eval()

        writer=config["writer"]

        score_func=config["score_func"]
        logger.info("Start training ...")

        maxv=-1008611
        ITERS=50
        def tmp_func():
            gen.update_old()
            Score_func.ft_onestep()
        if "iters" in config:
            ITERS=config["iters"]
        for epoch in range(config["train_epoch"]):
            #print(gen.training)

            losses=train_steps(gen.get_ppoloss, ITERS, [crit.optimizer, gen.optimizer], tmp_func)
            ret, scores, right, complete, unique=eval_func(gen, score_func, config["samplenum"], config["batchsize"], keep_fake=True)
            f = open(os.path.join(config["log_path"], "smiles{}.txt".format(epoch)), "w")
            for x in ret:
                f.writelines("{}\n".format(x[0]))
            f.close()
            if "avg_moving_rate" in config:
                if "reward_target" in config:
                    avg=[abs(x[-1]-config["reward_target"]) for x in ret]
                    avg=np.mean(avg)+config["reward_target"]
                else:
                    avg=np.mean([x[-1] for x in ret])
                generator.Generator.AVERAGE=generator.Generator.AVERAGE*(1.0-config["avg_moving_rate"])+avg*config["avg_moving_rate"]
                logger.info("Set avg as {}".format(generator.Generator.AVERAGE))
            writer.add_scalar("Gen/TrainLoss", sum(losses)/len(losses), epoch)
            torch.save({'gen': gen.state_dict(), "crit": crit.state_dict(), 'epoch': epoch, 'loss': sum(losses)/len(losses), "score_func":Score_func.state_dict()},
                       os.path.join(save_path, 'model.ckpt-{}'.format(epoch)))


            if "top_train" in config:
                if epoch%1==0:
                    moles=[x[0] for x in ret[-100:]]
                    list_moleculars = []
                    ori_grammar = load_json(os.path.join(config["data_path"], "grammar.jsonori"))
                    orilength = len(ori_grammar)
                    for smile in moles:
                        print(smile)
                        molecular = Chem.MolFromSmiles(smile)
                        start = 0
                        Chem.Kekulize(molecular)
                        while start < len(molecular.GetAtoms()):
                            tmpori = ori_grammar.copy()
                            try:
                                list_molecular, mol = tb.parsemol(molecular, tmpori, start, build_grammar=False)
                            except:
                                start += 1
                                continue
                            if len(tmpori) == orilength:
                                list_moleculars.append(Tree(list_molecular))
                            start += 1
                    traindata = DataLoader(list_moleculars, shuffle=True, batch_size=config['batchsize'], collate_fn=collate_wrapper,
                                           pin_memory=True)
                    for g in gen.optimizer.param_groups:
                        g['lr'] = config["pre_lr"]
                    logger.info("Adjust lr to {}".format(config["pre_lr"]))
                    config["pretrain_epoch"]=1
                    pretrain(config, gen, traindata)

                    for g in gen.optimizer.param_groups:
                        g['lr'] = config["gen_lr"]
                    logger.info("Adjust lr to {}".format(config["gen_lr"]))
            writer.add_scalar("Eval/top1 score", ret[-1][1], epoch)
            writer.add_scalar("Eval/top2 score", ret[-2][1], epoch)
            writer.add_scalar("Eval/top3 score", ret[-3][1], epoch)
            writer.add_scalar("Eval/mean score", np.mean([x[1] for x in ret]), epoch)
            logger.info("Epoch {}: Loss {}, Right {}, Complete {}, Unique {}, Top1 {}, Top2 {}, Top3 {}".
                        format(epoch, sum(losses)/len(losses), right, complete, unique, ret[-1][1], ret[-2][1], ret[-3][1]))
            logger.info("Gen.random_wieght {}".format(gen.random_weight))
            if ret[-1][1]>maxv:
                maxv=ret[-1][1]
                torch.save({'gen': gen.state_dict(), "crit": crit.state_dict(), 'epoch': epoch, 'loss': sum(losses)/len(losses), "score_func":Score_func.state_dict()},
                           os.path.join(save_path, 'model.ckpt-{}'.format("best")))
                logger.info("Saving best in epoch {} ...".format(epoch))
                logger.info("Top 3 molecules: {} {} {}".format(ret[-1][0], ret[-2][0], ret[-3][0]))
        logger.info("Finish training")

    else:
        assert config["run_mode"]==0
        logger.add (os.path.join(config['log_path'], 'test.log'))
        gen=generator.Generator(config, grammar, atoms, bonds, mask).cuda()
        gen._get_score=config["reward_func"]
        generator.Generator.BOTEM=config["reward_botem"]
        generator.Generator.AVERAGE=config["reward_avg"]
        gen.fail_penalty=config["reward_fp"]
        if "reward_target" in config:
            generator.Generator.TARGET=config["reward_target"]
        model_path=check_path(os.path.join(config["log_path"], "model/"), "model.ckpt-best")
        checkpoint=torch.load(model_path)
        gen.load_state_dict(checkpoint["gen"])
        score_func=config["score_func"]
        ret, scores, right, complete, unique = eval_func(gen, score_func, config["samplenum"], config["batchsize"], eval_mode=True)
        logger.info("Eval ... Right {}, Complete {}, Unique {}, Top1 {}, Top2 {}, Top3 {}".
                    format(right, complete, unique, ret[-1][1], ret[-2][1], ret[-3][1]))

        logger.info("The top 50 score is {}".format(sorted(scores)[-50]))
        logger.info("The top 50 scores avg {}".format(np.mean(sorted(scores)[-50:])))
        logger.info("The mean scores is {}".format(np.mean(scores)))

        f=open(os.path.join(config["log_path"], "smiles.txt"), "w")
        for x in ret:
            f.writelines("{}\n".format(x[0]))
        if "group_eval_func" in config:
            logger.info("{}".format(config["group_eval_func"]([x[0] for x in ret])))

if __name__=='__main__':
    import importlib
    parser=argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='the path to config file')
    args=parser.parse_args()

    config = importlib.import_module(args.config)
    config=config.config

    main(config)
