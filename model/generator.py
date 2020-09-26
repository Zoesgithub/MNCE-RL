import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .modelutils import gcn_layer, init_weights
import utils.datautils as dt
import NCE.toolbox as tb
from collections import deque
import networkx as nx
import re
import matplotlib.pyplot as plt
from .RL_utils import get_advantages, ppo_loss
from rdkit.Chem import rdchem, Crippen, QED
import MyGraph
from rdkit import Chem
from loguru import logger
from NCE.toolbox import global_mask
# generate graph in grammar

class Generator(nn.Module):
    AVERAGE=0.0
    TARGET=0.0
    BOTEM=-0.1
    def __init__(self, config, grammar, atoms, bonds, masks):
        """
        Init func
        :param config: the dict containing all hyper parameters
        :param grammar: the grammar dict
        :param atoms: the atoms dict
        :param bonds: the bonds dict
        :param masks: the mask dict
        """
        super(Generator, self).__init__()
        self.grammar=grammar #Productions of grammar
        self.atoms=atoms #Atoms should contain X and START
        self.bonds=bonds #Bonds should contain none
        self.masks={} #Masks related to bond type
        self.matrix=[]
        for x in masks:
            self.masks[x]=len(self.masks)
            self.matrix.append(masks[x])
        self.matrix=np.array(self.matrix)
        self.grammar={int(x):self.grammar[x] for x in self.grammar}

        MyGraph.Graph().set_atoms(atoms)

        self.num_grammar=config["grammar_size"]
        self.num_atoms=config["atom_size"]
        self.num_bonds=config["bond_size"]

        self.config=config

        self.layers=config["gen_layers"]
        self.emb_size=32
        self.bond_emb_size=8

        self.gen_gcn=[gcn_layer(self.emb_size, self.emb_size, self.bond_emb_size, bn=False) for _ in
                      range(self.layers)]
        self.gen_gcn=nn.Sequential(*self.gen_gcn)

        self.atom_linear=nn.Linear(self.num_atoms, self.emb_size, bias=True) #Same effect as embedding
        self.bond_linear=nn.Linear(self.num_bonds, self.bond_emb_size, bias=True)

        self.gen_linear=nn.Linear(self.emb_size*2, self.num_grammar) #Generate results

        self.softmax=nn.Softmax(1)
        self.optimizer=torch.optim.Adam(self.parameters(), lr=config['gen_lr'], weight_decay=1e-5)

        self.steps=10
        self.clock=self.steps
        self.step_penalty=0.03
        self.fail_penalty=-1.0
        self.ratio1=0.01
        self.ratio2=0.15
        self.iters=0
        self.diction={}
        if "random_weight" in self.config:
            self.random_weight=self.config["random_weight"]
        else:
            self.random_weight=0
        self.apply(init_weights)

    def update_old(self):
        if "old" in self.config:
            self.config["old"].load_state_dict(self.state_dict())
            self.config["old"].eval()

    def init_hidden(self, inp):
        #mol=torch.distributions.Normal(0.0,1.0).sample([inp, self.emb_size])#
        mol=(torch.ones(inp, 1)*self.atoms["START"]).cuda().long()
        adj=(torch.zeros([inp, 1, 1]).cuda()).long() #Only one atom in each mol, and adjacent matrix is zero
        return [mol, adj]

    def forward(self, inp):
        mol, adj=inp
        a1, a2, a3=adj.shape

        maskmol=mol.clamp(0.0, 1.0).unsqueeze(2).float()
        mask=(maskmol.unsqueeze(1)*maskmol.unsqueeze(2))-torch.eye(a2).unsqueeze(0).unsqueeze(3).cuda(mol.device)#+torch.eye(adj.shape[1]).cuda().unsqueeze(0).unsqueeze(3)*maskmol.unsqueeze(3)
        mask=mask.clamp(0,1)

        mol=nn.functional.one_hot(mol, self.num_atoms).float()
        adj=nn.functional.one_hot(adj, self.num_bonds).float()
        mol=self.atom_linear(mol)*maskmol
        adj=self.bond_linear(adj)*mask
        adj=adj.permute(0, 3, 1, 2)

        inp=[mol, adj, maskmol, mask]
        mol, adj, _, _=self.gen_gcn(inp)
        #mol=mol[:, 0]
        mol=torch.cat([mol[:, 0], mol.mean(1)],1)
        mol=self.gen_linear(mol)
        mol=self.softmax(mol)
        return mol

    def init_queues(self, num):
        qs=[deque() for _ in range(num)]
        return qs

    def update_queue(self, qs, pdics):
        def _sbn(x):
            return int(x.split("-")[1])
        for q, d in zip(qs, pdics):
            names=[x for x in d]
            xnames=[x for x in names if "X" in x]
            names=[x for x in names if "J" in x]
            names=sorted(names, key=_sbn)
            xnames=sorted(xnames, key=_sbn)
            for x in xnames[::-1]:
                q.append(d[x])
            for x in names[::-1]:
                q.append(d[x])

    def update_ans_queue(self, qs, trees):
        for q,t in zip(qs, trees):
            p=t.get_next()
            if p is None:
                continue
            q.appendleft(p)

    def get_mol_adj(self, graphs, indexes):
        mol, adj=graphs[0].get_mol_adjs(graphs, indexes)
        return mol, adj

    def generate_next(self, qs, graphs):
        ret=[]
        for x,g in zip(qs, graphs):
            if len(x)>0:
                ret.append(x.pop())
            else:
                n=[_ for _ in g.nodes]
                ret.append(min(n))
        return ret

    def pretrain_onestep(self, data): #The data should in [tree1, tree2, tree3...] form
        """
        """
        [x.re_init() for x in data]
        graphs=[MyGraph.Graph() for _ in data] #Save the decoded graphs
        loss=torch.zeros(len(data)).cuda() #Initialize losses
        qs=self.init_queues(len(data)) #Save the next decode indexes
        ansqs=self.init_queues(len(data)) #Save the ans

        self.zero_grad()
        self.optimizer.zero_grad()
        self.update_ans_queue(ansqs, data)
        index=-1

        step=0
        while max(len(x) for x in ansqs)>0 :
            step+=1
            if index==-1:
                mol, adj = self.init_hidden(len(data))
                _tmask=np.array([int(len(x)>0) for x in ansqs])
                tmask=torch.from_numpy(_tmask).cuda().float()
                ans=np.array([x.pop() if len(x)>0 else 0 for x
                                               in ansqs])
                masks=dt.generate_masks(graphs, index, self.masks, self.matrix, _tmask, c=True)
                nextindex = np.ones(len(data))
                nextindex*=-1
                index+=1
            else:
                nextindex = np.array(self.generate_next(qs, graphs))
                _tmask=np.array([int(len(x)>0) for x in ansqs])
                tmask=torch.tensor(_tmask).cuda().float()
                ans = np.array([x.pop() if len(x)>0
                                                 else 0 for x in ansqs])
                mol, adj=self.get_mol_adj(graphs, nextindex)
                mol=torch.tensor(mol).cuda().long()
                adj=torch.tensor(adj).cuda().long()
                masks=dt.generate_masks(graphs, nextindex, self.masks, self.matrix, _tmask, c=True)

            productions=self.forward([mol, adj]) #This value has been softmaxed
            productions=productions.clamp(1e-20, 1.0)*masks
            productions=productions/productions.sum(1, keepdim=True) #Renormalize the value

            prod_d=torch.distributions.Categorical(probs=productions)
            prod_prob=prod_d.log_prob(torch.tensor(ans).cuda())
            loss+=prod_prob*tmask
            qdics=[]

            for i, aidx in enumerate(ans):
                if _tmask[i]==1:
                    temp={}
                    qdics.append(temp)
                    tb._rewrite_node(graphs[i], self.grammar, nextindex[i], aidx, temp, self.bonds, c=True)
                else:
                    qdics.append({})
            self.update_queue(qs, qdics)
            self.update_ans_queue(ansqs, data)

        loss_=-loss.mean()
        loss_.backward()
        self.optimizer.step()
        return loss_, loss,  graphs


    def generate_reward(self, graph, tprobs, validp):
        crit=self.config["crit"]
        critdevice=next(crit.parameters()).device
        mask=(tprobs>0).float()
        rewret=validp.reshape(1, -1).cuda()*1.0
        critret=crit.forward([x.to(critdevice) for x in graph]).reshape(1, -1).to(mask.device)*mask
        return rewret, critret

    def get_ppoloss(self, return_graph=False):
        #init
        crit=self.config["crit"]
        crit.train()
        #self.train()
        fake, probs, fgraph, validp, tgraphs, tprobs, oldprobs,  complete_moles=self.samplematrix(self.config["batchsize"])

        reward=[]
        values=[]
        _over_lens=[]
        for i,x in enumerate(tgraphs):
            over_len=1.0-(abs(validp[i]-self.fail_penalty)<1e-9).float().reshape(1, -1)
            #print(over_len, self.fail_penalty)
            rew, cri=self.generate_reward(x, tprobs[i], validp[i])
            reward.append(rew)
            values.append(cri)
            _over_lens.append(over_len)
        over_lens=torch.cat(_over_lens, 0)
        over_lens=over_lens.min(0)[0]
        values.append(values[-1])
        if "keep_positive" in self.config:
            positive = self.config["keep_positive"]
        else:
            positive = False
        returns, adv, over_lens=get_advantages(values, reward, over_lens, positive)
        #print(adv, returns)
        values=torch.cat(values[:-1], 0)
        if "entropy_beta" in self.config:
            loss=ppo_loss(oldprobs, tprobs, adv, returns, values, self.config["entropy_beta"])
        else:

            loss=ppo_loss(oldprobs, tprobs, adv, returns, values)
        if return_graph:
            return loss, complete_moles
        if "keep_fake" in self.config:
            self.config["fake"]=fake
        return loss

    def sample(self, num):
        #should return graph and probability
        graphs=[MyGraph.Graph() for _ in range(num)] #Save the decoded graphs
        global_mask.clear()
        loss=torch.zeros(num).cuda() #Initialize losses
        qs=self.init_queues(num) #Save the next decode indexes
        tgraphs=[]
        index=-1
        step=0
        probs=[]
        valid_penalty=self.step_penalty
        valid_ps=[]
        oldprobs=[]
        rprobs=[]
        cprobs=[]
        tv=[]
        complete_moles=[]
        maxscore=[]

        if  self.training:
            random_weight = self.random_weight * 1.0#0.9995
            self.random_weight = random_weight
        else:
            random_weight = 0.0
        while (max(len(x) for x in qs)>0 or index==-1 ) and step<self.config["max_len"]:
            valid_pen=torch.ones(num)*valid_penalty
            right=torch.ones(num)
            step+=1
            if index==-1:
                mol, adj = self.init_hidden(num)
                _tmask=np.ones(num)
                tmask=torch.ones(num).cuda()#np.array([int(x.count()>0) for x in ansqs])
                masks=dt.generate_masks(graphs, index, self.masks, self.matrix, _tmask, c=True, penalty=self.fail_penalty)
                nextindex = np.ones(num)
                nextindex*=-1
                index+=1
            else:
                _tmask=np.array([int(len(x)>0) for x in qs])
                for i,x in enumerate(qs):
                    if len(x)==0:
                        if valid_ps[-1][i]>0:
                            try:
                                mol=tb.nodegraph2mol(graphs[i], c=True)
                                if tb.check_chemical_validity(mol, False):
                                    mol=Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                                    #k, v=dt.steric_strain_filter(mol)
                                    #if not k:
                                    #    valid_ps[-1][i]=self.fail_penalty 
                                    #    continue
                                    if "keep_fake" in self.config:
                                        valid_ps[-1][i], tmp = self._get_score(graphs[i])
                                    else:
                                        valid_ps[-1][i], tmp=self._get_score(mol)
                                    if "top_mode" in self.config and self.config["top_mode"]:
                                        maxscore.append([i, tmp, len(valid_ps)])
                                        valid_ps[-1][i]=0.0

                                    complete_moles.append([mol, tmp])
                                    tv.append(tmp)

                                else:
                                    valid_ps[-1][i]=self.fail_penalty
                                    #assert False
                            except:
                                valid_ps[-1][i]=self.fail_penalty
                                #assert False
                nextindex = np.array(self.generate_next(qs, graphs))
                mol, adj=self.get_mol_adj(graphs, nextindex)
                mol=torch.tensor(mol).cuda().long()
                adj=torch.tensor(adj).cuda().long()
                masks=dt.generate_masks(graphs, nextindex, self.masks, self.matrix, _tmask, right, valid_pen, qs, c=True, penalty=self.fail_penalty)
                tmask=torch.tensor(_tmask).cuda().float()


            productions = self.forward([mol.cuda(), adj.cuda()])  # This value has been softmaxed
            rprobs.append(productions*(1-masks))
            cprobs.append(productions*masks)
            productions = productions.clamp(1e-10, 1.0) * masks
            productions = productions / productions.sum(1, keepdim=True)*masks +torch.rand(productions.shape).to(masks.device)*masks*random_weight # Renormalize the value
            prod_d = torch.distributions.Categorical(probs=productions)
            ans=prod_d.sample()

            prod_prob = prod_d.log_prob(ans).exp()
            probs.append((prod_prob*tmask).reshape(1,-1))
            valid_ps.append(valid_pen.cuda()*tmask)
            #oldprobs.append(old_prob*tmask)
            if "old" in self.config:
                with torch.no_grad():
                    oprod=self.config["old"].forward([mol.cuda(), adj.cuda()])
                    oprod=oprod.clamp(1e-10, 1.0)*masks
                    oprod=oprod/oprod.sum(1, keepdim=True)
                    oprob=torch.distributions.Categorical(probs=oprod).log_prob(ans).exp()
                    oldprobs.append((oprob*tmask).reshape(1, -1))


            loss += prod_prob * tmask
            qdics = []
            ans=ans.cpu().numpy()
            for i, aidx in enumerate(ans):
                if _tmask[i] == 1 and right[i]>0:
                    temp = {}
                    qdics.append(temp)
                    tb._rewrite_node(graphs[i], self.grammar, nextindex[i], aidx, temp, self.bonds, c=True)
                else:
                    qdics.append({})

            tgraphs.append([mol, adj])
            self.update_queue(qs, qdics)
        ctmp=0
        for i,x in enumerate(qs):
            if len(x)>0:
                valid_ps[-1][i]=self.fail_penalty
                ctmp+=1
            elif valid_ps[-1][i]>0:
                try:
                    mol=tb.nodegraph2mol(graphs[i], c=True)
                    if tb.check_chemical_validity(mol, False):
                        mol=Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                        #k, v=dt.steric_strain_filter(mol)
                        #if not k:
                        #    valid_ps[-1][i]=self.fail_penalty 
                        #    continue
                        if "keep_fake" in self.config:
                            valid_ps[-1][i], tmp = self._get_score(graphs[i])
                        else:
                            valid_ps[-1][i], tmp = self._get_score(mol)
                        if "top_mode" in self.config and self.config["top_mode"]:
                            valid_ps[-1][i]=0.0
                            maxscore.append([i, tmp, len(valid_ps)])

                        complete_moles.append([mol, tmp])


                        tv.append(tmp)
                    else:
                        valid_ps[-1][i]=self.fail_penalty
                        #assert False
                except:
                    valid_ps[-1][i]=self.fail_penalty
                    #assert False
        if "old" in self.config:
            oldprobs=torch.cat(oldprobs,0)
        cprobs=torch.cat(cprobs, 0)
        failvalue=(torch.cat(valid_ps, 0)==self.fail_penalty).float().sum().long().cpu().numpy()
        avg=sum(tv)+failvalue*(Generator.TARGET+abs(00-Generator.TARGET))
        avg=avg/self.config["batchsize"]
        if "top_mode" in self.config and self.config["top_mode"]:
            maxscore=sorted(maxscore, key=lambda x: x[1])
            if self.config["batchsize"]>10:
                num=5
            else:
                num=2
            for i,maxi in enumerate(maxscore[-num:]):
                value=maxi[1]
                pos=maxi[2]
                maxi=maxi[0]
                valid_ps[pos-1][maxi]=1.0#max(1.0, value)
        return graphs, loss, valid_ps, tgraphs, torch.cat(probs,0), oldprobs, complete_moles#+(cprobs*torch.log(cprobs+1e-12)).sum(1).mean()*0.05

    def samplematrix(self, num):
        fake, probs, validp, tgraphs, tprobs, oldprobs, rprobs = self.sample(num)  # in graph format
        mol, adj=self.get_mol_adj(fake, [0 for _ in range(num)])
        return [torch.tensor(mol).cuda().long(),
                torch.tensor(adj).cuda().long()], probs, fake, validp, tgraphs, tprobs, oldprobs, rprobs

