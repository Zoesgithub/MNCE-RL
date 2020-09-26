import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from .modelutils import SN, gcn_layer, init_weights
from loguru import logger
import NCE.toolbox as tb
from scipy.linalg import block_diag
import random
from utils.datautils import treeDataset, collate_wrapper, hdf5Dataset, hdf5_jsonDataset
from .RL_utils import get_advantages, ppo_loss
from scipy import stats
from sklearn  import metrics

class Crit(nn.Module):
    def __init__(self, config, sn=False, outlayer=True):
        """
        The initial func
        :param config: The dict containing all needed parameters
        """
        super(Crit, self).__init__()
        self.atom_size=config["atom_size"]
        self.bond_size=config["bond_size"]
        self.layers=config["dis_layers"]
        self.config=config
        self.grammar=config["grammar"]
        self.bonds=config["bonds"]
        self.atoms=config["atoms"]
        self.outlayer=outlayer
        self.clock=100

        self.emb_size=32
        self.bond_emb_size=8

        self.atom_linear=nn.Linear(self.atom_size, self.emb_size, bias=True)
        self.bond_linear=nn.Linear(self.bond_size, self.bond_emb_size, bias=True)
        self.layer=[]
        for _ in range(self.layers):
            if sn:
                self.layer.append(SN(gcn_layer(self.emb_size, self.emb_size, self.bond_emb_size, sn=True)))
            else:
                self.layer.append(gcn_layer(self.emb_size, self.emb_size, self.bond_emb_size, bias=True))

        self.gcn=nn.Sequential(*self.layer)
        if sn:
            self.out_layer=SN(nn.Linear(self.emb_size, 1))
        else:
            self.out_layer = nn.Linear(self.emb_size, 1)
        self.act=nn.Sigmoid()

        self.cl=nn.NLLLoss()
        self.acc=[0,0,0]
        if "dis_optimizer" in config:
            self.optimizer=config['dis_optimizer']
        else:
            lr=config["dis_lr"]
            self.optimizer=torch.optim.Adam(self.parameters(), lr=lr,
                                           weight_decay=1e-4)
        self.apply(init_weights)

    def forward(self, inp):
        mol, adj=inp
        maskmol=mol.clamp(0.0, 1.0).float().unsqueeze(2)
        mask=(maskmol.unsqueeze(1)*maskmol.unsqueeze(2))-torch.eye(adj.shape[1]).cuda(mol.device).unsqueeze(0).unsqueeze(3)
        mask=mask.clamp(0,1)
        adj=nn.functional.one_hot(adj, self.bond_size).float()
        adj=self.bond_linear(adj)*mask # not very sure to remain
        mol=nn.functional.one_hot(mol, self.atom_size).float()
        mol=self.atom_linear(mol)*maskmol
        assert (adj*(1-mask)).sum()==0
        adj=adj.permute(0,3,1,2)
        assert (mol*(1-maskmol)).sum()==0

        inp=[mol, adj, maskmol, mask]
        mol, adj, _, _=self.gcn(inp) #the mol is in the shape of n*self.embdsize
        mol=mol.mean(1)#torch.cumsum(mol, 0)

        x=self.out_layer(mol)
        if self.outlayer:
            x=nn.Sigmoid()(x)
        return x

    def reg_onestep(self, data):
        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()
        mol, adj, score=data
        critdevice=next(self.parameters()).device
        mol=mol.cuda(critdevice)
        adj=adj.cuda(critdevice)
        score=(score.cuda(critdevice)<0.2).float().reshape(-1)
        pred=self.forward([mol, adj]).reshape(-1)
        loss=score*torch.log(pred+1e-12)+(1-score)*torch.log(1-pred+1e-12)
        loss=-loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def reg_eval(self, data):
        self.eval()
        Pred=[]
        Score=[]
        for x in data:
            mol, adj, score=x
            mol=mol.cuda()
            adj=adj.cuda()
            with torch.no_grad():
                pred=self.forward([mol, adj]).reshape(-1).cpu().numpy().tolist()
            Pred.extend(pred)
            Score.extend((score<0.2).float().numpy().tolist())
        fpr, tpr, _=metrics.roc_curve(Score, Pred)
        acc=[int(int(y>0.5)==x) for x,y in zip(Score, Pred)]
        acc=sum(acc)*1.0/len(acc)
        logger.info("Datasize is {} Pred size is {} Reg Evaluation auc is {}, acc is {}".format(len(Score), len(Pred), metrics.auc(fpr, tpr), acc))

    def get_score(self, mol):
        with torch.no_grad():
            critdevice = next(self.parameters()).device
            mol, adj = mol.get_mol_adjs([mol], [0])
            ret= self.forward(
                [torch.tensor(mol).to(critdevice), torch.tensor(adj).to(critdevice)])[0]

            ret=float(ret.cpu().numpy()[0])
            return ret, ret
    def get_single_score(self, mol):
        return self.get_score(mol)[0]
    def ft_onestep(self):
        fake=self.config["fake"]
        fakelabel=torch.ones(len(fake[0]))
        loss1=self.reg_onestep([fake[0], fake[1], fakelabel])
        try:
            pos_data=self.config["antib_train_data_pos"].next()
        except:
            self.config["antib_train_data_pos"]=iter(self.config["raw_atd_pos"])
            pos_data=self.config["antib_train_data_pos"].next()
        try:
            neg_data=self.config["antib_train_data_neg"].next()
        except:
            self.config["antib_train_data_neg"]=iter(self.config["raw_atd_neg"])
            neg_data=self.config["antib_train_data_neg"].next()
        loss2=self.reg_onestep(pos_data)
        loss3=self.reg_onestep(neg_data)
