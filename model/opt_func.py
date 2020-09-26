import sys
sys.path.append("..")
from .generator import Generator
from rdkit.Chem.Descriptors import qed
import utils.datautils as dt
from rdkit.Chem import rdMolDescriptors
from scipy.stats import norm
from loguru import logger
import numpy as np
from .Molecule import smileMol
from rdkit import Chem
from molvs import Standardizer
import NCE.toolbox as tb


molecular=Chem.MolFromSmiles(smileMol)
Standardizer().standardize(molecular)
Chem.Kekulize(molecular)
rmol=Chem.MolFromSmiles(smileMol)
Standardizer().standardize(rmol)
Chem.Kekulize(rmol)
rmol.UpdatePropertyCache()
Chem.GetSymmSSSR(rmol)
rmol.GetRingInfo().NumRings()
s=Chem.MolToSmiles(rmol, isomericSmiles=True)
rmol=Chem.MolFromSmiles(s)

class wrapper(object):
    def __init__(self, func):
        self.func=func
        self.smilelist=[]
        self.smiledict={}
        self.n=0
    def __call__(self, mol):
        score, _=self.func(mol)
        smile=Chem.MolToSmiles(mol)
        if len(self.smilelist)<300:
            if not smile in self.smiledict:
                i=0
                while i<len(self.smilelist) and self.smilelist[i][1]<score:
                    i+=1
                    continue
                self.smilelist.insert(i, [smile, score])
                self.smiledict[smile]=1.0
        elif score >self.smilelist[0][1]:
            if not smile in self.smiledict:
                i=0
                while i<len(self.smilelist) and self.smilelist[i][1]<score:
                    i+=1
                    continue
                self.smilelist.insert(i, [smile, score])
                self.smiledict.pop(self.smilelist[0][0])
                self.smilelist.pop(0)
                self.smiledict[smile]=1.0
        self.n+=1
        if self.n%6400==0:
            logger.info(self.smilelist)
        return score, score
def get_qurt(data, t=0.6):
    data=sorted(data)
    data=data[::-1]
    return data[int(len(data)*t)]

def get_prob(s1, s2):
    return norm.cdf(-s1)+norm.cdf(s2)

def get_cutoff(data, target, t=0.5, ins=10, ratio=0.85, botom=0.0):
    v=1-0.50
    #data=[(target-x) for x in data]
    mu=np.mean(data)
    sigma=np.std(data)
    #cat=mu-1.50*std
    #return cat
    ins=100
    if mu>target:
        res=0
        Dis=999999
        values=[mu-2*sigma+sigma*4*i/ins for i in range(ins+1)]
        values=values[::-1]
        for x in values:
            s1=(x-mu)/sigma
            s2=(2*target-x-mu)/sigma
            p=get_prob(s1, s2)
            dis=abs(v-p)
            if dis<Dis:
                Dis=dis
                res=x
        logger.info("The distance is {}".format(Dis))
        if abs(botom-target)<abs(res-target):
            return botom
        return res
    else:
        res=0
        Dis=999999
        values=[mu-sigma*2+sigma*4*i/ins for i in range(ins+1)]
        for x in values:
            s1=(x-mu)/sigma
            s2=(2*target-x-mu)/sigma
            p=get_prob(s2, s1)
            dis=abs(v-p)
            if dis<Dis:
                Dis=dis
                res=x
        logger.info("The distance is {}".format(Dis))
        if abs(botom-target)<abs(res-target):
            return botom
        return res


def logp_get_score(mol):
    ret=dt.reward_penalized_log_p(mol)
    ret=(ret-(Generator.TARGET*2-Generator.AVERAGE))/(Generator.TARGET)
    if ret<Generator.BOTEM:
        ret=Generator.BOTEM
    return ret, ret


def qed_get_score(mol):
    ret=qed(mol)
    ret=(ret-0.7)/0.3
    if ret<Generator.BOTEM:
        #print("TOO LOW {}".format(ret))
        ret=Generator.BOTEM
    return ret, ret

def logp_target_get_score(mol):
    ret=dt.reward_penalized_log_p(mol)
    ret=abs(ret-Generator.TARGET)
    rets=ret+Generator.TARGET
    #Generator.AVERAGE=Generator.AVERAGE*0.9999+(Generator.TARGET+ret)*0.0001
    std=max(0.15, abs(Generator.TARGET-Generator.AVERAGE))
    ret=(-ret+std)/(std)
    if ret<Generator.BOTEM:
    #    #print("TOO LOW {}".format(ret))
        ret=Generator.BOTEM
    return ret, rets

def MW_get_score(mol):
    ret=rdMolDescriptors.CalcExactMolWt(mol)/100.0
    ret=abs(ret-Generator.TARGET)
    rets=ret+Generator.TARGET
    #Generator.AVERAGE=Generator.AVERAGE*0.9999+(Generator.TARGET+ret)*0.0001
    std=abs(Generator.TARGET-Generator.AVERAGE)
    std=max(0.1, std)
    ret=(-ret+std)/(std)
    if ret<Generator.BOTEM:
    #    #print("TOO LOW {}".format(ret))
        ret=Generator.BOTEM
    return ret, rets

def MW_score(mol):
    return rdMolDescriptors.CalcExactMolWt(mol)/100.0

def get_sim_score( mol, refmol):

    sim=tb.reward_target_molecule_similarity(mol, refmol)
    if sim<0.4:
        return -1.0, -1.0
    elif sim<0.6:
        return -0.5, -0.5
    elif sim==1.0:
        return -0.4, -0.4
    else:
        logp=logp_get_score(mol)
        return logp
