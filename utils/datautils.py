import re
import numpy as np
from torch.utils.data import Dataset, RandomSampler, BatchSampler
import re
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
import json
from torchvision import transforms
import h5py
import rdkit
from rdkit.Chem import Draw
from molvs import Standardizer
from NCE import toolbox as tb
import torch
from loguru import logger
import random
from rdkit.Chem.Descriptors import qed, MolLogP
from .sascorer import calculateScore
import networkx as nx
import sys
sys.path.append("../")
from NCE.toolbox import global_mask
from rdkit.Chem import AllChem
import copy
import itertools
bond_keys=[x for x in tb.bond_dic]
def steric_strain_filter(mol, cutoff=0.82,
                         max_attempts_embed=1000,
                         max_num_iters=200):
    """
    Flags molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule
    :param mol: rdkit mol object
    :param cutoff: kcal/mol per angle . If minimized energy is above this
    threshold, then molecule fails the steric strain filter
    :param max_attempts_embed: number of attempts to generate initial 3d
    coordinates
    :param max_num_iters: number of iterations of forcefield minimization
    :return: True if molecule could be successfully minimized, and resulting
    energy is below cutoff, otherwise False
    """
    # check for the trivial cases of a single atom or only 2 atoms, in which
    # case there is no angle bend strain energy (as there are no angles!)
    if mol.GetNumAtoms() <= 2:
        return True

    # make copy of input mol and add hydrogens
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)

    # generate an initial 3d conformer
    #try:
    flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed, useRandomCoords=True)
    #    if flag == -1:
    #        print("Unable to generate 3d conformer p1")
    #        return False
    #except: # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
    #    print("Unable to generate 3d conformer p2")
    #    return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        #try:    # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
        ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        #except:
        #    print("Unable to get forcefield or sanitization error")
        #    return False
    else:
        print("Unrecognized atom type")
        return False

    # minimize steric energy
    #try:
    ff.Minimize(maxIts=max_num_iters)
    #except:
    #    print("Minimization error")
    #    return False

    # ### debug ###
    # min_e = ff.CalcEnergy()
    # print("Minimized energy: {}".format(min_e))
    # ### debug ###

    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)

    min_angle_e = ff.CalcEnergy()
    # print("Minimized angle bend energy: {}".format(min_angle_e))

    # find number of angles in molecule
    # TODO(Bowen): there must be a better way to get a list of all angles
    # from molecule... This is too hacky
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)  # get all
    # possible 3 atom indices groups. Currently, each angle is represented by
    #  2 duplicate groups. Should remove duplicates here to be more efficient
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles

    # print("Number of angles: {}".format(num_angles))

    avr_angle_e = min_angle_e / num_angles

    # print("Average minimized angle bend energy: {}".format(avr_angle_e))

    # ### debug ###
    # for i in range(7):
    #     termList = [['BondStretch', False], ['AngleBend', False],
    #                 ['StretchBend', False], ['OopBend', False],
    #                 ['Torsion', False],
    #                 ['VdW', False], ['Electrostatic', False]]
    #     termList[i][1] = True
    #     mmff_props.SetMMFFBondTerm(termList[0][1])
    #     mmff_props.SetMMFFAngleTerm(termList[1][1])
    #     mmff_props.SetMMFFStretchBendTerm(termList[2][1])
    #     mmff_props.SetMMFFOopTerm(termList[3][1])
    #     mmff_props.SetMMFFTorsionTerm(termList[4][1])
    #     mmff_props.SetMMFFVdWTerm(termList[5][1])
    #     mmff_props.SetMMFFEleTerm(termList[6][1])
    #     ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
    #     print('{0:>16s} energy: {1:12.4f} kcal/mol'.format(termList[i][0],
    #                                                  ff.CalcEnergy()))
    # ## end debug ###

    if avr_angle_e < cutoff:
        return True, avr_angle_e
    else:
        return False, avr_angle_e
class AntibDataset(Dataset):
    def __init__(self, hdf5_path):
        """
        :param path: the path to hdf5 file
        :param args: the required data name
        """
        self._File=h5py.File(hdf5_path, 'r', swmr=True, libver='latest')
        self.total_length=len(self._File["Mol"])



    def __getitem__(self, index):
        """
        :param index: The required index
        :return: a list of float32 data in args
        """
        return [self._File["Mol"][index], self._File["Adj"][index], self._File["Score"][index]]

    def __len__(self):
        return self.total_length

class AntibDataset_pos(Dataset):
    def __init__(self, hdf5_path):
        """
        :param path: the path to hdf5 file
        :param args: the required data name
        """
        self.File=h5py.File(hdf5_path, 'r', swmr=True, libver='latest')
        self._File={}
        idx=[i for i,x in enumerate(self.File["Score"]) if x<0.2]
        self._File["Mol"]=[self.File["Mol"][x] for x in idx]
        self._File["Adj"]=[self.File["Adj"][x] for x in idx]
        self._File["Score"]=[self.File["Score"][x] for x in idx]
        self.total_length=len(self._File["Mol"])



    def __getitem__(self, index):
        """
        :param index: The required index
        :return: a list of float32 data in args
        """
        return [self._File["Mol"][index], self._File["Adj"][index], self._File["Score"][index]]

    def __len__(self):
        return self.total_length
class AntibDataset_neg(Dataset):
    def __init__(self, hdf5_path):
        """
        :param path: the path to hdf5 file
        :param args: the required data name
        """
        self.File=h5py.File(hdf5_path, 'r', swmr=True, libver='latest')
        self._File={}
        idx=[i for i,x in enumerate(self.File["Score"]) if x>=0.2]
        self._File["Mol"]=[self.File["Mol"][x] for x in idx]
        self._File["Adj"]=[self.File["Adj"][x] for x in idx]
        self._File["Score"]=[self.File["Score"][x] for x in idx]
        self.total_length=len(self._File["Mol"])



    def __getitem__(self, index):
        """
        :param index: The required index
        :return: a list of float32 data in args
        """
        return [self._File["Mol"][index], self._File["Adj"][index], self._File["Score"][index]]

    def __len__(self):
        return self.total_length

class Tree(object):
    def __init__(self, List):
        self.tree=List
        self.queue=[]
        #print(List)
        self._parse_tree(self.tree)
        self.queue=torch.from_numpy(np.array(self.queue))
        self.index=0

    def __len__(self):
        return self.queue.shape[0]

    def _parse_tree(self, tree):
        if len(tree)==0:
            return
        #print(tree)
        try:
            p,box,n=tree
        except:
            print(tree)
            exit()
        self.queue.append(p)
        if len(box)>0:
            for x in box:
                self.queue.append(x)
        for x in n:
            self._parse_tree(x)

    def get_next(self):
        if self.index>=self.queue.shape[0]:
            return None
        ret=self.queue[self.index]
        self.index+=1
        return ret

    def re_init(self):
        self.index=0

    def pin_memory(self):
        self.queue=self.queue.pin_memory() #only works for torch.tensor
        return self

def collate_wrapper(batch):
    return batch#Tree(batch)

class treeDataset(Dataset):
    def __init__(self, path):
        self.File=path
        with open(self.File, "r") as f:
            content=json.load(f)
        self.number=len(content)
        self.content=[Tree(x) for x in content]
        logger.info("The max len of traindata is {} avg is {}".format(max([len(x) for x in self.content]), np.mean([len(x) for x in self.content])))
        logger.info("size {}".format(self.number))
        #print(self.content[0])

    def __getitem__(self, index):# there is no shuffle for the single child
        self.content[index].re_init()
        return self.content[index]

    def __len__(self):
        return self.number

class limitedTreeDataset(Dataset):
    def __init__(self, path):
        self.File=path
        with open(self.File, "r") as f:
            content=json.load(f)
        self.number=250
        content=random.sample(content, self.number)
        self.content=[Tree(x) for x in content]
        logger.info("The max len of traindata is {}".format(max([len(x) for x in self.content])))
        #print(self.content[0])

    def __getitem__(self, index):# there is no shuffle for the single child
        self.content[index].re_init()
        return self.content[index]

    def __len__(self):
        return self.number

class hdf5_jsonDataset(Dataset):
    def __init__(self, hdf5_path, json_path):
        """
        :param path: the path to hdf5 file
        :param args: the required data name
        """
        self._File=h5py.File(hdf5_path, 'r')
        self.keys=list(self._File.keys())
        self.File={}
        for x in self.keys:
            self.File[x]=np.array(self._File[x])
        self.keys=list(set([x.split("_")[1] for x in self.keys]))
        self.length=[len(self.File["Mol_{}".format(x)]) for x in self.keys]
        print(self.length)
        self.total_length=sum(self.length)
        self.LENGTH=self.length
        with open(json_path, "r") as f:
            self.Json=json.load(f)
        self.keylist={x:{} for x in self.keys}
        for line in self.Json:
            for word in line:
                self.keylist[str(word[0])][word[1]]=word[3]


    def __getitem__(self, index):
        """
        :param index: The required index
        :return: a list of float32 data in args
        """
        cidx, idx=index
        key=self.keys[cidx]
        index=idx%self.length[cidx]
        #print(key, index, self.length[self.TINDEX])
        return self.File["Mol_{}".format(key)][index].astype("float32"), self.File["Adj_{}".format(key)][index].astype("float32"), self.keylist[key][index]

    def __len__(self):
        return self.total_length


class hdf5Sampler(BatchSampler):
    def __iter__(self):
        cidx=random.randint(0, len(self.sampler.data_source.LENGTH)-1)
        batch=[]
        for idx in self.sampler:
            batch.append([cidx, idx])
            if len(batch)==self.batch_size:
                yield batch
                batch=[]
                cidx=random.randint(0, len(self.sampler.data_source.LENGTH)-1)
        if len(batch)>0 and not self.drop_last:
            yield batch
            

class hdf5Dataset(Dataset):
    def __init__(self, hdf5_path):
        """
        :param path: the path to hdf5 file
        :param args: the required data name
        """
        self._File=h5py.File(hdf5_path, 'r', swmr=True, libver='latest')
        self.keys=list(self._File.keys())
        self.File={}
        for x in self.keys:
            self.File[x]=np.array(self._File[x])
        self.keys=list(set([x.split("_")[1] for x in self.keys]))
        self.length=[self.File["Mol_{}".format(x)].shape[0] for x in self.keys]
        self.total_length=sum(self.length)

        self.LENGTH=self.length


    def __getitem__(self, index):
        """
        :param index: The required index
        :return: a list of float32 data in args
        """
        cidx, idx=index
        key=self.keys[cidx]
        index=idx%self.length[cidx]
        return self.File["Mol_{}".format(key)][index].astype("float32"), self.File["Adj_{}".format(key)][index].astype("float32")

    def __len__(self):
        return self.total_length

def compute_length_distribution(sequences, path):
    """
    :param sequences: The list of sequence
    :param path: The path to save png files
    :return: None
    """
    length=[len(item) for item in sequences]
    plt.hist(length, bins=20)
    plt.savefig(path, dpi=150)
    plt.close()

def reward_penalized_log_p(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def generate_mask(graph, index, masks, defaultv):
    assert sum(defaultv)==0
    defaultv=defaultv.copy()
    if index<0:
        return masks["START"]
    node=graph.nodes[index]
    if "mask" in node:
        tret=defaultv.copy()
        for x in node["mask"]:
            tret[x]=1
        assert tret.sum()>0
        assert tret.max()==1
        #return defaultv
    else:
        tret=defaultv.copy()+1
    lhs=node["lhs"]+node["xlist"]
    keys=[]
    diction={}
    for i,x in enumerate(lhs):
        if not x[0] in diction:
            diction[x[0]]=len(diction)
        keys.append("Y-{}--X-0[{}]".format(diction[x[0]], x[1]))
    #print(lhs)
    key="=".join(keys)

    if True:#not "none" in key:
        ret=defaultv
        if key in masks:
            ret+=np.array(masks[key])
        #if nkey in masks:
        #    ret+=np.array(masks[nkey])
        #print(keys, key)
        ret=np.minimum(ret, 1)
        #try:
        #try:
        ret=ret*tret
        assert  sum(ret)>0
        #except:
        #    print (keys)
        #    print("Wrong in mask, zero mask is returned ,please check")
        #    #ret=defaultv+1
        #    #exit()
        return ret


def generate_masks(graphs, indexes, masks, matrix, tmask, right=None, valid_pen=None, qs=None, c=False, penalty=0.5):
    penalty=abs(penalty)
    output=np.zeros(len(graphs), dtype=int)
    out_ring=[]
    mymasks=np.zeros([len(graphs), 1])
    n=len(graphs)
    if isinstance(indexes, int):
        for i in range(n):
            output[i]=masks["START"]
        return torch.tensor(matrix[output]).cuda().float()
    for it, g, idx in zip(range(n), graphs, indexes):
        if c:
            lhs=g.lhs(idx)+g.xlist(idx)
            mask=g.get_mask(idx)
            if len(mask)>0:
                rboxdict=global_mask[str(g)]
                mask=[int(x) for x in rboxdict]
            out_ring.append(mask)
            if len(mask)>0:
                mymasks[it]=0
            else:
                mymasks[it]=1
        else:
            node=g.nodes[idx]
            if "mask" in node:
                out_ring.append(node["mask"])
                assert len(out_ring[-1])>0
                mymasks[it]=0
            else:
                out_ring.append([])
                mymasks[it]=1
            lhs=node["lhs"]+node["xlist"]

        keys=[]
        diction={}
        for i,x in enumerate(lhs):
            if not x[0] in diction:
                diction[x[0]]=len(diction)
            keys.append("Y-{}--X-0[{}]".format(diction[x[0]], x[1]))

        key="=".join(keys)
        if key in masks:
            output[it]=masks[key]
        else:
            output[it]=masks["none"]
            try:
                assert tmask[it]==0
            except:
                if not (right is None):
                    print("False.mask", key)
                    right[it]=0
                    valid_pen[it]=0
                    valid_pen[it]=-penalty
                    qs[it].clear()
            mymasks[it]=2
    output=matrix[output]
    for i, x in enumerate(out_ring):
        if len(x)>0:
            pr=0
            for y in x:
                if output[i][y]>1e-1:
                    pr=1
                    output[i][y]+=1
            if pr==0:
                mymasks[i]=2
                try:
                    assert tmask[i]==0
                except:
                    if not (right is None):
                        print("False.mask", key)
                        right[i]=0
                        valid_pen[i]=0
                        valid_pen[i]=-penalty
                        qs[i].clear()
    output=output+mymasks
    output=torch.tensor(output)-1
    output=output.cuda()
    output=output.clamp(0, 1).float()
    return output

if __name__=='__main__':
    from rdkit.Chem import rdmolops
    import NCE.toolbox as tb



