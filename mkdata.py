from loguru import logger
from rdkit import Chem
from functools import partial
import sys
import NCE.toolbox as tb
import json
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import networkx as nx
import re
from molvs import Standardizer
mols=[]
def load_smile(path):
    """
    path: the path to smile files, each line end with '\r\n'
    return: a list, each term is a mol constructed from the smiles
    """
    with open(path, 'r') as f:
        content=f.readlines()
    content=[x.strip("\r\n").split(" ")[0] for x in content]
    #content=["O=C(N[C@H]1C[C@H](c2cncc(F)c2)C1)c1ccncc1Cl"]
    ret=[]
    cout=0
    for x in content:
        mol=Chem.MolFromSmiles(x)
        if tb.check_chemical_validity(mol, False):
            Standardizer().standardize(mol)
            kekul=Chem.Kekulize(mol)
            ret.append(mol)
        else:
            #print(x)
            pass
        cout+=1
        if cout%100000==0:
            print(cout)
    return ret

def write_json(List, path):
    """
    List: content
    path: the path to write
    """
    with open(path, 'w') as f:
        f.write(json.dumps(List))
    return True

def getorder(G, namelist=None):
    nodes=G.nodes
    if namelist!=None:
        res=[x for x in nodes]
        def keyfunc(x):
            return namelist.index(nodes[x]["name"])
        res=sorted(res, key=keyfunc)
        return res

def write_dict(diction, path):
    """
    Write the grammar dictionary
    :param diction: the raw grammar dict
    :param path: the path to save the dict
    :return: None
    """
    savedict={diction[x][0]:diction[x][1] for x in diction}
    write_json(savedict, path)

def getatom_bond(diction):
    """
    Obtain the dicts of atoms and bonds from grammar dict
    :param diction: the grammar dict
    :return: the dicts of atoms and bonds
    """
    atoms={}
    bonds={}
    for x in diction:
        rhs=diction[x][2]
        for y in rhs:
            bond=re.split("\[|\]", y)
            atom=bond[0]
            if len(bond)>1:
                bond=bond[1]
                if not bond in bonds:
                    bonds[bond]=len(bonds)
            atom=re.split("<-|->", atom)
            atom=[x.split("-")[0] for x in atom]
            for z in atom:
                if not z in atoms and z!="X" and z!="J":
                    print("{}\t{}".format(z,y))
                    atoms[z]=len(atoms)

    bonds["none"]=len(bonds)
    bonds["None"]=len(bonds)
    atoms["START"]=len(atoms)
    atoms["X"]=len(atoms)
    atoms["J"]=len(atoms)
    return atoms, bonds

def create_mask(diction):
    """
    Obtain masks from grammar dict
    :param diction: the grammar dict
    :return: the dict of masks
    """
    res={}
    names=sorted([int(x) for x in diction])
    for i,n in enumerate(names):
        assert i==n
        prod=diction[str(n)]
        lhs, embedding, rhs, box, _=prod
        name=("=".join(lhs)).replace("<", "-").replace(">", "-")
        if not name in res:
            #print(name)
            res[name]=np.zeros(len(names)).tolist()
        res[name][i]+=1

    for x in res:
        #print(x, "\n")
        for i,y in enumerate(res[x]):
            if y>0:
                #print(diction[str(i)][0:3])
                pass
    return res


def _get_mol_adj(graph, atoms, idx):
    ns=[x for x in graph.nodes]

    ns.pop(ns.index(idx))
    ns.insert(0, idx)
    mol=np.zeros(len(ns))
    adj=np.zeros([len(ns), len(ns)], dtype=int)
    nx.adjacency_matrix(graph, ns, weight="myweight").toarray(out=adj)
    for i,x in enumerate(ns):
        sym=graph.nodes[x]["name"]
        sym=sym.split("-")[0]
        mol[i]=atoms[sym]
    return mol, adj


def get_mol_adj(content, atoms, bonds, grammar, i):
    tree=content[i]
    graphs, idx, ans=tb.list2graphlist(tree, grammar, bonds, len(tree))
    ret=[]
    graphs=[graphs[-1]]
    idx=[idx[-1]]
    ans=[ans[-1]]
    for g, v in zip(graphs, idx):
        ret.append(_get_mol_adj(g, atoms, v))

    return ret, idx, ans

def check_coverage(mols, grammar):
    n=len(grammar)
    tgd=grammar.copy()

    testsavelist=[]
    num=0
    for i,mol in enumerate(mols):
        start=0
        while True:
            try:
                tree, mol=tb.parsemol(mol, tgd, start=start, build_grammar=False)
            except:
                num+=1
                tgd=grammar.copy()
                break
            if len(tgd)>n:
                start+=1
                tgd=grammar.copy()
            else:
                testsavelist.append(tree)
                break
    logger.info("The data size is {} suc size is {} fail size is {}".format(len(mols), len(testsavelist), num))
    return testsavelist

def main(config):
    grammar_path=os.path.join(config["data_path"], "grammar.json")
    grammarori_path=os.path.join(config["data_path"], "grammar.jsonori")
    pretest_path=os.path.join(config["predata_path"], "test.txt")
    logger.add(os.path.join(config['log_path'], 'data.log'))
    
    pretrain_path=os.path.join(config["predata_path"], "train.txt")
    prevalid_path=os.path.join(config["predata_path"], "valid.txt")

    mols=load_smile(pretrain_path)
    grammardiction={}
    savelist=[]
    for i, mol in enumerate(mols):
        if i%10000==0:
            print (i, len(grammardiction))
        tree, mol=tb.parsemol(mol, grammardiction)
        savelist.append(tree)
        #break
    train_path=os.path.join(config["data_path"], "train.json")
    test_path=os.path.join(config["data_path"], "test.json")
    valid_path=os.path.join(config["data_path"], "valid.json")

    atom_path=os.path.join(config["data_path"], "atom.json")
    bond_path=os.path.join(config["data_path"], "bond.json")
    mask_path=os.path.join(config["data_path"], "mask.json")

    write_json(savelist, train_path)
    write_dict(grammardiction, grammar_path)
    write_json(grammardiction, grammarori_path)

    logger.info("The dataset size is {} grammar size is {}".format(len(mols), len(grammardiction)))

    with open(grammarori_path, "r") as f:
        grammardiction=json.load(f)

    testmols=load_smile(pretest_path)
    testsavelist=check_coverage(testmols, grammardiction)
    write_json(testsavelist, test_path)
    validmols=load_smile(prevalid_path)
    validsavelist=check_coverage(validmols, grammardiction)
    write_json(validsavelist, valid_path)


    with open(grammar_path, "r") as f:
        content=json.load(f)
    atoms, bonds=getatom_bond(content)
    logger.info("The atoms size is {} bonds size is {}".format(len(atoms), len(bonds)))
    write_json(atoms, atom_path)
    write_json(bonds, bond_path)
    masks=create_mask(content)
    write_json(masks, mask_path)

if __name__=='__main__':
    import importlib
    parser=argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='the path to config file')
    args=parser.parse_args()

    config = importlib.import_module(args.config)
    config=config.config

    main(config)
