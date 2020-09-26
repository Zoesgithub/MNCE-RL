import networkx as nx
from rdkit.Chem.rdchem import RWMol
from rdkit import Chem
from rdkit.Chem import AllChem
from molvs import Standardizer
import rdkit
import time
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from rdkit.Chem import Draw
import torch
from scipy.linalg import block_diag
import numpy as np
import MyGraph
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
global_mask={}
#Note that all node in the networkx should have "name" and "type"
bond_dic= {
        "SINGLE":rdkit.Chem.rdchem.BondType.SINGLE,
        "AROMATIC":rdkit.Chem.rdchem.BondType.AROMATIC,
        "DOUBLE":rdkit.Chem.rdchem.BondType.DOUBLE,
        "TRIPLE":rdkit.Chem.rdchem.BondType.TRIPLE,
        "ZERO":rdkit.Chem.rdchem.BondType.ZERO}

def refine_name(lhs, embedding, frhs):
    """
    :param lhs: the left hand side of production
    :param embedding: the embedding rules of production
    :param frhs: the right hand side of production
    :return: the lhs, embedding and rhs with no direction
    """
    lhs = [x.replace("<", "-").replace(">", "-") for x in lhs]
    frhs = [x.replace("<", "-").replace(">", "-") for x in frhs]
    embedding = [x.replace("<", "-").replace(">", "-") for x in embedding]
    return lhs, embedding, frhs

def gen_name(lhs, embedding, frhs): # rules will be firstly refined
    """
    :param lhs: left hand side of production
    :param embedding: the embedding rules of production
    :param frhs: the right hand side of production
    :return: the generated name, in string format
    """
    lhs, embedding, frhs=refine_name(lhs, embedding, frhs)
    gname = "=".join((lhs)) + "+" + "=".join((embedding)) + "+" + "=".join((frhs))
    #print(gname)
    return gname

def _get_adjs(graph, idx):
    """
    To get all adjs of idx in graph, in the format of [parent, child]
    :param graph: the graph to explore
    :param idx: the indexes of vertex, in list or int format
    :return: a list of adjs, in [parent, child] form
    """
    if not isinstance(idx, list):
        idx=[idx]
    res=[]
    idx=list(set(idx))
    for x in idx:
        for y in graph.adj[x]:
            res.append([x,y])
    return res


def _sort_by_names(List):
    """
    To sort the string in List in Y-J-X order, used to reorder the lhs
    :param List: the strings
    :return: the ordered strings
    """
    def sn(x):
        x=re.split("<-|->",x)[0]
        return int(x.split("-")[1])
    xlist=[]
    ylist=[]
    jlist=[]
    for x in List:
        if "X" in x.split("-")[0]:
            xlist.append(x)
        elif "Y" in x.split("-")[0]:
            ylist.append(x)
        elif "J" in x.split("-")[0]:
            jlist.append(x)
    #xlist=sorted(xlist, key=sn)
    #ylist=sorted(ylist, key=sn)
    #jlist=sorted(jlist, key=sn)
    return ylist+jlist+xlist


def _get_bond(graph, x, y):
    """
    To get the bond between x,y if it exists
    :param graph: the graph to explore
    :param x: the begin vertex
    :param y: the end vertex
    :return: if bond exists, return the bond name and direction, if not, return none
    """
    try:
        return [graph.get_edge_data(x,y)["name"],"->"]
    except:
        try:
            return [graph.get_edge_data(y,x)["name"], "<-"]
        except:
            return None, None

def _get_direc(x):
    """
    To get the direction from a string
    :param x:
    :return:
    """
    if "<" in x:
        return "<-"
    else:
        return "->"

def _refine_lhs(lhs):
    """
    To transform lhs to standard format
    :param lhs: the left hand side of production
    :return: the standardlized lhs
    """
    if "START" in lhs:
        return lhs
    if len(lhs)==0:
        return []
    lhs=_sort_by_names(lhs)
    diction={}
    res=[]
    for x in lhs:
        name=re.split("<-|->", x)[0]
        if not name in diction:
            diction[name]=len(diction)
        res.append("Y-{}{}X-0[{}]".format(diction[name], _get_direc(x), re.split("\[|\]", x)[1]))
    return res

def _refine_rhs(rhs):
    """
    To reformat rhs to a standarlized form
    :param rhs: the right hand side of production
    :return: the standarlized lhs
    """
    def __sort_by_num(x):
        if "<-" in x or "->" in x:
            return int(re.split("-|\[",x)[3])#, int(re.split("<-|->|-",x)[1]), re.split("\[|\]",x)[1]]
        else:
            return int(x.split("-")[1])#,0,0]
    xlist=[]
    olist=[]
    for x in rhs:
        if "X" in x:
            xlist.append(x)
        else:
            olist.append(x)
    xlist=sorted(xlist,key=__sort_by_num)
    xdict={}
    for x in xlist:
        if "<-" in x or "->" in x:
            x=re.split("<-|->|\[",x)[1]
        else:
            x=x
        if not x in xdict:
            xdict[x]=len(xdict)
    txlist=xlist
    xlist=[]
    for x in txlist:
        try:
            end=re.split("<-|->|\[", x)[1]
        except:
            end=x
        xlist.append(x.replace(end, "X-{}".format(xdict[end])))
    return olist+xlist, xdict

def _sc(digraph, graph,grammar_dic, idx, lhs,bonds, newg=None, build_grammar=True):
    """
    To parse each graph into simple cycle form
    :param digraph: directed graph
    :param graph: undirected graph
    :param grammar_dic: the dictionary to save grammar productions
    :param idx: the index of vertexes to handel
    :param lhs: the left hand size of idx
    :param bonds: the bonds related to lhs, should be in the same order of lhs
    :param newg: the list to save trees
    :return: None
    """
    def sort_by_order(x):
        return digraph.nodes[x]["order"]

    idx=sorted(list(set(idx)), key=sort_by_order)
    #print(graph.nodes, idx, bonds)
    if len(idx)==1:
        lhs=_refine_lhs(lhs)#sort_by_order(lhs)
        tgraph = graph.copy()
        tgraph.remove_node(idx[0])
        tname=digraph.nodes[idx[0]]["name"]+"-0"
        embedding=[]
        for x,y in zip(lhs, bonds):
            teb=x.replace("X",digraph.nodes[y[1]]["name"])
            teb, _, _=re.split("\[|\]", teb)
            embedding.append("{}/{}[{}]".format(x, teb, y[2]))
        #embedding=["{}/{}".format(x, x.replace("X", digraph.nodes[y[1]]["name"])) for x,y in zip(lhs,bonds)]
        rhs=[]

        adj=_get_adjs(graph,idx)
        tadjs=sorted(list(set([x[1] for x in adj])), key=sort_by_order)
        adjdict={}

        newadjs=[]
        newbonds=[]
        newlhs=[]
        for i,x in enumerate(tadjs):
            if x in adjdict:
                continue
            adjdict[x]=len(newadjs)
            newbonds.append([])
            newadjs.append([x])
            newlhs.append([])
            for y in tadjs[i+1:]:
                try:
                    path=nx.shortest_path(tgraph, x,y)
                except:
                    path=[]
                if len(path)>0:
                    if y in adjdict:
                        adjdict[x]=adjdict[y]
                        newadjs[adjdict[x]].append(x)#extend(path)
                        newadjs.pop()
                        newbonds.pop()
                        newlhs.pop()
                        break
                    else:
                        adjdict[y]=adjdict[x]
                        newadjs[adjdict[y]].append(y)#extend(path)
        newadjs=[sorted(list(set(x)),key=sort_by_order) for x in newadjs]
        for i,x in enumerate(newadjs):
            for y in x:
                bond,direc=_get_bond(digraph, idx[0],y)
                if bond!=None:
                    if len(x)>1:
                        tbond=bond
                    else:
                        tbond=bond
                    rhs.append("{}{}{}[{}]".format(tname, direc, "X-{}".format(i),tbond))
                    newlhs[i].append("Y-0{}{}[{}]".format( direc, "X-{}".format(i),tbond))
                    newbonds[i].append([idx[0],y,bond, direc])
        #print(graph.nodes, idx, adjs)
        if len(rhs)==0:
            rhs.append(tname)
        gname=gen_name(lhs, embedding, rhs)
        if not gname in grammar_dic:
            grammar_dic[gname]=[len(grammar_dic),[lhs, embedding, rhs,[], []]]
        newg.append(grammar_dic[gname][0])
        newg.append([])
        newg.append([])
        graph.remove_node(idx[0])
        for x,y,z in zip(newadjs, newlhs, newbonds):
            x=sorted(x, key=sort_by_order)
            #print(x,y,z)
            y=_refine_lhs(y)
            temp=[]
            newg[2].append(temp)
            _sc(digraph, graph,grammar_dic,x, y,z,temp, build_grammar=build_grammar)
        return

    #When len(idx)>0
    sets=set(idx)
    tgraph=graph.copy()
    newadjs=_get_adjs(graph, idx)
    newadjs=[x  for x in  newadjs if not x[1] in sets] #no intersection between adjs and idx

    def sort_by_one(x):
        return [sort_by_order(x[1]), sort_by_order(x[0])]
    newadjs=sorted(newadjs, key=sort_by_one)
    for i,x in enumerate(idx):
        digraph.nodes[x]["tname"]="J-{}".format(i)

    def update_rules(tgraph, sets,bonds, lhs, name,newg, build_grammar=True):
        #print(lhs, bonds)
        path=sets
        adjs=newadjs
        renamedic={}

        for i, x in enumerate(path):
            renamedic[digraph.nodes[x]["tname"]]="{}-{}".format(name,i)

        #print(bonds, lhs)
        tlhs=lhs
        #print(lhs, tlhs)
        trhs=[]
        tembedding=[]
        box={x:[] for x in path}

        child_adj={}
        child_bond=[]
        child_lhs=[]
        childs=[]
        tindex=0
        temp_adjs=sorted(list(set([x[1] for x in adjs])), key=sort_by_order)
        tempgraph=tgraph.copy()
        for x in path:
            tempgraph.remove_node(x)
        for i,x in enumerate(temp_adjs):
            if x in child_adj:
                continue
            child_adj[x]=tindex
            child_bond.append([])
            child_lhs.append([])
            childs.append([x])
            tindex+=1
            for y in temp_adjs[i+1:]:
                try:
                    tpath=nx.shortest_path(tempgraph, x,y)#_get_path(tgraph, x,y)
                except:
                    tpath=[]
                if y in child_adj:
                    if len(tpath)>0:
                        child_adj[x]=child_adj[y]
                        childs[child_adj[x]].append(x)#.extend(tpath)
                        childs.pop()
                        child_lhs.pop()
                        child_bond.pop()
                        tindex-=1
                        break
                if len(tpath)>0:
                    child_adj[y]=child_adj[x]
                    childs[child_adj[x]].append(y)#.extend(tpath)
        for x,y in zip(lhs, bonds):
            start=re.split("<-|->",x)[0]
            assert (y[1] in path)
            tname=renamedic[digraph.nodes[y[1]]["tname"]]
            box[y[1]].append(["{}{}{}-0[{}]".format(start, y[3], digraph.nodes[y[1]]["name"],y[2]), y[2], y[2]])
            teb="{}{}{}[{}]".format(start, y[3], tname,y[2])
            tembedding.append("{}/{}".format(x, teb))
        #print(child_adj, childs)
        for i,x in enumerate(path):
            for y in path[i+1:]:
                bond, direc=_get_bond(digraph,x,y)
                if bond !=None:
                    #tbond="none"
                    trhs.append("{}{}{}[{}]".format(renamedic[digraph.nodes[x]["tname"]],direc,
                                renamedic[digraph.nodes[y]["tname"]],"none"))
                    box[y].append(["{}{}{}[{}]".format(renamedic[digraph.nodes[x]["tname"]],direc,
                                renamedic[digraph.nodes[y]["tname"]],bond), bond, bond])
                    bond, direc=_get_bond(digraph,y,x)
                    box[x].append(["{}{}{}[{}]".format(renamedic[digraph.nodes[y]["tname"]],direc,
                                renamedic[digraph.nodes[x]["tname"]],bond), bond, "none"])

        for x in adjs:
            bond,direc=_get_bond(digraph,x[0],x[1])
            child_bond[child_adj[x[1]]].append([x[0],x[1],bond,direc])
        fchild_bond=[]
        for i,cbonds in enumerate(child_bond):
            cbonds=sorted(cbonds, key=sort_by_one)
            fchild_bond.append(cbonds)
            for x in cbonds:
                bond,direc=_get_bond(digraph,x[0],x[1])
                tbond="none"
                r="{}{}{}[{}]".format(renamedic[digraph.nodes[x[0]]["tname"]], direc,
                                      "X-{}".format(i),bond)
                tr="{}{}{}[{}]".format(renamedic[digraph.nodes[x[0]]["tname"]], direc,
                                      "X-{}".format(i),"none")
                trhs.append(tr)
                child_lhs[i].append(r)
                box[x[0]].append(["{}{}{}[{}]".format("X-{}".format(i),direc,
                                renamedic[digraph.nodes[x[0]]["tname"]],bond), bond,tbond])
        tbox=[]
        rbox=[]
        child_bond=fchild_bond
        for x in path:
            ttlhs=[x[0].replace(x[1],x[2]) for x in box[x]]
            #print(ttlhs)
            ttembedding=[x[0] for x in box[x]]
            ttlhs=_refine_lhs(ttlhs)
            ttembedding=_refine_lhs(ttembedding)
            tname=digraph.nodes[x]["name"]
            ttrhs=["{}-{}".format(tname,0)]
            ttembedding=["{}/{}".format(x,y.replace("X", tname))
                         for x,y in zip(ttlhs, ttembedding)]
            #print(ttlhs,ttembedding, box[x], tname,x,sets)
            tgname=gen_name(ttlhs, ttembedding, ttrhs)
            if not tgname in grammar_dic:
                grammar_dic[tgname]=[len(grammar_dic),[ttlhs, ttembedding, ttrhs,[], []]]
            tbox.append(ttlhs)
            rbox.append(grammar_dic[tgname][0])
        #print(trhs, tembedding,bonds)
        #trhs, xvlist=_refine_rhs(trhs)

        bn=[x for x in bond_dic]
        for name in bn:
            #tlhs=[x.replace(name, "none") for x in tlhs]
            trhs=[x.replace(name, "none") for x in trhs]
            #tembedding=[x.replace(name, "none") for x in tembedding]
        if len(trhs)==0:
            trhs=["J-0"]
        gname=gen_name(tlhs, tembedding, trhs)

        if not gname in grammar_dic:
            rboxdict={rbox[-1]:{}}
            for x in range(len(rbox)-1)[::-1]:
                rboxdict={rbox[x]:rboxdict}

            #grammar_dic[gname]=[len(grammar_dic), [tlhs, tembedding, trhs, tbox, [[x] for x in rbox]]]
            grammar_dic[gname]=[len(grammar_dic), [tlhs, tembedding, trhs, tbox, rboxdict]]
        else:
            rboxdict=grammar_dic[gname][1][4]
            for i,x in enumerate(rbox):
                #print(build_grammar, rboxdict, x)
                if not build_grammar:
                    try:
                        assert x in rboxdict
                    except:
                        x=str(x)
                        assert x in rboxdict
                if x in rboxdict:
                    rboxdict=rboxdict[x]
                else:
                    rboxdict[x]={}
                    rboxdict=rboxdict[x]
                #grammar_dic[gname][1][4][i].append(x)
        newg.append(grammar_dic[gname][0])
        newg.append(rbox)
        newg.append([])
        #print(tlhs, tembedding, trhs)
        for x in path:
            tgraph.remove_node(x)

        for x,y,z in zip(childs, child_lhs, child_bond):
            #print(x,y,z)
            x=sorted(x,key=sort_by_order)
            y=_refine_lhs(y)
            temp=[]
            newg[2].append(temp)
            _sc(digraph,tgraph.copy(), grammar_dic, x, y,z,temp, build_grammar=build_grammar)

    update_rules(tgraph,idx,bonds, lhs,"J",newg, build_grammar)

def _eg(digraph, graph, grammar_dic,newg, build_grammar=True): #extract grammar, to simple cycle and node level
    def sort_by_order(x):
        return digraph.nodes[x]["order"]
    child=sorted(digraph.adj[-1], key=sort_by_order)
    assert len(child)==1
    digraph.remove_node(-1)
    _sc(digraph, graph, grammar_dic, child, ["START"],[],newg, build_grammar=build_grammar)

def _rewrite_node(newg, grammardic, index, idx, pdic, bond_dic=None, c=False):
    """
    newg: The graph
    grammardic: The dictionary in which all productions are included
    index: The index of the node in the graph
    idx: The index of the production in the grammardic
    pdic: dictionary
    """
    def _sn(x):
        return int(x.split("-")[1])
    def __refine_pdic(diction):
        names=[x for x in diction if diction[x]!=index]
        names=sorted([x for x in names if "X" in names], key=_sn)
        for i,name in enumerate(names):
            value=diction[name]
            diction.pop(name)
            diction["X-{}".format(i)]=value

    lhs, embedding, rhs, tlsh, masks=grammardic[idx]

    __refine_pdic(pdic)
    if c:
        if newg.size>0:
            tmsk=newg.get_mask(index)
            if len(tmsk)>0:
                gmask=global_mask[str(newg)]
                #print(str(idx), gmask.keys())
                gmask=gmask[str(idx)]
                global_mask[str(newg)]=gmask
            gidx=newg.get_max+1
            assert len(newg.adj[index]) <= len(lhs)
        else:
            gidx=0
    else:
        if len(newg.nodes)>0:
            gidx=max([x for x in newg.nodes])+1
            assert len(newg.adj[index]) <= len(lhs)
        else:
            gidx=0
    nameset=[]
    xlist=[]
    for x in rhs:
        if "<-" in x or "->" in x:
            start, end,_=re.split("<-|->|\[",x)
            if "X" in start:
                xlist.append(start)
            else:
                nameset.append(start)
            if "X" in end:
                xlist.append(end)
            else:
                nameset.append(end)
        else:
            start=x
            if "X" in start:
                xlist.append(start)
            else:
                nameset.append(start)
    nameset=sorted(list(set(nameset)), key=_sn)
    xlist=sorted(list(set(xlist)), key=_sn)
    for x in nameset:
        pdic[x]=gidx
        if "J" in x:
            type="N"
        else:
            type="A"
        newg.add_node(pdic[x],lhs=[], xlist=[], name=re.split("-", x)[0], tname=x, type=type)
        gidx+=1
    for x in xlist:
        pdic[x]=gidx
        newg.add_node(pdic[x],lhs=[], xlist=[], name=x, tname=x, type="N")
        gidx+=1
    if index>-1:
        if c:
            tlhs=newg.lhs(index)
            xlist=newg.xlist(index)
        else:
            tlhs=newg.nodes[index]["lhs"]
            xlist=newg.nodes[index]["xlist"]
        #print(tlhs+xlist, embedding, lhs, rhs)
        assert len(tlhs+xlist)==len(embedding)
        for x,y in zip(tlhs, embedding[:len(tlhs)]):
            bond=re.split("/",y)[1]
            start, end, bond, _=re.split("<-|->|\[|\]",bond)
            if bond=="none":
                bond=x[1]
            if x[1]!="none" and bond!="none":
                try:
                    assert  x[1]==bond
                except:
                    print(tlhs, embedding, lhs, gen_name(lhs, embedding,rhs), index)
                    print(x[1], bond)
                    nx.draw(newg, labels={x:str(x) for x in newg.nodes})
                    print(newg.nodes[51])
                    plt.savefig("debug.png")
                    assert False
                    exit()
                    pass
            if not end in pdic:
                print(rhs, embedding)
                assert False
            if not bond_dic is None:
                myweight=bond_dic[bond]
            else:
                myweight=None
            if c:
                #print(x[0], pdic[end], bond, myweight, newg.nodes, newg.adj)
                newg.add_edge(x[0],pdic[end], name=bond,myweight=str(myweight))
                #print(newg.nodes, newg.adj)
                newg.add_lhs(pdic[end], [x[0],bond, x[2]])
                newg.modify_xlist(x[0], index, pdic[end], bond)
                newg.modify_lhs(x[0], index, pdic[end], bond)
            else:
                newg.add_edge(x[0],pdic[end], name=bond,myweight=myweight)
                newg.nodes[pdic[end]]["lhs"].append([x[0],bond, x[2]])
                for z in newg.nodes[x[0]]["xlist"]:
                    if z[0]==index:
                        z.pop(0)
                        z.pop(0)
                        z.insert(0, bond)
                        z.insert(0, pdic[end])
                for z in newg.nodes[x[0]]["lhs"]:
                    if z[0]==index:
                        z.pop(0)
                        z.pop(0)
                        z.insert(0,bond)
                        z.insert(0, pdic[end])
    for x in rhs:
        try:
            start, end, bond,_=re.split("<-|->|\[|\]",x)
        except:
            start=x
            end=None
            bond=None

        if end !=None:
            if "X" in end:
                if c:
                    newg.add_xlist(pdic[start], [pdic[end],bond, end])
                    newg.add_lhs(pdic[end], [pdic[start],bond, "Y-{}".format(start.split("-"    )[1])])
                else:
                    newg.nodes[pdic[start]]["xlist"].append([pdic[end],bond, end])
                    newg.nodes[pdic[end]]["lhs"].append([pdic[start],bond, "Y-{}".format(start.split("-")[1])])
            else:
                if c:
                    newg.add_lhs(pdic[start], [pdic[end],bond, end])
                    newg.add_lhs(pdic[end], [pdic[start],bond, start])
                else:
                    newg.nodes[pdic[start]]["lhs"].append([pdic[end],bond, end])
                    newg.nodes[pdic[end]]["lhs"].append([pdic[start],bond, start])
            if not bond_dic is None:
                myweight=bond_dic[bond]
            else:
                myweight=None
            if c:
                newg.add_edge(pdic[start], pdic[end], name=bond, myweight=str(myweight))
            else:
                newg.add_edge(pdic[start], pdic[end], name=bond, myweight=myweight)
    if index>-1:
        if c:
            tlhs=newg.lhs(index)
        else:
            tlhs=newg.nodes[index]["lhs"]
        for x, y in zip(xlist, embedding[len(tlhs):]):
            bond = re.split("/", y)[1]
            start, end, bond, _ = re.split("<-|->|\[|\]", bond)
            if bond == "none":
                bond = x[1]
            if x[1] != "none" and bond != "none":
                assert x[1] == bond
            if not end in pdic:
                assert False
            if not bond_dic is None:
                myweight=bond_dic[bond]
            else:
                myweight=None
            if c:
                newg.add_edge(x[0], pdic[end], name=bond, myweight=str(myweight))
                newg.add_lhs(pdic[end], [x[0], bond, x[2]])
                newg.modify_lhs(x[0], index, pdic[end], bond)
            else:
                newg.add_edge(x[0], pdic[end], name=bond, myweight=myweight)
                newg.nodes[pdic[end]]["lhs"].append([x[0], bond, x[2]])
                for z in newg.nodes[x[0]]["lhs"]:
                    if z[0] == index:
                        z.pop(0)
                        if z[0]=="none":
                            z.pop(0)
                            z.insert(0, bond)
                            z.insert(0, pdic[end])
                            break
                        else:
                            z.insert(0, pdic[end])
    jlist=[x for x in pdic if "J" in x]
    if len(jlist)>0:
        rboxdict=masks
        if c:
            global_mask[str(newg)]=rboxdict
        for i in range(len(jlist)):
            tmask=[1]
            name="J-{}".format(i)
            if c:
                newg.add_mask(pdic[name], list(set(tmask)))
            else:
                newg.nodes[pdic[name]]["mask"]=list(set(tmask))

    if index<0:
        return pdic
    ####Debuging 
    s1=("=".join(lhs)).replace("<", "-").replace(">", "-")
    ret=[]
    ndiction={}
    for x in tlhs+xlist:
        if not x[0] in ndiction:
            ndiction[x[0]]=len(ndiction)
        ret.append("Y-{}--X-0[{}]".format(ndiction[x[0]], x[1]))
    s2="=".join(ret)
    #if not "none" in s1 and not "none" in s2:
    try:
        assert s1==s2
    except:
        print(s1, s2, index)
    ###END debug
    newg.remove_node(index)

    return pdic



def reconstruct(tree, diction, newg=None, inidx=None, pdic={}):
    """
    Reconstruct a molecular graph from a parse tree
    :param tree: the parse tree
    :param diction: the grammar dict
    :param newg: the reconstructed graph
    :param inidx: the node to be rewrite
    :param pdic: the diction of label:node
    :return: None
    """
    def sort_by_name(x):
        if isinstance(x, list):
            x=x[-1]
        return int(x.split("-")[1])

    p,box, n=tree
    lhs, embedding, rhs, tbox, masks=diction[p]
    pdic={}

    if lhs[0]=="START":
        inidx=-1
    #print(pdic)
    pdic=_rewrite_node(newg, diction, inidx, p,pdic)
    xlist=[[pdic[x],x] for x in pdic if "X" in x]
    xlist=sorted(xlist, key=sort_by_name)
    stack=[]

    if len(box)>0:
        for i,x in enumerate(box):
            stack.append([pdic["J-{}".format(i)], x])
    for i in range(len(stack)):
        idx, value=stack[i]
        _rewrite_node(newg, diction, idx, value, {})

    for x,y in zip(xlist, n):
        reconstruct(y, diction, newg, x[0]) # difinitly no mistake here



def extract_grammar(G, grammardict, newg=None, start=None, build_grammar=True):
    """
    Extract productions from a graph G
    :param G: the graph
    :param grammardict: the grammar dict
    :param newg: the reconstructed graph
    :param start: the start node of G
    :return: the parse tree
    """
    def sort_by_adj(x):
        return [len(G.adj[x]), G.nodes[x]["name"]]
    digraph=nx.DiGraph()
    digraph.add_node(-1, name="START", order=0) # add start node
    if start is None:
        start=[x for x in G.nodes if len(G.adj[x])==1]
        if len(start)==0:
            start=0
        else:
            start=min(start)
    stack=[[-1,start]]
    pdic={-1:0}
    while len(stack)>0:
        p,n=stack.pop(0)
        if n in pdic:
            if digraph.get_edge_data(p,n)==None and digraph.get_edge_data(n,p)==None:
                digraph.add_edge(p,n, name=G.get_edge_data(p,n)["name"])
            continue
        pdic[n]=len(pdic)
        digraph.add_node(n, name=G.nodes[n]["name"], order=pdic[n])
        #print (digraph.nodes[n])
        if p!=n:
            name=G.get_edge_data(p,n)
            if name!=None:
                digraph.add_edge(p,n, name=name["name"])
            else:
                digraph.add_edge(p,n, name="start") # mark the edge between the start node and the other node
        adjs=G.adj[n]
        adjs=sorted(adjs, key=sort_by_adj)
        for x in adjs:
            if not x in pdic:
                stack.append([n,x])
            else:
                if digraph.get_edge_data(n,x)==None and digraph.get_edge_data(x,n)==None:
                    digraph.add_edge(n,x, name=G.get_edge_data(n,x)["name"])
    tree=[]
    #print(G.adj[13], G.adj[15])
    _eg(digraph, G, grammardict, tree, build_grammar=build_grammar)
    gdic={grammardict[x][0]:grammardict[x][1] for x in grammardict}
    if newg ==None:
        newg=nx.Graph()
    reconstruct(tree, gdic, newg)
    #newg=G
    return tree



CHI_dict={
"CHI_UNSPECIFIED":rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
     "CHI_TETRAHEDRAL_CW":rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
     "CHI_TETRAHEDRAL_CCW":rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
}

formal_charge_dic={-1:0, 0:1, 1:2, 2:3, 3:4, -2: 5, 5:6}
reformal_charge_dic={0:-1, 1:0, 2:1, 3:2, 4:3, 5: -2, 6:5}
bonddir_dict={'BEGINDASH': rdkit.Chem.rdchem.BondDir.BEGINDASH, 'BEGINWEDGE': rdkit.Chem.rdchem.BondDir.BEGINWEDGE, 'EITHERDOUBLE': rdkit.Chem.rdchem.BondDir.EITHERDOUBLE, 'ENDDOWNRIGHT': rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT, 'ENDUPRIGHT': rdkit.Chem.rdchem.BondDir.ENDUPRIGHT, 'NONE': rdkit.Chem.rdchem.BondDir.NONE, 'UNKNOWN': rdkit.Chem.rdchem.BondDir.UNKNOWN}
plb=Chem.GetPeriodicTable()


def sort_atom_by_chiral(idx, graph, center):
    ret_dict = {x: i for i, x in enumerate(idx)}
    pair_idx = [[x, graph.nodes[x]["name"].split("_")[0]] for x in idx]
    key = 1
    while key == 1:
        key = 0
        pair_idx = sorted(pair_idx, key=lambda x: ret_dict[x[0]])
        for i in range(len(pair_idx) - 1):
            weight1 = plb.GetAtomicWeight(pair_idx[i][1])
            weight2 = plb.GetAtomicWeight(pair_idx[i + 1][1])
            neighbor1=[pair_idx[i][0]]
            neighbor2=[pair_idx[i+1][0]]
            if not center is None:
                bond1=[graph.adj[center][neighbor1[0]]["name"]]
                bond2=[graph.adj[center][neighbor2[0]]["name"]]
            else:
                bond1=[]
                bond2=[]
            p1dict={pair_idx[i][0]:1, center:1}
            p2dict={pair_idx[i+1][0]:1, center:1}
            while abs(weight1-weight2)<0.1:
                n1=len(neighbor1)
                n2 = len(neighbor2)
                bonus1=0
                bw1=0
                bw2=0
                bonus2=0
                bonus_c1=0
                bonus_c2=0
                if n1==n2 and n1==0:
                    #print(bond1, bond2, weight1, weight2)
                    return sorted(idx)
                if n1==0:
                    weight2+=weight1
                    break
                if n2==0:
                    weight1+=weight2
                    break
                for p in range(n1):
                    neighbor=graph.adj[neighbor1[0]]
                    for c in neighbor:
                        if not c in p1dict:
                            neighbor1.append(c)
                            weight1+=plb.GetAtomicWeight(graph.nodes[c]["name"].split("_")[0])
                            bonus1+=int(graph.nodes[c]["name"].split("_")[1])
                            bonus_c1+=int(graph.nodes[c]["name"].split("_")[2])
                            if "isaromatic" in graph.nodes[c] and int(graph.nodes[c]["isaromatic"])>0:
                                bw1+=graph.nodes[c]["isaromatic"]*10
                            else:
                                bond1.append(graph.adj[neighbor1[0]][c]["name"])
                            p1dict[c]=1
                    neighbor1.pop(0)

                for p in range(n2):
                    neighbor = graph.adj[neighbor2[0]]
                    for c in neighbor:
                        if not c in p2dict:
                            neighbor2.append(c)
                            weight2 += plb.GetAtomicWeight(graph.nodes[c]["name"].split("_")[0])
                            bonus2+=int(graph.nodes[c]["name"].split("_")[1])
                            bonus_c2+=int(graph.nodes[c]["name"].split("_")[2])
                            if "isaromatic" in graph.nodes[c] and int(graph.nodes[c]["isaromatic"])>0:
                                bw2+=graph.nodes[c]["isaromatic"]*10
                            else:
                                bond2.append(graph.adj[neighbor2[0]][c]["name"])
                            p2dict[c]=1
                    neighbor2.pop(0)
                if abs(weight1-weight2)<0.1:
                    for b in bond1:
                        if "AROMATIC" in b:
                            bw1 += 10
                        elif "DOUBLE" in b:
                            bw1 += 500
                        elif "TRIPLE" in b:
                            bw1 += 10000
                        elif "SINGLE" in b:
                            bw1+=1
                    for b in bond2:
                        if "AROMATIC" in b:
                            bw2 += 10
                        elif "DOUBLE" in b:
                            bw2 += 500
                        elif "TRIPLE" in b:
                            bw2 += 10000
                        elif "SINGLE" in b:
                            bw2 += 1
                    #print(bw1, bw2)
                    if bw1>bw2:
                        weight1+=weight2
                    elif bw1<bw2:
                        weight2+=weight1
                    else:
                        weight1+=bonus1
                        weight2+=bonus2
                if abs(weight1-weight2)<0.1:
                    weight1+=bonus_c1
                    weight2+=bonus_c2
            if weight2 < weight1:
                tmp_idx = ret_dict[pair_idx[i][0]]
                ret_dict[pair_idx[i][0]] = ret_dict[pair_idx[i + 1][0]]
                ret_dict[pair_idx[i + 1][0]] = tmp_idx
                key = 1
    return sorted(idx, key=lambda x: ret_dict[x]) #small to large


def get_atom_from_name(name):
    name = re.split("_", name)
    symbol = name[0]
    nehs = int(name[1])
    fc = reformal_charge_dic[int(name[2])]
    cw = CHI_dict["_".join(name[3:-1])]
    atom = Chem.Atom(symbol)
    #isaromatic = int(name[-2])
    nre = int(name[-1])
    atom.SetNumExplicitHs(nehs)
    atom.SetFormalCharge(fc)
    atom.SetChiralTag(cw)
    #atom.SetIsAromatic(isaromatic)
    atom.SetNumRadicalElectrons(nre)
    return atom

def nodegraph2molwithnodeidx(graph, c=False, node_sorted=None, root=0):
    """
    Decode molecular graph to molecules
    :param graph: the molecular graph
    :param c: whether use the c package
    :return: the molecule
    """
    emol=RWMol()
    nodes=graph.nodes
    diction={}
    if node_sorted is None:
        Node=list(nodes.keys())
    else:
        Node=node_sorted

    for x in Node:
        if nodes[x]["name"]=="START"  or x in diction:
            continue
        name=nodes[x]["name"]
        atom=get_atom_from_name(name)
        index=emol.AddAtom(atom)
        diction[x]=index

    if c:
        adjs=graph.adj
        for i,x in enumerate(Node):
            for y in Node[i+1:]:
                if x==y:
                    continue
                if y in adjs[x]:
                    bond=adjs[x][y]["name"]
                    bond, Dir=bond.split("_")
                    bond=bond_dic[bond]
                    Dir=bonddir_dict[Dir]
                    head = diction[x]
                    tail = diction[y]
                    emol.AddBond(head, tail, bond)
    else:

        for i,x in enumerate(Node):
            for y in Node[i+1:]:
                if x==y:
                    continue
                if graph.get_edge_data(x, y)!=None or graph.get_edge_data(y, x)!=None:
                    try:
                        bond=graph.get_edge_data(x,y)["name"]
                    except:
                        bond=graph.get_edge_data(y,x)["name"]
                    bond, Dir=bond.split("_")
                    bond=bond_dic[bond]
                    Dir=bonddir_dict[Dir]
                    head = diction[x]
                    tail = diction[y]
                    tmp = emol.AddBond(head, tail, bond)
                    emol.GetBondWithIdx(tmp - 1).SetBondDir(Dir)
    mol=emol.GetMol()
    mol.UpdatePropertyCache()
    s=Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True, rootedAtAtom=root, allHsExplicit=False)
    rev_diction = {diction[x]: x for x in diction.keys()}
    order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']  #
    order=list(order)
    return mol, diction, rev_diction, order, s

def nodegraph2molwithbondidx(graph, c=False, node_sorted=None, bond_sorted=None, root=0, node_dict=None):
    """
    Decode molecular graph to molecules
    :param graph: the molecular graph
    :param c: whether use the c package
    :return: the molecule
    """
    emol=RWMol()
    nodes=node_dict
    diction={}
    if node_sorted is None:
        Node=list(nodes.keys())
    else:
        Node=node_sorted
    for x in Node:
        if nodes[x]["name"]=="START"  or x in diction:
            continue
        name=nodes[x]["name"]
        atom=get_atom_from_name(name)
        index=emol.AddAtom(atom)
        diction[x]=index

    if c:
        adjs=graph.adj
        for x in bond_sorted:
            x,y=x
            bond=adjs[x][y]["name"]
            if "UP" in bond or "DOWN" in bond:
                old_key = "{}_{}".format(x,y)
                _sorted_bond = sort_atom_by_chiral([x,y],graph, None)
                new_key = [str(_) for _ in _sorted_bond]
                new_key = "_".join(new_key)
                if old_key != new_key:
                    if "UP" in bond:
                        bond = bond.replace("UP", "DOWN")
                    else:
                        assert "DOWN" in bond
                        bond = bond.replace("DOWN", "UP")
            bond, Dir=bond.split("_")
            bond=bond_dic[bond]
            Dir=bonddir_dict[Dir]
            head = diction[x]
            tail = diction[y]
            tmp = emol.AddBond(head, tail, bond)
            emol.GetBondWithIdx(tmp - 1).SetBondDir(Dir)
    else:
        for x in bond_sorted:
            x,y=x
            bond=graph.get_edge_data(x,y)["name"]
            if "UP" in bond or "DOWN" in bond:
                old_key = "{}_{}".format(x,y)
                _sorted_bond = sort_atom_by_chiral([x,y],graph, None)
                new_key = [str(_) for _ in _sorted_bond]
                new_key = "_".join(new_key)
                if old_key != new_key:
                    if "UP" in bond:
                        bond = bond.replace("UP", "DOWN")
                    else:
                        assert "DOWN" in bond
                        bond = bond.replace("DOWN", "UP")

            bond, Dir=bond.split("_")
            bond=bond_dic[bond]
            Dir=bonddir_dict[Dir]
            head = diction[x]
            tail = diction[y]
            tmp = emol.AddBond(head, tail, bond)
            emol.GetBondWithIdx(tmp - 1).SetBondDir(Dir)
    mol=emol.GetMol()
    mol.UpdatePropertyCache()

    s=Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True, rootedAtAtom=root, allHsExplicit=False)
    rev_diction = {diction[x]: x for x in diction.keys()}
    order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']  #
    order=list(order)
    return mol, diction, rev_diction, order, s

def nodegraph2mol(graph, c=False):
    """
    Decode molecular graph to molecules
    :param graph: the molecular graph
    :param c: whether use the c package
    :return: the molecule
    """
    cw_dict = [x for x in graph.nodes if "CW" in graph.nodes[x]["name"]]
    mol, diction, rev_diction, order, s=nodegraph2molwithnodeidx(graph, c,None,0)
    node_dict=graph.nodes
    if True:
        cmol = Chem.MolFromSmiles(s, sanitize=True)
        #if cmol is None:
        #    print(s, order, len(graph.nodes))
        #    exit
        for i,x in enumerate(order):
            catom=cmol.GetAtomWithIdx(i)
            cisatomatic=catom.GetIsAromatic()
            node_dict[rev_diction[x]]["isaromatic"]=int(cisatomatic)
        order_dic = {i: x for i, x in enumerate(order)}
        bond_order = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetIdx()] for bond in cmol.GetBonds()]
        bond_order = sorted(bond_order, key=lambda x: x[-1])
        bond_order = [[order_dic[_] for _ in x[:2]] for x in bond_order]
        bond_order = [[rev_diction[_] for _ in x] for x in bond_order]
        cw_name_dict = {x: graph.nodes[x]["name"] for x in cw_dict}
        cw_neighbors=[sort_atom_by_chiral(graph.adj[x].keys(), graph, x)[::-1] for x in cw_dict]

        for cw, cwn in zip(cw_dict, cw_neighbors):
            _bkey=[x for x in bond_order if cw in x]
            bkey=[]
            for b in _bkey:
                for _ in b:
                    if _!=cw:
                        bkey.append(_)
            bkey=bkey[:3]
            reverse_key=sorted([cwn.index(_) for _ in bkey])
            cwn=[cwn[_] for _ in reverse_key]
            key=[str(_) for _ in cwn]
            key="_".join(key)
            key=key+"_"+key
            reverse_key=[str(_) for _ in reverse_key]
            reverse_key="_".join(reverse_key)
            if reverse_key=="0_1_2" or reverse_key=="0_2_3":
                reverse_key=1
            else:
                assert (reverse_key=="1_2_3" or reverse_key=="0_1_3")
                reverse_key=-1
            bkey=[str(_) for _ in bkey]
            bkey="_".join(bkey)
            pkey=int(bkey in key)-0.5
            if (pkey*reverse_key)<0:
                if "CCW" in cw_name_dict[cw]:
                    node_dict[cw]["name"]=cw_name_dict[cw].replace("CCW", "CW")
                else:
                    node_dict[cw]["name"]=cw_name_dict[cw].replace("CW", "CCW")
        root=0
        mol, diction, rev_diction, order, s=nodegraph2molwithbondidx(graph, c, None, bond_order, root, node_dict)
    return mol

def mol2nodegraph(mol):
    """
    Encode a molecule to a molecular graph
    :param mol: the molecule to be encoded
    :return: the molecular graph
    """
    atoms_info=[(atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol()
        , atom.GetNumExplicitHs(), atom.GetChiralTag(), atom.GetFormalCharge(), atom.GetIsAromatic(), int(atom.GetNumRadicalElectrons())#, atom.GetNumRadicalElectrons()
        )
                for atom in mol.GetAtoms()]
    #print([atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()])
    bonds_info=[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType(),
                 bond.GetBondTypeAsDouble(), bond.GetBondDir(), bond.GetIdx()) for bond in mol.GetBonds()]
    G=nx.Graph()
    index=0
    for atom_info in atoms_info:
        G.add_node(atom_info[0], name="{}_{}_{}_{}_{}".format(atom_info[2], atom_info[3], formal_charge_dic[atom_info[5]], atom_info[4],  atom_info[7]), type="A")

        index=max(index, atom_info[0])
    index+=1
    #print([G.nodes[x]["isaromatic"] for x in G.nodes])
    #print([G.nodes[x]["name"] for x in G.nodes])
    special_bonds=[]
    for bond_info in bonds_info:
        type=bond_info[2]
        G.add_edge(bond_info[0], bond_info[1], name="{}_{}".format(type, bond_info[4]))
        if "UP" in "{}".format(bond_info[4]) or "DOWN" in "{}".format(bond_info[4]):
            special_bonds.append([bond_info[0], bond_info[1]])

    sub= max(nx.connected_components(G))
    G=G.subgraph(sub).copy()

    _, _, rev_diction, order, s = nodegraph2molwithnodeidx(G, False, None, 0)
    cmol=Chem.MolFromSmiles(s)
    for i,n in enumerate(order):
        catom = cmol.GetAtomWithIdx(i)
        cisatomatic = catom.GetIsAromatic()
        G.nodes[rev_diction[n]]["isaromatic"] = int(cisatomatic)

    sorted_bond=[[x[0], x[1], x[-1]] for x in bonds_info]
    sorted_bond=sorted(sorted_bond, key=lambda x: x[-1])
    for bond in special_bonds:
        x,y=bond
        old_key=[str(_) for _ in bond]
        old_key="_".join(old_key)
        _sorted_bond=sort_atom_by_chiral(bond, G, None)
        new_key=[str(_) for _ in _sorted_bond]
        new_key="_".join(new_key)
        if old_key!=new_key:
            if "UP" in G.adj[x][y]["name"]:
                G.adj[x][y]["name"]=G.adj[x][y]["name"].replace("UP", "DOWN")
            else:
                assert "DOWN" in G.adj[x][y]["name"]
                G.adj[x][y]["name"]=G.adj[x][y]["name"].replace("DOWN", "UP")

    for n in G.nodes:
        if "CW" in G.nodes[n]["name"]:
            neighbors=G.adj[n]
            _p1=[x for x in sorted_bond if n in x[:2]]
            p1=[]
            sorted_neighbors=sort_atom_by_chiral(neighbors.keys(), G, n)[::-1]
            for bond in _p1:
                for _ in bond[:2]:
                    if _!=n:
                        p1.append(_)
            p1=p1[:3]
            pidx=sorted([sorted_neighbors.index(x) for x in p1])
            sorted_neighbors=[sorted_neighbors[_] for _ in pidx]
            pidx=[str(_)  for _ in pidx]
            pidx="_".join(pidx)
            if pidx=="0_1_2" or pidx=="0_2_3":
                reverse_key=1
            else:
                assert (pidx=="1_2_3" or pidx=="0_1_3")
                reverse_key=-1
            p1=[str(x) for x in p1]
            p1="_".join(p1)
            p2=[str(_) for _ in sorted_neighbors]
            p2="_".join(p2)
            p2=p2+"_"+p2
            pkey=int(p1 in p2)-0.5
            if (pkey*reverse_key)<0:
                if "CCW" in G.nodes[n]["name"]:
                    G.nodes[n]["name"]=G.nodes[n]["name"].replace("CCW", "CW")
                else:
                    G.nodes[n]["name"] = G.nodes[n]["name"].replace("CW", "CCW")
    return G

def edge_match(x,y):
    return x["name"]==y["name"]

def node_match(x,y):
    return x["name"]==y["name"]

def parsemol(mol, grammardict, start=None, build_grammar=True):
    """
    Parse a molecule
    :param mol: a molecule
    :param grammardict: the grammar dict
    :param start: the start node of the molecule
    :return: the parse tree and the reconstructed mole
    """
    #try:
    emol=mol#nodegraph2mol(mol2nodegraph(mol))

    n=len(grammardict)
    G=mol2nodegraph(mol)
    newg=nx.Graph()
    if not start is None:
        start=[x for x in G.nodes][start]
    tree=extract_grammar(G, grammardict, newg, start, build_grammar=build_grammar)
    mol=nodegraph2mol(newg)

    s=Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)), isomericSmiles=True)
    s_old=Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(emol, isomericSmiles=True)), isomericSmiles=True)
    mol=Chem.MolFromSmiles(s)
    s=Chem.MolToSmiles(mol)

    try:
        assert s==s_old
    except:
        print(s, s_old)

    return tree, mol

def check_chemical_validity(mol, train=False):
    """
    Check the chemical validity of molecules
    :param mol: a molecule
    :param train: whether check the embedding, default=False
    :return: bool value
    """
    s=Chem.MolToSmiles(mol, isomericSmiles=True)
    m=Chem.MolFromSmiles(s)
    if m:
        if train:
            flag=AllChem.EmbedMolecule(m, useRandomCoords=False, randomSeed=10086)
            if flag>-1:
                return True
            else:
                return False
        return True
    else:
        return False

def list2graph(tree, grammar, bond_dic):
    def sn(x):
        return int(x.split("-")[1])
    idx=tree.get_next()
    newg=nx.Graph()
    inidx=-1
    stack=[]
    while (not (idx is None)):
        idx=str(int(idx))
        pdic={}
        _rewrite_node(newg, grammar, inidx, idx, pdic, bond_dic)
        xlist=sorted([x for x in pdic if "X" in x], key=sn)
        jlist=sorted([x for x in pdic if "J" in x], key=sn)
        for x in xlist[::-1]:
            stack.append(pdic[x])
        for x in jlist[::-1]:
            stack.append(pdic[x])
        if len(stack)>0:
            inidx=stack.pop()
        else:
            inidx=0
        idx=tree.get_next()
    return newg

def list2graphlist(tree, grammar, bond_dic, n):
    def sn(x):
        return int(x.split("-")[1])
    tree.re_init()
    idx=tree.get_next()
    newg=nx.Graph()
    inidx=-1
    stack=[]
    ret=[]
    retidx=[]
    retans=[]
    while (not (idx is None)):
        idx=str(int(idx))
        pdic={}
        _rewrite_node(newg, grammar, inidx, idx, pdic, bond_dic)
        xlist=sorted([x for x in pdic if "X" in x], key=sn)
        jlist=sorted([x for x in pdic if "J" in x], key=sn)
        for x in xlist[::-1]:
            stack.append(pdic[x])
        for x in jlist[::-1]:
            stack.append(pdic[x])
        if len(stack)>0:
            inidx=stack.pop()
        else:
            inidx=0
        idx=tree.get_next()
        ret.append(newg.copy())
        retidx.append(inidx)
        retans.append(idx)
    while (len(ret)<n):
        ret.append(ret[-1].copy())
    ret=ret[:n]
    return ret, retidx, retans

def list2matrix(trees, grammar, bond_dic, atoms_dic):
    graphs=[list2graph(x, grammar, bond_dic) for x in trees]
    resmol=[]
    residx=[]
    resadj=[]
    for g in graphs:
        n=[x for x in g.nodes()]
        n=sorted(n)
        residx.append(len(n))
        resadj.append(nx.adjacency_matrix(g, n,weight="myweight").toarray())
        n = [re.split("-", g.nodes[x]["name"])[0] for x in n]
        n = [atoms_dic[x] for x in n]
        resmol.extend(np.array(n))
    resmol=torch.tensor(resmol)
    resadj=torch.tensor(block_diag(*resadj))
    residx=torch.tensor(residx)
    return resmol, resadj, residx


def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target,
                                                            radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)
