# distutils: language=c++
cimport cython
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
#from libcpp.algorithm cimport find, copy
from cython.operator cimport dereference, preincrement
from cython.parallel import prange, parallel
from libc.stdio cimport printf
import numpy as np
cimport numpy

cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, Func](Iter first, Iter last, Func pred)

cdef int stoi(string x) nogil:
    cdef int out=0
    cdef int n=x.length()
    cdef int i=0
    cdef char st=b"0"
    while i<n:
        out*=10
        out+=x[i]-st
        i+=1
    return out

cdef struct neighbor:
    int neigh
    string bond
    string name
cdef map[string, int] atoms
cdef class Graph:
    cdef vector[int] _nodes
    cdef map[int, map[string, string]] nodes
    cdef map[int, map[int, map[string, string]]] adj
    cdef map[int, vector[neighbor]] _xlist
    cdef map[int, vector[neighbor]] _lhs
    cdef map[int, vector[int]] _mask


    @property
    def size(self):
        return self._nodes.size()

    @property
    def get_max(self):
        return max(self._nodes)

    @property
    def nodes(self):
        _ret=dict(self.nodes)
        ret={}
        for x in _ret:
            ret[x]={}
            for y in _ret[x]:
                ret[x][y.decode("utf-8")]=_ret[x][y].decode("utf-8")
        return ret

    @property
    def adj(self):
        _ret=dict(self.adj)
        ret={}
        for x in _ret:
            ret[x]={}
            for y in _ret[x]:
                ret[x][y]={}
                for z in _ret[x][y]:
                    ret[x][y][z.decode("utf-8")]=_ret[x][y][z].decode("utf-8")
        return ret

    def set_atoms(self, patoms):
        for x in patoms:
            atoms[x.encode("utf-8")]=patoms[x]


    def lhs(self, index):
        cdef vector[neighbor] vec=self._lhs.at(index)
        n=vec.size()
        ret=[]
        for i in range(n):
            ret.append(self.neighbor2list(vec[i]))
        return ret

    def xlist(self, index):
        cdef vector[neighbor] vec=self._xlist.at(index)
        n=vec.size()
        ret=[]
        for i in range(n):
            ret.append(self.neighbor2list(vec[i]))
        return ret

    def get_mask(self, index):
        return dereference(self._mask.find(index)).second

    cdef neighbor2list(self, neighbor neigbor):
        return [neigbor.neigh, neigbor.bond.decode("utf-8"), neigbor.name.decode("utf-8")]

    def add_mask(self, index, mask):
        for x in mask:
            self._mask.at(index).push_back(x)

    def add_lhs(self, index, lhs):
        cdef neigh=neighbor(lhs[0], lhs[1].encode("utf-8"), lhs[2].encode("utf-8"))
        self._lhs.at(index).push_back(neigh)

    def add_xlist(self, index, xlist):
        cdef neigh=neighbor(xlist[0], xlist[1].encode("utf-8"), xlist[2].encode("utf-8"))
        self._xlist.at(index).push_back(neigh)

    def modify_xlist(self, idx, target_v, new_v, bond):
        cdef int n=self._xlist.at(idx).size()
        cdef int i=0
        cdef int nv=new_v
        cdef int ov=target_v
        cdef int Idx=idx
        cdef string nb=bond.encode("utf-8")

        while i<n:
            if self._xlist.at(Idx).at(i).neigh==ov:
                self._xlist.at(Idx).at(i).neigh=nv
                self._xlist.at(Idx).at(i).bond=nb
            i+=1

    cpdef modify_lhs(self, idx, target_v, new_v, bond):
        cdef int Idx=idx
        cdef int n=self._lhs.at(Idx).size()
        cdef int i=0
        cdef int nv=new_v
        cdef int ov=target_v
        cdef string nb=bond.encode("utf-8")
        while i<n:
            if self._lhs.at(Idx).at(i).neigh==ov:
                #print(tmp[i].neigh, ov,nv)
                self._lhs.at(Idx).at(i).neigh=nv
                self._lhs.at(Idx).at(i).bond=nb
                break
            i+=1

    def add_node(self,idx,lhs, xlist, **args):
        if self.nodes.find(idx)==self.nodes.end():
            self.nodes[idx]=map[string, string]()
            self._nodes.push_back(idx)
            self.adj[idx]=map[int, map[string, string]]()
            self._lhs[idx]=vector[neighbor]()
            self._xlist[idx]=vector[neighbor]()
            self._mask[idx]=vector[int]()
        for x in lhs:
            self.add_lhs(idx, x)
        for x in xlist:
            self.add_xlist(idx, x)
        for x in args:
            #print(x, args[x])
            self.nodes[idx][x.encode('utf-8')]=args[x].encode('utf-8')

    def add_edge(self, start, end, **args):
        self.add_node(start, [], [])
        self.add_node(end, [], [])
        if self.adj[start].find(end)==self.adj[start].end():
            self.adj[start][end]=map[string, string]()

        if self.adj[end].find(start)==self.adj[end].end():
            self.adj[end][start]=map[string, string]()
        for x in args:
            self.adj[start][end][x.encode('utf-8')]=args[x].encode('utf-8')
            self.adj[end][start][x.encode('utf-8')]=args[x].encode('utf-8')

    def remove_node(self, int idx):
        cdef int size=self._nodes.size()
        cdef i=0
        cdef map[int, map[string, string]].iterator iter
        cdef int Idx=idx
        cdef map[int, vector[int]].iterator miter
        self.nodes.erase(Idx)
        miter=self._mask.find(Idx)
        if miter!=self._mask.end():
            self._mask.erase(miter)
        self._lhs.erase(self._lhs.find(Idx))
        self._xlist.erase(self._xlist.find(Idx))
        while i<size:
            iter=self.adj[self._nodes[i]].find(Idx)
            if iter!=self.adj[self._nodes[i]].end():
                self.adj[self._nodes[i]].erase(iter)
            i+=1
        self._nodes.erase(find(self._nodes.begin(), self._nodes.end(),Idx))
        self.adj.erase(Idx)


    def copy(self):
        tgraph=Graph()
        cdef int size=self._nodes.size()
        cdef i=0
        cdef map[int, map[string, string]].iterator iter
        while i<size:
            wargs=self.nodes[self._nodes[i]]
            wargs={x.decode("utf-8"):wargs[x].decode("utf-8") for x in wargs}
            tgraph.add_node(self._nodes[i],[],[], **wargs)
            lhs=self.lhs(self._nodes[i])
            xlist=self.xlist(self._nodes[i])
            mask=self.get_mask(self._nodes[i])
            for x in lhs:
                tgraph.add_lhs(self._nodes[i], x)
            for x in xlist:
                tgraph.add_xlist(self._nodes[i],x)
            tgraph.add_mask(self._nodes[i], mask)

            iter=self.adj[self._nodes[i]].begin()
            while iter!=self.adj[self._nodes[i]].end():
                wargs=dereference(iter).second
                wargs={x.decode("utf-8"):wargs[x].decode("utf-8") for x in wargs}
                tgraph.add_edge(self._nodes[i], dereference(iter).first, **wargs)
                preincrement(iter)
            i+=1
        return tgraph

    cdef void get_mol_adj(self, int idx, int maxl, vector[int]& rnodes, vector[int]& adjs) nogil:
        cdef vector[int] tnodes
        cdef map[int, int]cmap
        cdef int n
        cdef int i
        cdef map[int, map[string, string]].iterator iter

        cdef string tname
        cdef int wv
        cdef int tidx
        n=self._nodes.size()
        tnodes.push_back(idx)
        i=0
        while i<n:
            if self._nodes.at(i)==idx:
                i+=1
                continue
            else:
                tnodes.push_back(self._nodes.at(i))
            i+=1

        i=0

        while i<n:
            cmap[tnodes[i]]=i
            tname=dereference(self.nodes[tnodes[i]].find(b"name")).second
            rnodes[i]=atoms[tname.substr(0, tname.find(b"-"))]
            i+=1
        i=0

        for i in range(n):
            tidx=tnodes.at(i)
            iter=self.adj[tidx].begin()
            while iter!=self.adj[tidx].end():
                wv=stoi(dereference(dereference(iter).second.find(b"myweight")).second)
                adjs[cmap[tidx]*maxl+cmap[dereference(iter).first]]=wv
                preincrement(iter)

        return

    cpdef get_mol_adjs(self, Graphs, index):
        cdef int maxl
        maxl=max([g.size for g in Graphs])
        cdef int n=len(Graphs)
        cdef vector[vector[int]] Ns=vector[vector[int]](n)
        cdef vector[vector[int]] Adjs=vector[vector[int]](n)
        cdef int i=0
        cdef int Idx=0
        cdef int Maxl=maxl
        cdef numpy.ndarray tg=np.array(Graphs, dtype=Graph)
        #cdef numpy.ndarray Ns=np.full([n, maxl], "r")
        #cdef numpy.ndarray Adjs=np.zeros([n, maxl, maxl])
        cdef void**ptr=<void**>(tg.data)
        cdef vector[int]ptidx
        for x in index:
            ptidx.push_back(x)


        with nogil, parallel():
            for i in prange(n):
                Ns[i].assign(maxl, 0)
                Adjs[i].assign(maxl*maxl, 0)
                (<Graph>ptr[i]).get_mol_adj(ptidx[i], Maxl, Ns[i], Adjs[i])
        return np.array(Ns), np.array(Adjs).reshape(n, maxl, maxl)

def testfunc():
    cdef int Idx=0
    cdef int i
    with nogil, parallel():
        for i in prange(10):
            Idx+=1
    print(Idx)
