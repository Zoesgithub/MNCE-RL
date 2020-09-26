import torch
import torch.nn as nn
from torch.nn import Parameter
from loguru import logger
import math

def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Gumbel(object):
    def __init__(self):
        self.uniform=torch.distributions.uniform.Uniform(0, 1)
    def sample(self, n):
        mu=self.uniform.sample(n)
        return torch.log(mu.cuda())

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature):
        super(GumbelSoftmax, self).__init__()
        self.temperature=temperature
        self.gumbel=Gumbel()
        self.softmax=nn.Softmax(-1)
        self.iters=0

    def forward(self, inp):
        x=self.softmax(inp)
        x=torch.log(x)
        g=self.gumbel.sample(x.shape)
        x=x+g*self.temperature
        self.temperature*=0.9999
        logger.info("TEM {}".format(self.temperature))


        return self.softmax(x)

class stemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, n):
        idx=inp.argmax(-1)
        idx=nn.functional.one_hot(idx, n).float()
        return idx
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def l2norm(v, eps=1e-12):
    return v/(v.norm()+eps)

class SN(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1 ):
        """
        spectral normalization
        :param module: the module to use SN
        :param name: name to use SN
        """
        super(SN, self).__init__()
        self.module=module
        self.name=name
        self.power_iterations=power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u=getattr(self.module, self.name+'_u')
        v=getattr(self.module, self.name+'_v')
        w=getattr(self.module, self.name+'_bar')
        height=w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data=l2norm(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data=l2norm(torch.mv(w.view(height, -1).data, v.data))
        sigma=u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w/sigma.expand_as(w))

    def _made_params(self):
        try:
            u=getattr(self.module, self.name+'_u')
            v=getattr(self.module, self.name+'_v')
            w=getattr(self.module, self.name+'_bar')
            return True
        except AttributeError:
            return False
    def _make_params(self):
        w=getattr(self.module, self.name)
        height=w.data.shape[0]
        width=w.view(height, -1).data.shape[1]

        u=Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v=Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)

        u.data=l2norm(u.data)
        v.data=l2norm(v.data)
        w_bar=Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name+'_u', u)
        self.module.register_parameter(self.name+'_v', v)
        self.module.register_parameter(self.name+'_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class gcn_layer(nn.Module):
    def __init__(self, in_features, out_features, num_head, bias=True, sn=False, bn=False):
        """
        One layer of GCN
        https://arxiv.org/abs/1609.02907
        """
        super(gcn_layer, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.num_head=num_head

        self.weight=Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias=Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        self.adj_linear=nn.Linear(self.out_features*2, 16)
        self.adj_out_linear=nn.Linear(16+self.num_head, self.num_head)
        self.bn=bn
        if bn:
            self.mol_bn1=nn.BatchNorm1d(self.num_head*self.out_features)
            self.mol_bn2=nn.BatchNorm1d(self.out_features)
            self.adj_bn1=nn.BatchNorm1d(16)
            self.adj_bn2=nn.BatchNorm1d(self.num_head)
        if sn:
            self.out_linear=SN(nn.Linear(self.out_features*self.num_head, self.out_features))
        else:
            self.out_linear=(nn.Linear(self.out_features*self.num_head, self.out_features, bias=True))
        self.reset_parameters()

    def reset_parameters(self):#intialize weight and bis
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inp):
        """
        The func to forward
        :param inp: data
        :return: results in the format of [mole, adj]
        """
        mol, adj, mask, adjmask=inp
        a0, a1, a2, a3=adj.shape #batch*channel*sha1*sha2
        m1, m2, m3=mol.shape #batch*sha1*emb
        assert a2==a3
        #print(adj.shape, mol.shape)
        support=torch.mm(mol.reshape(m1*m2, m3), self.weight).reshape(m1, m2, self.out_features)
        output=torch.bmm(adj.reshape(a0, a1*a2, a3), support).reshape(a0,a1,a2, self.out_features)+support.unsqueeze(1)
        #print(output.sum(), mask.sum())
        assert ((output*(1-mask.unsqueeze(1))).sum()==0)
        if self.bias is not None:
            output=output+self.bias
        output=self.tanh(output.permute(0,2,3,1).reshape(m1, m2, m3*self.num_head))
        if self.bn:
            output=self.mol_bn1(output.reshape(-1, m3*self.num_head)).reshape(m1, m2, m3*self.num_head)
        output=output*mask
        output=self.out_linear(output)
        output=self.tanh(output)
        if self.bn:
            output=self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        output=output*mask

        tadj=torch.cat([output.unsqueeze(1)*torch.ones([1, a2, 1, 1]).cuda(output.device), 
            output.unsqueeze(2)*torch.ones([1, 1, a3, 1]).cuda(output.device)], 3)
        tadj=self.relu(self.adj_linear(tadj))
        if self.bn:
            tadj=self.adj_bn1(tadj.reshape(-1, 16)).reshape(a0, a2, a3, 16)
        tadj=torch.cat([tadj, adj.permute(0,2,3,1)], 3)
        _adj=adj
        adj=self.relu(self.adj_out_linear(tadj))
        if self.bn:
            adj=self.adj_bn2(adj.reshape(-1, self.num_head)).reshape(a0, a2, a3, self.num_head)
        adj=adj*adjmask
        adj=adj.permute(0,3, 1,2)+_adj
        return [output, adj, mask, adjmask]
