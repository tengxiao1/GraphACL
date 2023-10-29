import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
import copy


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_ln=False):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.use_ln = use_ln
        self.lns = nn.ModuleList()
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            for i in range(n_layers - 1):
                self.lns.append(nn.BatchNorm1d(hid_dim))
                # self.lns.append(nn.LayerNorm(hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            if not self.use_ln:
                x = F.relu(self.convs[i](graph, x))
            else:
                x = F.relu(self.lns[i](self.convs[i](graph, x)))

        x = self.convs[-1](graph, x)

        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(Predictor, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        h = self.linears[self.num_layers - 1](h)
        return h

def udf_u_add_log_e(edges):
    return {'m': torch.log(edges.dst['neg_sim'] + edges.data['sim'])}

class Model(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, temp, moving_average_decay=1.0, num_MLP=1):
        super(Model, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, out_dim, num_layers)
        self.encoder_target = copy.deepcopy(self.encoder)
        set_requires_grad(self.encoder_target, False)

        self.temp = temp
        self.out_dim = out_dim
        self.target_ema_updater = EMA(moving_average_decay)
        self.num_MLP = num_MLP
        
        self.projector = Predictor(hid_dim, hid_dim, num_MLP)


    def get_embedding(self, graph, feat):
        h = self.encoder(graph, feat)
        return h.detach()

    def pos_score(self, graph, v, u):
        graph.ndata['q'] = F.normalize(self.projector(v))
        graph.ndata['u'] = F.normalize(u, dim=-1)
        graph.apply_edges(fn.u_mul_v('u', 'q', 'sim'))
        graph.edata['sim'] = graph.edata['sim'].sum(1) / self.temp
        graph.update_all(fn.copy_e('sim', 'm'), fn.mean('m', 'pos'))
        pos_score = graph.ndata['pos']

        return pos_score, graph

    def neg_score(self, h, graph, rff_dim=None):
        z = F.normalize(h, dim=-1)
        graph.edata['sim'] = torch.exp(graph.edata['sim'])
        neg_sim = torch.exp(torch.mm(z, z.t()) / self.temp)
        neg_score = neg_sim.sum(1)
        graph.ndata['neg_sim'] = neg_score
        graph.update_all(udf_u_add_log_e, fn.mean('m', 'neg'))
        neg_score = graph.ndata['neg']
        return neg_score

    def update_moving_average(self):
        # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.encoder_target is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.encoder_target, self.encoder)

    def forward(self, graph, feat):
        v = self.encoder(graph, feat)
        u = self.encoder_target(graph, feat)
        pos_score, graph = self.pos_score(graph, v, u)
        neg_score = self.neg_score(v, graph)
        loss = (- pos_score + neg_score).mean()

        return loss
