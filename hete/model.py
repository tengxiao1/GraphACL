from hete.layers import  MLP_generator, PairNorm
import torch
import torch.nn as nn
from dgl.nn import  GraphConv
import random
import copy
from torch.nn.functional import normalize
import numpy as np


def contrastive_loss(projected_emd, v_emd, sampled_embeddings_u, sampled_embeddings_neg_v, tau, lambda_loss):
    pos = torch.exp(torch.bmm(projected_emd, sampled_embeddings_u.transpose(-1, -2)).squeeze()/tau)
    neg_score = torch.log(pos + torch.sum(torch.exp(torch.bmm(v_emd, sampled_embeddings_neg_v.transpose(-1, -2)).squeeze()/tau), dim=1).unsqueeze(-1))
    neg_score = torch.sum(neg_score, dim=1)
    pos_socre = torch.sum(torch.log(pos), dim=1)
    total_loss = torch.sum(lambda_loss * neg_score - pos_socre)
    total_loss = total_loss/sampled_embeddings_u.shape[0]/sampled_embeddings_u.shape[1]
    return total_loss

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


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num, sample_size, tau, norm_mode="PN-SCS", norm_scale=20, lambda_loss=1,moving_average_decay=0.0, num_MLP=3):
        super(Model, self).__init__()
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss = lambda_loss
        self.tau = tau
        # GNN Encoder
        self.graphconv1 = GraphConv(in_dim, hidden_dim)
        self.graphconv2 = GraphConv(hidden_dim, hidden_dim)
        self.target_graphconv1 = copy.deepcopy(self.graphconv1)
        self.target_graphconv2 = copy.deepcopy(self.graphconv2)
        set_requires_grad(self.target_graphconv1, False)
        set_requires_grad(self.target_graphconv2, False)

        self.target_ema_updater = EMA(moving_average_decay)
        self.num_MLP = num_MLP
        self.projector = MLP_generator(hidden_dim, hidden_dim, num_MLP)

        self.in_dim = in_dim
        self.sample_size = sample_size

    def update_moving_average(self):
        # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_graphconv1 or self.target_graphconv2 is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_graphconv1, self.graphconv1)
        update_moving_average(self.target_ema_updater, self.target_graphconv2, self.graphconv2)

    def forward_encoder(self, g, h):
        v_emd = self.graphconv2(g, self.graphconv1(g, h))
        u_emd = normalize(self.target_graphconv2(g, self.target_graphconv1(g, h)), p=2, dim=-1)
        projected_emd = normalize(self.projector(v_emd), p=2, dim=-1)

        return v_emd, u_emd, projected_emd

    def get_emb(self, neighbor_indexes, gt_embeddings):
        sampled_embeddings = []
        if len(neighbor_indexes) < self.sample_size:
            sample_indexes = neighbor_indexes
            sample_indexes += np.random.choice(neighbor_indexes, self.sample_size - len(sample_indexes)).tolist()
        else:
            sample_indexes = random.sample(neighbor_indexes, self.sample_size)

        for index in sample_indexes:
            sampled_embeddings.append(gt_embeddings[index])

        return torch.stack(sampled_embeddings)

    # Sample neighbors from neighbor set, if the length of neighbor set less than sample size, then do the padding.
    def sample_neighbors(self, neighbor_dict, u_emd,v_emd):
        sampled_embeddings_list = []
        sampled_embeddings_neg_list = []
        for index, embedding in enumerate(u_emd):
            neighbor_indexes = neighbor_dict[index]
            sampled_embeddings = self.get_emb(neighbor_indexes, u_emd)
            sampled_embeddings_list.append(sampled_embeddings)
            sampled_neg_embeddings = self.get_emb(range(0, len(neighbor_dict)), v_emd)
            sampled_embeddings_neg_list.append(sampled_neg_embeddings)

        return torch.stack(sampled_embeddings_list), torch.stack(sampled_embeddings_neg_list)

    def reconstruction_neighbors(self, projected_emd, u_emd,v_emd, neighbor_dict):
        sampled_embeddings_u, sampled_embeddings_neg_v = self.sample_neighbors(neighbor_dict, u_emd, normalize(v_emd, p=2, dim=-1))
        projected_emd = projected_emd.unsqueeze(1)
        neighbor_recons_loss = contrastive_loss(projected_emd, normalize(v_emd, p=2, dim=-1).unsqueeze(1), sampled_embeddings_u, sampled_embeddings_neg_v, self.tau, self.lambda_loss)
        return neighbor_recons_loss


    def neighbor_decoder(self, v_emd, u_emd, projected_emd, neighbor_dict):
        neighbor_recons_loss = self.reconstruction_neighbors(projected_emd, u_emd, v_emd, neighbor_dict)
        return neighbor_recons_loss
    

    def forward(self, g, h, neighbor_dict, device):
        v_emd, u_emd, projected_emd = self.forward_encoder(g, h)
        loss = self.neighbor_decoder(v_emd, u_emd, projected_emd,neighbor_dict)
        return loss, v_emd
