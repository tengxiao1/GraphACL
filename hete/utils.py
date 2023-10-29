### Random tools useful for saveing stuff and manipulating pickle/numpy objects
import numpy as np
import networkx as nx
import dgl
import torch
import logging
from functools import lru_cache
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.datasets import WebKB, Actor
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from hete.dataset import WikipediaNetwork


class DataSplit:

    def __init__(self, dataset, train_ind, val_ind, test_ind, shuffle=True):
        self.train_indices = train_ind
        self.val_indices = val_ind
        self.test_indices = test_ind
        self.dataset = dataset

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


def read_real_datasets(datasets):
    if datasets in ["cornell", "texas", "wisconsin"]:
        torch_dataset = WebKB(root=f'../datasets/', name=datasets,
                          transform=T.NormalizeFeatures())
    elif datasets in ['squirrel', 'chameleon']:
        torch_dataset = WikipediaNetwork(root=f'../datasets/', name=datasets, geom_gcn_preprocess=True)
    elif datasets in ['crocodile']:
        torch_dataset = WikipediaNetwork(root=f'../datasets/', name=datasets, geom_gcn_preprocess=False)
    elif datasets == 'film':
        torch_dataset = Actor(root=f'../datasets/film/', transform=T.NormalizeFeatures())
    data = torch_dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    G = nx.from_edgelist(data.edge_index.transpose(0, 1).numpy().tolist())
    g = dgl.from_networkx(G)
    g.ndata['attr'] = data.x
    data.train_mask = data.train_mask.transpose(0, 1)
    data.val_mask = data.val_mask.transpose(0, 1)
    data.test_mask = data.test_mask.transpose(0, 1)
    split_list = []
    for i in range(0, len(data.train_mask)):
        split_list.append({'train_idx': torch.where(data.train_mask[i])[0],
                           'valid_idx': torch.where(data.val_mask[i])[0],
                           'test_idx': torch.where(data.test_mask[i])[0]})
    labels = data.y

    return g, labels, split_list


