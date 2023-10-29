import numpy as np
import torch as th

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from torch_geometric.datasets import WebKB, Actor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import networkx as nx
import dgl
import torch
import random
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)
def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']
    if name in citegraph:
        graph = dataset[0]
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

        num_class = dataset.num_classes

    if name in cograph:
        graph = dataset[0]
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.LongTensor(train_idx)
        val_idx = th.LongTensor(val_idx)
        test_idx = th.LongTensor(test_idx)
        num_class = dataset.num_classes

    # if name in hetergraph:
    #     data = torch_dataset[0]
    #     data.edge_index = to_undirected(data.edge_index)
    #     G = nx.from_edgelist(data.edge_index.transpose(0, 1).numpy().tolist())
    #     graph = dgl.from_networkx(G)
    #     graph.ndata['feat'] = data.x
    #     data.train_mask = data.train_mask.transpose(0, 1)
    #     data.val_mask = data.val_mask.transpose(0, 1)
    #     data.test_mask = data.test_mask.transpose(0, 1)
    #     train_idx = []
    #     val_idx = []
    #     test_idx = []
    #     for i in range(0, len(data.train_mask)):
    #         train_idx.append(torch.where(data.train_mask[i])[0])
    #         val_idx.append(torch.where(data.val_mask[i])[0])
    #         test_idx.append(torch.where(data.test_mask[i])[0])
    #
    #     labels = data.y
    #     graph.ndata['label'] = labels
    #     num_class = int(max(labels)) + 1


    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx
