# Official implementation of GraphACL

**This repository is an official PyTorch implementation of**  ["Simple and Asymmetric Graph Contrastive Learning without Augmentations"](https://arxiv.org/abs/2310.18884) (NeurIPS 2023)


##  Abstract

Graph Contrastive Learning (GCL) has shown superior performance in representation learning in graph-structured data. Despite their success, most existing GCL methods rely on prefabricated graph augmentation and homophily assumptions. Thus, they fail to generalize well to heterophilic graphs where connected nodes may have different class labels and dissimilar features. In this paper, we study the problem of conducting contrastive learning on homophilic and heterophilic graphs. We find that we can achieve promising performance simply by considering an asymmetric view of the neighboring nodes. The resulting simple algorithm, Asymmetric Contrastive Learning for Graphs (GraphACL), is easy to implement and does not rely on graph augmentations and homophily assumptions. We provide theoretical and empirical evidence that GraphACL can capture one-hop local neighborhood information and two-hop monophily similarity, which are both important for modeling heterophilic graphs. Experimental results show that the simple GraphACL significantly outperforms state-of-the-art graph contrastive learning and self-supervised learning methods on homophilic and heterophilic graphs.

##  Requirements

- CUDA Version: 12.2  
- dgl==1.1.2+cu117
- matplotlib==3.7.3
- networkx==3.1
- numpy==1.24.3
- seaborn==0.13.0
- torch==1.13.0
- torch_geometric==2.3.1
- tqdm==4.66.1

## Usage

- To replicate the  results on homophilous graphs, run the following script
```sh
sh homo/run.sh
```
- To replicate the results on heterophilous graphs, run the following script
```sh
sh hete/run.sh
```
##  Questions

If you have any questions related to the code or the paper, feel free to email us (tengxiao01@gmail.com). Please try to specify the problem with details so we can help you better and quicker.

##  Reference

If you find this repo to be useful, please cite our paper:

```bibtex
@inproceedings{
  Xiao2023GraphACL,
  title={Simple and Asymmetric Graph Contrastive Learning without Augmentations},
  author={Xiao, Teng; Zhu, Huaisheng; Chen, Zhengyu; Wang, Suhang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems ({NeurIPS})},
  year={2023}
}
```
