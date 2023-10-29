# Official implementation of GraphACL

**This repository is an official PyTorch implementation of the paper**  ["Simple and Asymmetric Graph Contrastive Learning without Augmentations"]() (NeurIPS 2023)


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


##  Reference

If you find this repo to be useful, please cite our paper:

```bibtex
@inproceedings{
  Xiao2023,
  title={GraphACL: Graph Asymmetric Contrastive Learning without Augmentations},
  author={Xiao, Teng; Zhu, Huaisheng; Chen, Zhengyu; Wang, Suhang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems ({NeurIPS})},
  year={2023}
}
```
