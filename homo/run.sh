python main.py --dataname cora --epochs 50 --lr1 5e-4 --lr2 1e-3 --wd1 1e-6 --wd2 1e-5 --n_layers 2 --hid_dim 2048  --temp 1.0 --moving_average_decay 0.97 --num_MLP 1
python main.py --dataname citeseer --epochs 15 --lr1 1e-3 --lr2 1e-2 --wd1 1e-4 --wd2 1e-2  --n_layers 1 --hid_dim 2048  --temp 0.99 --moving_average_decay 0.95 --num_MLP 1
python main.py --dataname pubmed --epochs 60 --lr1 5e-4 --lr2 1e-2 --wd1 0 --wd2 1e-4 --n_layers 2 --hid_dim 1024  --temp 0.75 --moving_average_decay 0.99 --num_MLP 1
python main.py --dataname comp --epochs 50 --lr1 5e-4 --lr2 1e-2 --wd1 0 --wd2 5e-4 --n_layers 1 --hid_dim 2048  --temp 0.99 --moving_average_decay 0.99 --num_MLP 1
python main.py --dataname photo --epochs 70 --lr1 5e-4 --lr2 1e-2 --wd1 0 --wd2 1e-3 --n_layers 1 --hid_dim 2048  --temp 0.75 --moving_average_decay 0.90 --num_MLP 1

