#!/bin/bash

python main.py -num_clients 100 -com_round 100 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 37 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar100 -alpha 0.5 -optim adam & \

python baselines/fedavg.py -num_clients 100 -com_round 50 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100 -optim adam & \

python baselines/fedavg.py -num_clients 100 -com_round 50 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100 -optim adam & \

python baselines/fedavg.py -num_clients 100 -com_round 50 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 10 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100  -optim adam

python baselines/fedavg.py -num_clients 100 -com_round 100 -sample_ratio 0.1 -batch_size 10 -epochs 1 -lr 0.1 -glr 1 -dseed 37 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar100 -alpha 0.5 -optim adam

python baselines/fedavg.py -num_clients 100 -com_round 100 -sample_ratio 0.1 -batch_size 128 -epochs 5 -lr 0.1 -glr 1 -dseed 37 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar100 -alpha 0.5 -optim adam