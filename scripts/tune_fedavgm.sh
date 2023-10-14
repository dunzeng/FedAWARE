#!/bin/bash


python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.7 -agnostic 1 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.9 -agnostic 1 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.97 -agnostic 1 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.997 -agnostic 1 & \

wait