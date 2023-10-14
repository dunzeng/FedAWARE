#!/bin/bash


python baselines/feddyn.py -num_clients 100 -com_round 50 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -alpha_dyn 0.1  & \

python baselines/feddyn.py -num_clients 100 -com_round 50 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -alpha_dyn 0.01  & \

python baselines/feddyn.py -num_clients 100 -com_round 50 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -alpha_dyn 0.001  & \


wait