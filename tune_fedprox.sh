#!/bin/bash


python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -mu 0.01 -agnostic 1  & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -mu 0.1 -agnostic 1  & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -mu 1 -agnostic 1  & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -mu 10 -agnostic 1  & \

wait