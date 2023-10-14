#!bin/bash

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 0  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -agnostic 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.1 -dataset mnist -dseed 0 -seed 0  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -agnostic 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dataset mnist -dseed 0 -seed 0  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -agnostic 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.001 -dataset mnist -dseed 0 -seed 0  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -agnostic 1 & \

wait
