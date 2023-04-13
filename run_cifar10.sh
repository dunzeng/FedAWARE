#!/bin/bash

python main.py -num_clients 100 -com_round 3 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 42 -partition dirichlet -dir 0.3 -dataset cifar10 -alpha 0.5 & \

python baselines/fedavg.py -num_clients 100 -com_round 3 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10  & \

python baselines/fedprox.py -num_clients 100 -com_round 3 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 -mu 0.01 & \

python baselines/scaffold.py -num_clients 100 -com_round 3 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 & \

python baselines/fednova.py -num_clients 100 -com_round 3 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 & \

python baselines/fednova.py -num_clients 100 -com_round 3 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.001 -dataset mnist -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

wait
