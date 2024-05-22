#!/bin/bash

# python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.02 -glr 1 -dseed 2023 -seed 42 -partition dirichlet -dir 0.3 -dataset cifar10 -alpha 0.5 & \

# python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.02 -glr 1 -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10  & \

# python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.02 -glr 1 -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 -mu 0.01 & \

# python baselines/scaffold.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.02 -glr 1 -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 & \

# python baselines/fednova.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.02 -glr 1 -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 & \

# python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.02 -glr 0.001 -dseed 2023 -seed 42  -partition dirichlet -dir 0.3 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

# wait

python main.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 42 -partition pathological -dataset cifar10 -alpha 0.5 -agnostic 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 42  -partition pathological -dataset cifar10  & \

python baselines/fedprox.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 42  -partition pathological -dataset cifar10 -mu 0.01 & \

python baselines/scaffold.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 42  -partition pathological -dataset cifar10 & \

python baselines/fednova.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 42  -partition pathological -dataset cifar10 & \

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 42  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

wait

python fedaware+fedyogi.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 37 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -beta1 0.9 -beta2 0.99 -tau 0.0001 

python fedaware+fedyogi.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 42  -partition dirichlet -dir 0.1 -dataset cifar10 -beta1 0.9 -beta2 0.99 -tau 0.0001 