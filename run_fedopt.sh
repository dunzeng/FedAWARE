#!bin/bash

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 0  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.1 -dataset mnist -dseed 2023 -seed 0  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dataset mnist -dseed 2023 -seed 0  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.001 -dataset mnist -dseed 2023 -seed 0  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

wait