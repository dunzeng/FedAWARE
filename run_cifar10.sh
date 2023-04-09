#!/bin/bash

python main.py -num_clients 100 -com_round 250 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 42 -alpha 0.5 -partition dir -dir 0.3 -dataset cifar10 & \

python baselines/fedavg.py -num_clients 100 -com_round 250 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 42  -partition dir -dir 0.3 -dataset cifar10  & \

python baselines/fedprox.py -num_clients 100 -com_round 250 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 42  -partition dir -dir 0.3 -dataset cifar10 -mu 0.1 & \

python baselines/scaffold.py -num_clients 100 -com_round 250 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 42  -partition dir -dir 0.3 -dataset cifar10 & \

python baselines/fednova.py -num_clients 100 -com_round 250 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 42  -partition dir -dir 0.3 -dataset cifar10 & \

wait
