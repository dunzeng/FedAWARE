#!/bin/bash


# python baselines/fedavg.py -num_clients 100 -com_round 300 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 20230409 -partition dir -dir 0.3 -dataset cifar10 -agnostic 1 & \
#sleep 3s
python baselines/fedavg.py -num_clients 100 -com_round 300 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 20230410 -partition dir -dir 0.3 -dataset cifar10 -agnostic 0 & \
sleep 3s
# python baselines/fedavg.py -num_clients 100 -com_round 300 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 20230409 -partition path -dataset cifar10 -agnostic 1 & \
#sleep 3s
python baselines/fedavg.py -num_clients 100 -com_round 300 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 0 -seed 20230410 -partition path -dataset cifar10 -agnostic 0 & \

wait