#!/bin/bash

python main.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998 -partition pathological -dataset mnist -alpha 0.5 -startup 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 1998  -partition pathological -dataset mnist -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

wait