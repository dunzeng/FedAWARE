#!/bin/bash

python main.py -num_clients 100 -com_round 3 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.05 -glr 1 -dataset mnist -dseed 4399 -seed 42 -alpha 1 -partition path -dataset cifar10 & \

python main.py -num_clients 100 -com_round 200 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.05 -glr 1 -dataset mnist -dseed 4399 -seed 42 -alpha 0.5 -partition path & \

python main.py -num_clients 100 -com_round 200 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.05 -glr 1 -dataset mnist -dseed 4399 -seed 42 -alpha 0.1 -partition path & \ 

wait
