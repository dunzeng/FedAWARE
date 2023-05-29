#!/bin/bash

# python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 1998 -partition dirichlet -dir 0.1  -alpha 0.3  & \

# python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 37 -partition dirichlet -dir 0.1  -alpha 0.3  & \

# python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 73 -partition dirichlet -dir 0.1  -alpha 0.3  & \


python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998 -partition pathological -dataset mnist -alpha 0.3 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37 -partition pathological -dataset mnist -alpha 0.3 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73 -partition pathological -dataset mnist -alpha 0.3 & \

wait

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 1998 -partition dirichlet -dir 0.1  -alpha 0.3 -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 37 -partition dirichlet -dir 0.1  -alpha 0.3 -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -lr 0.01 -glr 1 -dataset mnist -dseed 2023 -seed 73 -partition dirichlet -dir 0.1  -alpha 0.3 -agnostic 1 & \

wait