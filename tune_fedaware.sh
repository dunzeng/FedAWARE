#!/bin/bash

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.1  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.3  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.5  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.7  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.9  -agnostic 1 & \

wait

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 37 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.1  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 37 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.3  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 37 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.5  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 37 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.7  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 37 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.9  -agnostic 1 & \

wait

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 73 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.1  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 73 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.3  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 73 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.5  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 73 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.7  -agnostic 1 & \

python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 0 -seed 73 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.9  -agnostic 1 & \

wait