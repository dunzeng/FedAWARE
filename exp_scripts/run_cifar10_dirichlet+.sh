#!/bin/bash

python fedaware.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar100 -alpha 0.5 -agnostic 1 & \

python fedaware.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37 -partition dirichlet -dir 0.1 -dataset cifar100 -alpha 0.5 -agnostic 1 & \

python fedaware.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73 -partition dirichlet -dir 0.1 -dataset cifar100 -alpha 0.5 -agnostic 1 & \

wait

python baselines/fedavg.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

wait

python baselines/fedavgm.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100 -fedm_beta 0.7  -agnostic 1 & \

python baselines/fedavgm.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar100 -fedm_beta 0.7  -agnostic 1 & \

python baselines/fedavgm.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar100 -fedm_beta 0.7  -agnostic 1 & \

wait

python baselines/fedprox.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar100 -mu 0.01  -agnostic 1 & \

python baselines/fedprox.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100 -mu 0.01  -agnostic 1 & \

python baselines/fedprox.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar100 -mu 0.01  -agnostic 1 & \

wait

python baselines/scaffold.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

python baselines/scaffold.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

python baselines/scaffold.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

wait

python baselines/fednova.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

python baselines/fednova.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

python baselines/fednova.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar100  -agnostic 1 & \

wait

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar100 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001  -agnostic 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001  -agnostic 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar100 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001  -agnostic 1 & \

wait

python baselines/feddyn.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar100 -alpha_dyn 0.01  -agnostic 1 & \

python baselines/feddyn.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar100 -alpha_dyn 0.01  -agnostic 1 & \

python baselines/feddyn.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar100 -alpha_dyn 0.01  -agnostic 1 & \

wait

python baselines/fedams.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar100 -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8  -agnostic 1 & \

python baselines/fedams.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar100 -dseed 2023 -seed 1998 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8  -agnostic 1 & \

python baselines/fedams.py -num_clients 100 -com_round 1000 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar100 -dseed 2023 -seed 73 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8  -agnostic 1 & \

wait