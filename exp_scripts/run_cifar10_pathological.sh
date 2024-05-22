#!/bin/bash


python fedaware.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998 -partition pathological -dataset cifar10 -alpha 0.5 -startup 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition pathological -dataset cifar10 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 2023 -seed 1998  -partition pathological -dataset cifar10 -fedm_beta 0.97 & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition pathological -dataset cifar10 -mu 0.01 & \

python baselines/scaffold.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition pathological -dataset cifar10 & \

python baselines/fednova.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition pathological -dataset cifar10 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 1998  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

python baselines/feddyn.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition pathological -dataset cifar10 -alpha_dyn 0.01 & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 1998 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

wait

python fedaware.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37 -partition pathological -dataset cifar10 -alpha 0.5 -startup 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition pathological -dataset cifar10 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 2023 -seed 37  -partition pathological -dataset cifar10 -fedm_beta 0.97 & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition pathological -dataset cifar10 -mu 0.01 & \

python baselines/scaffold.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition pathological -dataset cifar10 & \

python baselines/fednova.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition pathological -dataset cifar10 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 37  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

python baselines/feddyn.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition pathological -dataset cifar10 -alpha_dyn 0.01 & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 37 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-8 & \


wait

python fedaware.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73 -partition pathological -dataset cifar10 -alpha 0.5 -startup 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition pathological -dataset cifar10 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 2023 -seed 73  -partition pathological -dataset cifar10 -fedm_beta 0.97 & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition pathological -dataset cifar10 -mu 0.01 & \

python baselines/scaffold.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition pathological -dataset cifar10 & \

python baselines/fednova.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition pathological -dataset cifar10 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 73  -partition pathological -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 & \

python baselines/feddyn.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition pathological -dataset cifar10 -alpha_dyn 0.01 & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 73 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-8
wait
