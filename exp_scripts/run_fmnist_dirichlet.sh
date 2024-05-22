#!/bin/bash


python fedaware.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 1998 -partition dirichlet -dir 0.1  -alpha 0.3  & \

python fedaware.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 73 -partition dirichlet -dir 0.1  -alpha 0.3  & \

python fedaware.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 37 -partition dirichlet -dir 0.1  -alpha 0.3  & \

sleep 3s

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1   & \

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 73  -partition dirichlet -dir 0.1   & \

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 37  -partition dirichlet -dir 0.1   & \

wait

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1  -mu 0.1  & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 73  -partition dirichlet -dir 0.1  -mu 0.1  & \

python baselines/fedprox.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 37  -partition dirichlet -dir 0.1  -mu 0.1  & \

sleep 3s

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1   -fedm_beta 0.7 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 73  -partition dirichlet -dir 0.1   -fedm_beta 0.7 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 37  -partition dirichlet -dir 0.1   -fedm_beta 0.7 & \

wait

python baselines/fednova.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1   & \

python baselines/fednova.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 73  -partition dirichlet -dir 0.1   & \

python baselines/fednova.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 37  -partition dirichlet -dir 0.1   & \

sleep 3s

python baselines/scaffold.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1   & \

python baselines/scaffold.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 73  -partition dirichlet -dir 0.1   & \

python baselines/scaffold.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dataset fmnist -dseed 2023 -seed 37  -partition dirichlet -dir 0.1   & \

wait

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dataset fmnist -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1  -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001  & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dataset fmnist -dseed 2023 -seed 73  -partition dirichlet -dir 0.1  -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001  & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dataset fmnist -dseed 2023 -seed 37  -partition dirichlet -dir 0.1  -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001  & \

wait

python baselines/feddyn.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset fmnist -alpha_dyn 0.01  & \

python baselines/feddyn.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset fmnist -alpha_dyn 0.01  & \

python baselines/feddyn.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset fmnist -alpha_dyn 0.01  & \

sleep 3s

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset fmnist -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8  & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset fmnist -dseed 2023 -seed 1998 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8  & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset fmnist -dseed 2023 -seed 73 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8  & \

wait