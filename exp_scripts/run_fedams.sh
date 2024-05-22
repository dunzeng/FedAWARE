#!bin/bash

# CUDA_VISIBLE_DEVICES=0 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

# CUDA_VISIBLE_DEVICES=1 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-4 & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 -projection 1 & \

CUDA_VISIBLE_DEVICES=1 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition pathological -dataset cifar10 -dseed 2023 -seed 37 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 & \

CUDA_VISIBLE_DEVICES=2 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition pathological -dataset cifar10 -dseed 2023 -seed 2023 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 & \


CUDA_VISIBLE_DEVICES=3 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 0  -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 & \

CUDA_VISIBLE_DEVICES=4 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 37 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 & \

CUDA_VISIBLE_DEVICES=5 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 2023 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 & \


CUDA_VISIBLE_DEVICES=7 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 0  -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 -agnostic 1 & \

CUDA_VISIBLE_DEVICES=2 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 37 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 -agnostic 1 & \

CUDA_VISIBLE_DEVICES=0 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 2023 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 -agnostic 1 & \

# CUDA_VISIBLE_DEVICES=3 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-2 & \

# CUDA_VISIBLE_DEVICES=4 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-1 & \

wait

CUDA_VISIBLE_DEVICES=0 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

# CUDA_VISIBLE_DEVICES=1 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-4 & \

# CUDA_VISIBLE_DEVICES=2 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-3 & \

# CUDA_VISIBLE_DEVICES=3 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-2 & \

# CUDA_VISIBLE_DEVICES=4 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-1 & \

wait


CUDA_VISIBLE_DEVICES=1 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 0  -option fedamsgrad -beta1 0.9 -beta2 0.99 -eps 1e-3 & \

CUDA_VISIBLE_DEVICES=2 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 0  -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

# 10−8, 10−4, 10−3, 10−2, 10−1



CUDA_VISIBLE_DEVICES=0 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition pathological -dataset cifar10 -dseed 2023 -seed 0 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

CUDA_VISIBLE_DEVICES=1 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition pathological -dataset cifar10 -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

CUDA_VISIBLE_DEVICES=2 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition pathological -dataset cifar10 -dseed 2023 -seed 2023 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \


CUDA_VISIBLE_DEVICES=3 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 0  -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

CUDA_VISIBLE_DEVICES=4 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \

CUDA_VISIBLE_DEVICES=5 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 2023 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 & \


CUDA_VISIBLE_DEVICES=7 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 0  -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -agnostic 1 & \

CUDA_VISIBLE_DEVICES=2 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -agnostic 1 & \

CUDA_VISIBLE_DEVICES=0 python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 2023 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -agnostic 1 & \