python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -agnostic 1 -projection 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -agnostic 1 -projection 1 & \

python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -agnostic 1 -projection 1 & \

sleep 3s

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -agnostic 1 -projection 1 & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 1998 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -agnostic 1 -projection 1 & \

python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.005  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 73 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -agnostic 1 -projection 1 & \

wait

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -agnostic 1 -projection 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar10 -agnostic 1 -projection 1 & \

python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar10 -agnostic 1 -projection 1 & \

sleep 3s

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.7 -agnostic 1 -projection 1 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.7 -agnostic 1 -projection 1 & \

python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.7 -agnostic 1 -projection 1 & \

wait

# python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -projection 1 & \

# python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar10 -projection 1 & \

# python baselines/fedavg.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar10 -projection 1 & \

# sleep 3s
# w
# python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.7 -projection 1 & \

# python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.7 -projection 1 & \

# python baselines/fedavgm.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.1 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -fedm_beta 0.7 -projection 1 & \

# wait

# python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -projection 1 & \

# python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -projection 1 & \

# python baselines/fedopt.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset cifar10 -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -projection 1 & \

# sleep 3s

# python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -projection 1 & \

# python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 1998 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -projection 1 & \

# python baselines/fedams.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 0.01  -partition dirichlet -dir 0.1 -dataset cifar10 -dseed 2023 -seed 73 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -projection 1 & \

# wait