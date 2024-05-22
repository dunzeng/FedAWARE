python baselines/fedopt.py -num_clients 100 -com_round 305 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 0.001 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset agnews -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -freq 5 -projection 1 -alpha 0.99

python baselines/fedopt.py -num_clients 100 -com_round 305 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 0.001 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset agnews -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -freq 5 -projection 1 -alpha 0.99

python baselines/fedopt.py -num_clients 100 -com_round 305 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 0.001 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset agnews -option yogi -beta1 0.9 -beta2 0.99 -tau 0.0001 -freq 5 -projection 1 -alpha 0.99


python baselines/fedams.py -num_clients 100 -com_round 305 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 0.001  -partition dirichlet -dir 0.1 -dataset agnews -dseed 2023 -seed 37 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -freq 5 -projection 1 -alpha 0.99

python baselines/fedams.py -num_clients 100 -com_round 305 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 0.001  -partition dirichlet -dir 0.1 -dataset agnews -dseed 2023 -seed 73 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -freq 5 -projection 1 -alpha 0.99

python baselines/fedams.py -num_clients 100 -com_round 205 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 0.001  -partition dirichlet -dir 0.1 -dataset agnews -dseed 2023 -seed 1998 -option fedams -beta1 0.9 -beta2 0.99 -eps 1e-8 -freq 5 -projection 1 -alpha 0.99


python baselines/fedavg.py  -num_clients 100 -com_round 205 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset agnews -freq 5 -projection 1 -alpha 0.99

python baselines/fedavg.py  -num_clients 100 -com_round 205 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset agnews -freq 5 -projection 1 -alpha 0.99

python baselines/fedavg.py  -num_clients 100 -com_round 205 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset agnews -freq 5 -projection 1 -alpha 0.99


python baselines/fedavgm.py  -num_clients 100 -com_round 205 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 1 -dseed 2023 -seed 37  -partition dirichlet -dir 0.1 -dataset agnews -freq 5 -fedm_beta 0.7 -projection 1 -alpha 0.99

python baselines/fedavgm.py  -num_clients 100 -com_round 205 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 1 -dseed 2023 -seed 73  -partition dirichlet -dir 0.1 -dataset agnews -freq 5 -fedm_beta 0.7 -projection 1 -alpha 0.99

python baselines/fedavgm.py  -num_clients 100 -com_round 205 -sample_ratio 0.1 -batch_size 32 -epochs 1 -lr 0.0001 -glr 1 -dseed 2023 -seed 1998  -partition dirichlet -dir 0.1 -dataset agnews -freq 5 -fedm_beta 0.7 -projection 1 -alpha 0.99