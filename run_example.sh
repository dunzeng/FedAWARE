# running command cexamples

# statistical only
python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.5 -startup 1 -preproecss 1

# hybrid
python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.5 -startup 1 -preproecss 1

# hybird+
python main.py -num_clients 100 -com_round 500 -sample_ratio 0.1 -batch_size 64 -epochs 3 -lr 0.01 -glr 1 -dseed 37 -seed 1998 -partition dirichlet -dir 0.1 -dataset cifar10 -alpha 0.5 -startup 1 -agnostic 1 -preproecss 1