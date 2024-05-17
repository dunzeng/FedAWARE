# FedAWARE

## Dependencies

 `pip install -r requirement.txt`

## Run

```
python fedaware.py -num_clients 100 \
                    -com_round 500 \
                    -sample_ratio 0.1 \
                    -batch_size 64 \
                    -epochs 3 \
                    -lr 0.01 \
                    -glr 1 \
                    -dseed 37 \
                    -seed 1998 \
                    -partition [pathological/dirichlet] \
                    -dataset [mnist\fmnist\cifar10] \
                    -alpha 0.5 \
                    -startup 1 \
                    -agnostic [0\1] \
                    -preprocess 1
```
