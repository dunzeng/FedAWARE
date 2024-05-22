# FedAWARE

## Dependencies

 `pip install -r requirement.txt`

## Run

Plesee see the scripts in ```exp_scripts```.

Here is an example:
```
python fedaware.py -num_clients 100 \
                    -com_round 500 \
                    -sample_ratio 0.1 \
                    -batch_size 64 \
                    -epochs 3 \
                    -lr 0.01 \
                    -glr 1 \
                    -dseed 37 [data partition random seed] \
                    -seed 1998 [running random seed] \
                    -partition [pathological/dirichlet] \
                    -dataset [mnist\fmnist\cifar10\agnews] \
                    -alpha 0.5 [hyperparameters]\
                    -startup 1 \
                    -agnostic [0\1] \
                    -preprocess 1 [dataset preprocesssing]
```

For Agnews task, please run ```python agnews_dataset.py``` to preprocess Agnews dataset. And, download pythia model from https://huggingface.co/EleutherAI/pythia-70m.