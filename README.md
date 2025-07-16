# FedAWARE

 Code for AISTATS'25 paper "[On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond](https://arxiv.org/abs/2310.02702)".


## Dependencies

 `pip install -r requirements.txt`

## Run

```
python fedaware.py.py -num_clients 100 \
                    -com_round 500 \
                    -sample_ratio 0.1 \
                    -batch_size 64 \
                    -epochs 3 \
                    -lr 0.01 \
                    -glr 1 \
                    -dseed 37 [data partition random seed] \
                    -seed 42 [running random seed] \
                    -partition dirichlet \
                    -dir 0.1 \
                    -dataset [cifar10/cifar100/agnews] \
                    -alpha 0.5 [hyperparameters]\
                    -preprocess 1 [dataset preprocesssing] 
```

Note:

- For Agnews task, please run ```python agnews_dataset.py``` to preprocess Agnews dataset. And, download pythia model from https://huggingface.co/EleutherAI/pythia-70m.

- Please see utils.py FedAWARE_Projector class for our implementation details.

- Leave an issue if you have any questions.

## Reference

Please cite our paper if you found the code useful.

```
@inproceedings{zeng2025power,
  title={On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond},
  author={Zeng, Dun and Xu, Zenglin and LIU, SHIYU and Pan, Yu and Wang, Qifan and Tang, Xiaoying},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1081--1089},
  year={2025},
  organization={PMLR}
}
```
