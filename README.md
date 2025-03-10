# FedAWARE

## Dependencies

 `pip install -r requirement.txt`

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

For Agnews task, please run ```python agnews_dataset.py``` to preprocess Agnews dataset. And, download pythia model from https://huggingface.co/EleutherAI/pythia-70m.

Note:

Due to attachment size limitations, we only show the implementation details of our work here.  And, please see utils.py FedAWARE_Projector class for our implementation details.

Full experiment results and their reproduction scripts will be released if accepted.

## Reference

Please cite our paper if you found the code useful.

```
@misc{zeng2024poweradaptiveweightedaggregation,
      title={On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond}, 
      author={Dun Zeng and Zenglin Xu and Shiyu Liu and Yu Pan and Qifan Wang and Xiaoying Tang},
      year={2024},
      eprint={2310.02702},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.02702}, 
}

@inproceedings{zengpower,
  title={On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond},
  author={Zeng, Dun and Xu, Zenglin and LIU, SHIYU and Pan, Yu and Wang, Qifan and Tang, Xiaoying},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics}
}
```