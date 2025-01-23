# FedAWARE

 Code for paper "[On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond](https://arxiv.org/abs/2310.02702)".


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

## Citation

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
```
