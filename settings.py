import numpy as np
import os
import torch
from torch import nn, softmax
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import argparse
import time


from model import TransformerModel, fastText
from fedlab.models.mlp import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10
from fedlab.utils.dataset.partition import CIFAR100Partitioner

from model import (
    ToyCifarNet,
    LinearReg,
    resnet18,
    ToyCifar100Net,
    vgg11_bn,
    RNN_Shakespeare,
)


from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.models.cnn import CNN_FEMNIST
from fedlab.core.standalone import StandalonePipeline

from partitioned_cifar100 import PartitionedCIFAR100
from partitioned_fmnist import PartitionedFMNIST, PathologicalFMNIST
from agnews_dataset import PartitionedAGNews, AGNews_TestDataset

# from shakespeare import ShakespeareDataset
from tqdm import tqdm


def get_settings(args):
    if args.dataset == "cifar10":
        model = ToyCifarNet()
        # model = resnet18()
        # model = vgg11_bn(bn=False, num_class=10)
        if args.partition == "dirichlet":
            dataset = PartitionedCIFAR10(
                root="./datasets/cifar10/",
                path="./datasets/Dirichlet_cifar_{}_{}_{}".format(args.dir, args.num_clients, args.dseed),
                dataname="cifar10",
                num_clients=args.num_clients,
                preprocess=args.preprocess,
                balance=None,
                partition="dirichlet",
                dir_alpha=args.dir,
                transform=transforms.Compose(
                    [
                        # transforms.ToPILImage(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            )

        weights = np.array(
            [len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)]
        )
        weights = weights / weights.sum()

        test_data = torchvision.datasets.CIFAR10(
            root="./datasets/cifar10/",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )
        gen_test_loader = DataLoader(test_data, num_workers=4, batch_size=1024)

        test_data = torchvision.datasets.CIFAR10(
            root="./datasets/cifar10/",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )

        gen_train_loader = DataLoader(test_data, num_workers=4, batch_size=1024)

    elif args.dataset == "cifar100":
        from resnet_gn import ResNet18_gn
        model = ResNet18_gn()
        trainset = torchvision.datasets.CIFAR100(
            root="./datasets/cifar100/", train=True, download=True
        )
        
        if args.partition == "dirichlet":
            hetero_dir_part = CIFAR100Partitioner(
                trainset.targets,
                args.num_clients,
                balance=None,
                partition="dirichlet",
                dir_alpha=args.dir,
                seed=args.seed,
            )

            dataset = PartitionedCIFAR100(
                root="./datasets/cifar100/",
                path="./datasets/Dirichlet_cifar100_{}".format(args.dir),
                dataname="cifar100",
                num_clients=args.num_clients,
                preprocess=args.preprocess,
                partitioner=hetero_dir_part,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),

        weights = np.array(
            [len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)]
        )
        weights = weights / weights.sum()

        test_data = torchvision.datasets.CIFAR100(
            root="./datasets/cifar100/",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ])
        )
        gen_test_loader = DataLoader(test_data, batch_size=1024, num_workers=4)

    elif args.dataset == "agnews":
        from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification
        from transformers import AutoTokenizer, DataCollatorWithPadding

        # model = AutoModelForSequenceClassification.from_pretrained("/data/distilbert", num_labels=4)
        # tokenizer=AutoTokenizer.from_pretrained("/data/distilbert")

        model = AutoModelForSequenceClassification.from_pretrained("/data/pythia-70m", num_labels=4)
        tokenizer=AutoTokenizer.from_pretrained("/data/pythia-70m")
        model.config.pad_token_id = tokenizer.pad_token_id

        dataset = PartitionedAGNews(root="datasets", path="datasets/partitioned_agnews", num_clients=100)
        gen_test_loader = AGNews_TestDataset(tokenizer)
        weights = np.array(
            [len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)]
        )
        weights = weights / weights.sum()

    else:
        assert False

    return model, dataset, weights, gen_test_loader


def get_logs(args):
    run_time = time.strftime("%m-%d-%H:%M:%S")
    if args.partition == "dirichlet":
        data_log = "{}_{}_{}_{}".format(
            args.dataset, args.partition, args.dir, args.dseed
        )
    else:
        data_log = "{}_{}_{}".format(args.dataset, args.partition, args.dseed)
    dir = "./{}-logs-aistats/{}/Run{}_N{}_BS{}_EP{}_LLR{}_K{}_T{}_H{}_Projection{}".format(
        args.dataset,
        data_log,
        args.seed,
        args.num_clients,
        args.batch_size,
        args.epochs,
        args.lr,
        args.k,
        args.com_round,
        args.agnostic,
        args.projection
    )

    if args.method == "fedavg":
        log = "Setting_{}_GLR{}_{}".format(args.method, args.glr, run_time)
    elif args.method == "fedavgm":
        log = "Setting_{}_GLR{}_momentum{}_{}".format(
            args.method, args.glr, args.fedm_beta, run_time
        )
    elif args.method == "fedprox":
        log = "Setting_{}_GLR{}_mu{}_{}".format(
            args.method, args.glr, args.mu, run_time
        )
    elif args.method == "scaffold":
        log = "Setting_{}_GLR{}_{}".format(args.method, args.glr, run_time)
    elif args.method == "fedopt":
        log = "Setting_{}_GLR{}_{}_{}".format(
            args.method, args.glr, args.option, run_time
        )
    elif args.method == "fednova":
        log = "Setting_{}_GLR{}_{}".format(args.method, args.glr, run_time)
    elif args.method == "feddyn":
        log = "Setting_{}_GLR{}_alpha{}_{}".format(
            args.method, args.glr, args.alpha_dyn, run_time
        )
    elif args.method == "fedams":
        log = "Setting_{}_GLR{}_{}_eps{}_{}".format(
            args.method, args.glr, args.option, args.eps, run_time
        )
    elif args.method == "ours":
        log = "Setting_{}_GLR{}_momentum{}_{}_{}".format(
                "fedaware", args.glr, args.alpha, args.label, run_time)
    elif args.method == "fedaware_ablation":
        log = "Setting_{}_GLR{}_momentum{}_{}_{}".format(
                "fedaware-no-reweight", args.glr, args.alpha, args.label, run_time
            )
    elif args.method == "fedcm":
        log = "Setting_{}_alpha{}_{}".format(args.method, args.alpha, run_time)
    else:
        assert False

    path = os.path.join(dir, log)
    return path


def get_heterogeneity(args, datasize):
    if args.agnostic == 1:
        eps = np.random.randint(2, 5 + 1)
        batch_size = np.random.randint(10, datasize) if datasize > 10 else datasize
        # print("size {} - batch {} - ep {}".format(datasize, batch_size, eps))
        return batch_size, eps
    else:
        return args.batch_size, args.epochs
        # steps = 10
        # eps = args.epochs
        # batch_size = int(np.ceil(datasize/(steps/eps)))
        # print("size {} - batch {} - ep {}".format(datasize, batch_size, eps))
        # return batch_size, eps
        # return args.batch_size, args.epochs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default="None")

    parser.add_argument("-num_clients", type=int)
    parser.add_argument("-com_round", type=int)
    parser.add_argument("-sample_ratio", type=float)

    # local solver
    parser.add_argument("-optim", type=str)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-lr", type=float)
    parser.add_argument("-glr", type=float)
    parser.add_argument("-agnostic", type=float, default=0)
    parser.add_argument("-local_momentum", type=float, default=0)

    # data & reproduction
    parser.add_argument("-dataset", type=str, default="synthetic")
    parser.add_argument(
        "-partition", type=str, default="dirichlet"
    )  # dirichlet, pathological
    parser.add_argument("-dir", type=float, default=0.1)
    parser.add_argument("-preprocess", type=bool, default=False)
    parser.add_argument("-seed", type=int, default=0)  # run seed
    parser.add_argument("-dseed", type=int, default=0)  # data seed

    parser.add_argument("-freq", type=int, default=1) 
    # fedavgm
    parser.add_argument("-fedm_beta", type=float)

    # fedprox
    parser.add_argument("-mu", type=float)

    # fedopt
    # adagrad, yogi, adam
    parser.add_argument("-option", type=str, default="yogi")
    parser.add_argument("-beta1", type=float)
    parser.add_argument("-beta2", type=float)
    parser.add_argument("-tau", type=float)

    # fedams
    # parser.add_argument('-option', type=str, default="fedams") # fedams, fedamsgrad
    parser.add_argument("-eps", type=float)
    # parser.add_argument('-max_init', type=float)

    # fednova

    # scaffold
    
    # fedcm
    parser.add_argument("-alpha_cm", type=float, default=0.05)

    # feddyn
    parser.add_argument("-alpha_dyn", type=float)

    # ours
    parser.add_argument("-alpha", type=float, default=0.5)
    parser.add_argument("-startup", type=int, default=0)
    parser.add_argument("-projection", type=int, default=0)
    parser.add_argument("-label", type=str)
    parser.add_argument("-ablation", type=int, default=0)

    return parser.parse_args()
