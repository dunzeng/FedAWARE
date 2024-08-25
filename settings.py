import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import time

from fedlab.models.mlp import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

from model import ToyCifarNet

from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from partitioned_fmnist import PartitionedFMNIST, PathologicalFMNIST
from fedlab.utils.dataset.partition import CIFAR100Partitioner
from partitioned_cifar100 import PartitionedCIFAR100
from agnews_dataset import PartitionedAGNews, AGNews_TestDataset

def get_settings(args):
    if args.dataset == "mnist":
        model = MLP(784, 10)
        if args.partition == "dirichlet":
            dataset = PartitionedMNIST(
                root="./datasets/mnist/",
                path="./datasets/mnist/fedmnist_{}/".format(args.dir),
                num_clients=args.num_clients,
                partition="noniid-labeldir",
                dir_alpha=args.dir,
                seed=args.dseed,
                preprocess=args.preprocess,
                download=True,
                verbose=True,
                transform=transforms.Compose(
                    [transforms.ToPILImage(), transforms.ToTensor()]
                ),
            )
        if args.partition == "pathological":
            dataset = PathologicalMNIST(
                root="./datasets/mnist/",
                path="./datasets/mnist/pathological_mnist_{}/".format(args.dseed),
                preprocess=args.preprocess,
            )

        weights = np.array(
            [len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)]
        )
        weights = weights / weights.sum()

        test_data = torchvision.datasets.MNIST(
            root="./datasets/mnist/", train=False, transform=transforms.ToTensor()
        )
        gen_test_loader = DataLoader(test_data, batch_size=1024)

    elif args.dataset == "fmnist":
        model = MLP(784, 10)
        if args.partition == "dirichlet":
            dataset = PartitionedFMNIST(
                root="./datasets/fmnist/",
                path="./datasets/fmnist/fed_fmnist_{}/".format(args.dir),
                num_clients=args.num_clients,
                partition="noniid-labeldir",
                dir_alpha=args.dir,
                seed=args.dseed,
                preprocess=args.preprocess,
                download=True,
                verbose=True,
                transform=transforms.Compose(
                    [transforms.ToPILImage(), transforms.ToTensor()]
                ),
            )
        if args.partition == "pathological":
            dataset = PathologicalFMNIST(
                root="./datasets/fmnist/",
                path="./datasets/fmnist/pathological_fmnist_{}/".format(args.dseed),
                preprocess=args.preprocess,
            )

        weights = np.array(
            [len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)]
        )
        weights = weights / weights.sum()

        test_data = torchvision.datasets.FashionMNIST(
            root="./datasets/fmnist/", train=False, transform=transforms.ToTensor()
        )

        gen_test_loader = DataLoader(test_data, batch_size=1024)

    elif args.dataset == "cifar10":
        model = ToyCifarNet()
        if args.partition == "dirichlet":
            dataset = PartitionedCIFAR10(
                root="./datasets/cifar10/",
                path="./datasets/Dirichlet_cifar_{}".format(args.dir),
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
        if args.partition == "pathological":
            dataset = PartitionedCIFAR10(
                root="./datasets/cifar10/",
                path="./datasets/pathological_cifar",
                dataname="cifar10",
                num_clients=args.num_clients,
                preprocess=args.preprocess,
                balance=None,
                partition="shards",
                num_shards=200,
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

    elif args.dataset == "cifar100":
        # model = vgg11_bn(bn=False, num_class=100)
        # model = resnet18()
        # model = ToyCifar100Net()
        from model import ResNet18_gn
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
            )
        if args.partition == "pathological":
            pathological_part = CIFAR100Partitioner(
                trainset.targets,
                args.num_clients,
                balance=None,
                num_shards=200,
                partition="shards",
                dir_alpha=args.dir,
                seed=args.seed,
            )

            dataset = PartitionedCIFAR100(
                root="./datasets/cifar100/",
                path="./datasets/Pathological_cifar100",
                dataname="cifar100",
                num_clients=args.num_clients,
                preprocess=args.preprocess,
                partitioner=pathological_part,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            )
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
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        gen_test_loader = DataLoader(test_data, batch_size=1024, num_workers=4)

    elif args.dataset == "agnews":
        from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification
        from transformers import AutoTokenizer, DataCollatorWithPadding

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
    dir = "./{}-logs/{}/Run{}_N{}_BS{}_EP{}_LLR{}_K{}_T{}_H{}_Projection{}".format(
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
    elif args.method == "fedaware":
        log = "Setting_{}_GLR{}_momentum{}_{}_{}".format(
            "fedaware", args.glr, args.alpha, args.label, run_time
        )
    elif args.method == "fedawyogi":
        log = "Setting_{}_GLR{}_{}".format(args.method, args.glr, run_time)
    else:
        assert False

    path = os.path.join(dir, log)
    return path


def get_heterogeneity(args, datasize):
    if args.agnostic == 1:
        eps = np.random.randint(2, 5 + 1)
        batch_size = np.random.randint(10, datasize) if datasize > 10 else datasize
        return batch_size, eps
    else:
        return args.batch_size, args.epochs


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
    parser.add_argument("-option", type=str, default="yogi") # adagrad, yogi, adam
    parser.add_argument("-beta1", type=float)
    parser.add_argument("-beta2", type=float)
    parser.add_argument("-tau", type=float)

    # fedams
    # parser.add_argument('-option', type=str, default="fedams") # fedams, fedamsgrad
    parser.add_argument("-eps", type=float)

    # fednova

    # scaffold

    # feddyn
    parser.add_argument("-alpha_dyn", type=float)

    # ours
    parser.add_argument("-alpha", type=float, default=0.5)
    parser.add_argument("-startup", type=int, default=0)
    parser.add_argument("-projection", type=int, default=0)
    parser.add_argument("-label", type=str)

    return parser.parse_args()
