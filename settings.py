
import numpy as np
from torch import nn, softmax
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from fedlab.contrib.dataset.basic_dataset import FedDataset

from fedlab.utils import functional as F
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu, AverageMeter


from fedlab.models.mlp import MLP
from fedlab.models.cnn import CNN_MNIST, CNN_FEMNIST
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

from synthetic_dataset import SyntheticDataset
from model import ToyCifarNet, LinearReg



from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.models.mlp import MLP

def get_settings(args):
    if args.dataset == "synthetic":
        model = LinearReg(100, 10)
        # synthetic_path = "./datasets/synthetic/data_{}_{}_num{}_seed{}".format(args.a, args.b, args.num_clients, args.dseed)
        synthetic_path = "./datasets/synthetic/data_{}_{}_num{}_seed{}".format(args.a, args.b, 130, args.dseed)
        dataset = SyntheticDataset(synthetic_path, synthetic_path + "/feddata/", args.preprocess)
        
        gen_test_data = ConcatDataset([dataset.get_dataset(i, "test") for i in range(100, 130)])
        gen_test_loader = DataLoader(gen_test_data, batch_size=1024)

        weights = np.array([len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)])
        weights = weights/weights.sum()

    elif args.dataset == "mnist":
        model = MLP(784,10)
        if args.partition == "dir":
            dataset = PartitionedMNIST(root="./datasets/mnist/",
                                path="./datasets/mnist/fedmnist_{}/".format(args.dir),
                                num_clients=args.num_clients,
                                partition="noniid-labeldir",
                                dir_alpha=args.dir,
                                seed=args.dseed,
                                preprocess=args.preprocess,
                                download=True,
                                verbose=True,
                                transform=transforms.Compose(
                                    [transforms.ToPILImage(), transforms.ToTensor()]))
        if args.partition == "path":
            dataset = PathologicalMNIST(root="./datasets/mnist/",
                                path="./datasets/mnist/pathological_mnist_{}/".format(args.dseed),
                                preprocess=args.preprocess)

        weights = np.array([len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)])
        weights = weights/weights.sum()

        test_data = torchvision.datasets.MNIST(root="./datasets/mnist/",
                                        train=False,
                                        transform=transforms.ToTensor())
        gen_test_loader = DataLoader(test_data, batch_size=1024)

    elif args.dataset == "cifar10":
        model = ToyCifarNet()
        if args.partition == "dir":
            dataset = PartitionedCIFAR10(root="./datasets/cifar10/",
                            path="./datasets/Dirichlet_cifar_{}".format(args.dir),
                            dataname="cifar10",
                            num_clients=args.num_clients,
                            preprocess=args.preprocess,
                            balance=False,
                            partition="dirichlet",
                            dir_alpha=args.dir,
                            transform=transforms.Compose([
                                # transforms.ToPILImage(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                            ]))
        if args.partition == "path":
            dataset = PartitionedCIFAR10(root="./datasets/cifar10/",
                            path="./datasets/pathological_cifar",
                            dataname="cifar10",
                            num_clients=args.num_clients,
                            preprocess=args.preprocess,
                            balance=None,
                            partition="shards",
                            num_shards=200,
                            transform=transforms.Compose([
                                # transforms.ToPILImage(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                            ]))

        weights = np.array([len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)])
        weights = weights/weights.sum()

        test_data = torchvision.datasets.CIFAR10(root="./datasets/cifar10/",
                                                train=False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        (0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                                                ]))
        gen_test_loader = DataLoader(test_data, batch_size=1024)

    else: 
        assert False
    
    return model, dataset, weights, gen_test_loader