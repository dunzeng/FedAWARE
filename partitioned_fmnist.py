import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from fedlab.contrib.dataset.basic_dataset import FedDataset, Subset, BaseDataset
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner, MNISTPartitioner
from fedlab.utils.dataset.functional import noniid_slicing



class PartitionedFMNIST(PartitionedMNIST):
    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True,
                   transform=None,
                   target_transform=None):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        trainset = torchvision.datasets.FashionMNIST(root=self.root,
                                                train=True,
                                                download=download)

        partitioner = MNISTPartitioner(trainset.targets,
                                        self.num_clients,
                                        partition=partition,
                                        dir_alpha=dir_alpha,
                                        verbose=verbose,
                                        seed=seed)

        # partition
        subsets = {
            cid: Subset(trainset,
                        partitioner.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))
            


class PathologicalFMNIST(FedDataset):
    """The partition stratigy in FedAvg. See http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com

        Args:
            root (str): Path to download raw dataset.
            path (str): Path to save partitioned subdataset.
            num_clients (int): Number of clients.
            shards (int, optional): Sort the dataset by the label, and uniformly partition them into shards. Then 
            download (bool, optional): Download. Defaults to True.
        """
    def __init__(self, root, path, num_clients=100, shards=200, download=True, preprocess=False) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.shards = shards
        if preprocess:
            self.preprocess(download)

    def preprocess(self, download=True):
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
        
        if os.path.exists(os.path.join(self.path, "train")) is not True:
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))
            
        # train
        mnist = torchvision.datasets.FashionMNIST(self.root, train=True, download=self.download,
                                           transform=transforms.ToTensor())
        data_indices = noniid_slicing(mnist, self.num_clients, self.shards)

        samples, labels = [], []
        for x, y in mnist:
            samples.append(x)
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.path, "train", "data{}.pkl".format(id)))

    def get_dataset(self, id, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
            cid (int): client id
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
