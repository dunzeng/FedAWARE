import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision

from fedlab.contrib.dataset.basic_dataset import FedDataset, BaseDataset

class SyntheticDataset(FedDataset):
    def __init__(self, root, path, preprocess=False) -> None:

        self.root = root
        self.path = path
        if preprocess is True:
            self.preprocess(root, path)
        else:
            print("Warning: please make sure that you have preprocess the data once!")

    def preprocess(self, root, path):
        if os.path.exists(self.path) is not True:
            os.makedirs(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "test"))
        
        with open(os.path.join(root, "train.json"),'r') as load_f:
            train_data = json.load(load_f)
        
        with open(os.path.join(root, "test.json"),'r') as load_f:
            test_data = json.load(load_f)
            
        # num_clients = len(train_data["user_data"])
        assert len(train_data["user_data"]) == len(test_data["user_data"])
        
        for i, key in enumerate(train_data['users']):
            # train
            raw_data = train_data['user_data'][key]
            trainset = BaseDataset(torch.Tensor(raw_data['x']), torch.Tensor(raw_data['y']).type(torch.LongTensor))
            torch.save(trainset, os.path.join(path, "train","data{}.pkl".format(i)))
            # test
            raw_data = test_data['user_data'][key]
            testset = BaseDataset(torch.Tensor(raw_data['x']), torch.Tensor(raw_data['y']).type(torch.LongTensor))
            torch.save(testset, os.path.join(path, "test","data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader

class LEAFSyntheticDataset(FedDataset):
    def __init__(self, root, path, preprocess=False) -> None:

        self.root = root
        self.path = path
        if preprocess is True:
            self.preprocess(root, path)
        else:
            print("Warning: please make sure that you have preprocess the data once!")

    def preprocess(self, root, path, partition=0.2):
        
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        raw_data = torch.load(root)
        users, user_data = raw_data["users"], raw_data["user_data"]

        for id in users:
            data, label = user_data[id]['x'], user_data[id]['y']
            train_size = int(len(label)*partition)

            trainset = BaseDataset(torch.Tensor(data[0:train_size]), label[0:train_size])
            torch.save(trainset, os.path.join(path, "train","data{}.pkl".format(id)))

            testset = BaseDataset(torch.Tensor(data[train_size:]), label[train_size:])
            torch.save(testset, os.path.join(path, "test","data{}.pkl".format(id)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size=1024, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader