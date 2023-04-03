# %%
import random
import numpy as np
import json
import os
import argparse
import random
from copy import deepcopy
from munch import Munch
import math

from tqdm import tqdm
import sys
import torch
from torch import nn, softmax
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from fedlab.contrib.dataset.basic_dataset import FedDataset

from fedlab.utils import functional as F
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu

from fedlab.models.mlp import MLP
from fedlab.models.cnn import CNN_MNIST, CNN_FEMNIST
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer

from synthetic_dataset import SyntheticDataset

from model import ToyCifarNet, LinearReg
from scipy.special import softmax
import time
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

from min_norm_solvers import MinNormSolver, gradient_normalizers

class UniformSampler:
    def __init__(self, n, probs):
        self.name = "uniform"
        self.n = n
        self.p = probs

    def sample(self, k):
        sampled = np.random.choice(self.n, k, p=self.p, replace=False)
        self.last_sampled = sampled, self.p[sampled]
        return np.sort(sampled)
        
    def update(self, probs, beta=1):
        self.p = (1-beta)*self.p + beta*probs

def solver(weights, k, n):
        norms = np.sqrt(weights)
        idx = np.argsort(norms)
        probs = np.zeros(len(norms))
        l=0
        for l, id in enumerate(idx):
            l = l + 1
            if k+l-n > sum(norms[idx[0:l]])/norms[id]:
                l -= 1
                break
        
        m = sum(norms[idx[0:l]])
        for i in range(len(idx)):
            if i <= l:
                probs[idx[i]] = (k+l-n)*norms[idx[i]]/m
            else:
                probs[idx[i]] = 1
        return np.array(probs)

class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def query_loss(self):
        loss = []
        acc = []
        for i in range(self.num_clients):
            test_loader = self.dataset.get_dataloader(i)
            eval_loss, eval_acc = evaluate(self._model, nn.CrossEntropyLoss(), test_loader)
            loss.append(eval_loss)
            acc.append(acc)
        return np.array(loss), np.array(acc)
    
    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in tqdm(id_list):
            data_loader = self.dataset.get_dataloader(id)
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)

class Server_MomentumGradientCache(SyncServerHandler):
    def setup_optim(self, sampler, args):  
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler

        self.lr = args.glr
        self.k = args.k
        self.method = args.method
        self.solver = MinNormSolver
        
    def momentum_update(self, gradients, indices):
        for i, idx in enumerate(indices):
            self.momentum[idx] = (1-self.alpha)*self.momentum[idx] + self.alpha*gradients[i]
        norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in self.momentum])
        norm_momentum = [self.momentum[i]/norms[i] for i in range(self.num_clients)]
        sol, _ = self.solver.find_min_norm_element_FW(norm_momentum)
        sol = sol/sol.sum()
        # sol = 0.8*sol + 0.2 * 1.0/self.num_clients # mixing
        assert sol.sum()-1 < 1e-5
        return sol
    
    @property
    def num_clients_per_round(self):
        return self.round_clients
           
    def sample_clients(self, k):
        clients = self.sampler.sample(k)
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients
        
    def global_update(self, buffer):
        # print("Theta {:.4f}, Ws {}".format(self.theta, self.ws))
        gradient_list = [torch.sub(self.model_parameters, ele[0]) for ele in buffer]

        if self.method == "mgda":
            norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in gradient_list])
            normlized_gradients = [grad/n for grad, n in zip(gradient_list, norms)]
            sol, val = self.solver.find_min_norm_element_FW(normlized_gradients)
            print("GDA {}".format(val))
            assert val > 1e-5
            estimates = Aggregators.fedavg_aggregate(normlized_gradients, sol)
        elif self.method == "fedavg":
            estimates = Aggregators.fedavg_aggregate(gradient_list)

        serialized_parameters = self.model_parameters - self.lr*estimates
        SerializationTool.deserialize_model(self._model, serialized_parameters)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-num_clients', type=int)
    parser.add_argument('-com_round', type=int)
    parser.add_argument('-sample_ratio', type=float)

    # local solver
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-epochs', type=int)
    parser.add_argument('-lr', type=float)
    parser.add_argument('-glr', type=float)
    # data & reproduction
    
    parser.add_argument('-preprocess', type=bool, default=False)
    parser.add_argument('-seed', type=int, default=0) # run seed
    
    # setting
    parser.add_argument('-dataset', type=str, default="synthetic")
    parser.add_argument('-dseed', type=int, default=0) # data seed

    parser.add_argument('-a', type=float, default=0.0)
    parser.add_argument('-b', type=float, default=0.0)

    # mgda, fedavg, mgda+
    parser.add_argument('-method', type=str, default="mgda")

    return parser.parse_args()

args = parse_args()

# if args.sampler in ['arbi', 'independent', 'optimal']:
args.k = int(args.num_clients*args.sample_ratio)

# format
dataset = args.dataset

setup_seed(args.seed)
if args.dataset == "synthetic":
    dataset = "synthetic_{}_{}".format(args.a, args.b)

run_time = time.strftime("%m-%d-%H:%M")
base_dir = "logs/"
dir = "./{}/{}_seed_{}/Run{}_NUM{}_BS{}_LR{}_EP{}_K{}_R{}_{}".format(base_dir, dataset, args.dseed, args.seed, args.num_clients, args.batch_size, args.lr, args.epochs, args.k,
            args.com_round, args.method)
log = "{}".format(run_time)

path = os.path.join(dir, log)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

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
    dataset = PartitionedMNIST(root="./datasets/mnist/",
                         path="./datasets/mnist/fedmnist/",
                         num_clients=args.num_clients,
                         partition="noniid-labeldir",
                         dir_alpha=0.3,
                         seed=args.dseed,
                         preprocess=args.preprocess,
                         download=True,
                         verbose=True,
                         transform=transforms.Compose(
                             [transforms.ToPILImage(), transforms.ToTensor()]))
    
    weights = np.array([len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)])
    weights = weights/weights.sum()

    test_data = torchvision.datasets.MNIST(root="./datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor())
    gen_test_loader = DataLoader(test_data, batch_size=1024)
else: 
    assert False

# probs = np.ones(args.num_clients)/args.num_clients
if args.method == "mgda":
    sampler = UniformSampler(args.num_clients, np.ones(args.num_clients)/float(args.num_clients))
if args.method == "fedavg":
    sampler = UniformSampler(args.num_clients, weights)


trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = Server_MomentumGradientCache(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
    
handler.num_clients = trainer.num_clients
handler.setup_optim(sampler, args)

t = 0

json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

while handler.if_stop is False:
    print("running..")
    # server side
    broadcast = handler.downlink_package
    sampled_clients = handler.sample_clients(args.k)

    # client side
    trainer.local_process(broadcast, sampled_clients)
    full_info = trainer.uplink_package
    
    for pack in full_info:
        handler.load(pack)

    t += 1
    # 
    acc_vec, loss_vec = [], []
    for i in range(args.num_clients):
        test_data = dataset.get_dataset(i, "train")
        test_loader = DataLoader(test_data, batch_size=1024)
        tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)
        acc_vec.append(tacc)
        loss_vec.append(tloss)

    w = weights
    acc_vec, loss_vec = np.array(acc_vec), np.array(loss_vec)

    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
    
    writer.add_scalar('Loss/Avg/{}'.format(args.dataset), loss_vec.dot(w), t)
    writer.add_scalar('Loss/Std/{}'.format(args.dataset), loss_vec.std(), t)

    writer.add_scalar('Accuracy/Avg/{}'.format(args.dataset), acc_vec.dot(w), t)
    # writer.add_scalar('Accuracy/Std/{}'.format(args.dataset), acc_vec.std(), t)

    writer.add_scalar('Loss/Generalization/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Accuracy/Generalization/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}-{:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, loss_vec.dot(w), loss_vec.std(), acc_vec.dot(w), tacc, tloss))
    

# %%
