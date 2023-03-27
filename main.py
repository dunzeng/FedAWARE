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
        
    def update(self, probs):
        self.p = probs

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

class Server_MomentumGradientCache(SyncServerHandler):
    def setup_optim(self, sampler, alpha, pareto):  
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler
        self.pareto = pareto 

        self.k = 20
        # MGDA momentum sampling
        if pareto:
            self.momentum = [torch.zeros_like(self.model_parameters) for i in range(self.n)]
            self.alpha = alpha
            self.solver = MinNormSolver
        

    def momentum_update(self, gradients, indices):
        for i, idx in enumerate(indices):
            self.momentum[idx] = self.alpha*self.momentum[idx] + (1-self.alpha)*gradients[i]
        norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in self.momentum])
        norm_momentum = [self.momentum[i]/norms[i] for i in range(self.num_clients)]
        sol, _ = self.solver.find_min_norm_element_FW(norm_momentum)
        sol = sol/sol.sum()
        sol = 0.8*sol + 0.2 * 1.0/self.num_clients # mixing
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

        if self.pareto:
            sol = self.momentum_update(gradient_list, np.arange(args.num_clients))
            self.sampler.update(sol)
            indices = self.sampler.sample(self.k)
            estimates = Aggregators.fedavg_aggregate([self.momentum[i] for i in indices])
        else:
            indices = self.sampler.sample(self.k)
            estimates = Aggregators.fedavg_aggregate([gradient_list[i] for i in indices])

        serialized_parameters = self.model_parameters - estimates
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

    # data & reproduction
    
    parser.add_argument('-preprocess', type=bool, default=False)
    parser.add_argument('-seed', type=int, default=0) # run seed
    
    # setting
    parser.add_argument('-dataset', type=str, default="synthetic")
    parser.add_argument('-solver', type=str, default="fedavg")
    parser.add_argument('-freq', type=int, default=10)
    parser.add_argument('-dseed', type=int, default=0) # data seed

    parser.add_argument('-a', type=float, default=1.0)
    parser.add_argument('-b', type=float, default=1.0)

    # MGDA
    parser.add_argument('-pareto', type=int, default=0)
    parser.add_argument('-alpha', type=float, default=0.8)
    return parser.parse_args()

args = parse_args()

# if args.sampler in ['arbi', 'independent', 'optimal']:
args.k = int(args.num_clients*args.sample_ratio)

# format
dataset = args.dataset
if args.dataset == "synthetic":
    dataset = "synthetic_{}_{}".format(args.a, args.b)

run_time = time.strftime("%m-%d-%H:%M")
pareto = "pareto" if args.pareto==1 else "basedline"
base_dir = "logs/"
dir = "./{}/{}_seed_{}/Run{}_NUM{}_BS{}_LR{}_EP{}_K{}_R{}_{}".format(base_dir, dataset, args.dseed, args.seed, args.num_clients, args.batch_size, args.lr, args.epochs, args.k,
            args.com_round, pareto)
log = "{}".format(run_time)

path = os.path.join(dir, log)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

setup_seed(args.seed)

if args.dataset == "synthetic":
    model = LinearReg(100, 10)
    # synthetic_path = "./datasets/synthetic/data_{}_{}_num{}_seed{}".format(args.a, args.b, args.num_clients, args.dseed)
    synthetic_path = "./datasets/synthetic/data_{}_{}_num{}_seed{}".format(args.a, args.b, 130, args.dseed)
    dataset = SyntheticDataset(synthetic_path, synthetic_path + "/feddata/", args.preprocess)
    
    gen_test_data = ConcatDataset([dataset.get_dataset(i, "test") for i in range(100, 130)])
    gen_test_loader = DataLoader(gen_test_data, batch_size=1024)

    weights = np.array([len(dataset.get_dataset(i, "test")) for i in range(args.num_clients)])
else: 
    assert False


# probs = np.ones(args.num_clients)/args.num_clients
weights = weights/weights.sum()
sampler = UniformSampler(args.num_clients, weights)

trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = Server_MomentumGradientCache(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
    
handler.num_clients = trainer.num_clients
handler.setup_optim(sampler, args.alpha, args.pareto)

t = 0

# loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)

# writer.add_scalar('Test/Loss/{}'.format(args.dataset), loss, t)
# writer.add_scalar('Test/Accuracy/{}'.format(args.dataset), acc, t)

# regret
dyrgt = 0
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

while handler.if_stop is False:
    print("running..")
    # server side
    # if t==0:
    #     sampled_clients = handler.sample_clients(args.num_clients)
    # else:
    #     sampled_clients = handler.sample_clients(args.k)
    sampled_clients = np.arange(args.num_clients)
    broadcast = handler.downlink_package

    # client side
    trainer.local_process(broadcast, sampled_clients)
    full_info = trainer.uplink_package
    
    for pack in full_info:
        handler.load(pack)

    t += 1
    # 
    acc_vec, loss_vec = [], []
    for i in range(args.num_clients):
        test_data = dataset.get_dataset(i, "test")
        test_loader = DataLoader(test_data, batch_size=1024)
        tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)
        acc_vec.append(tacc)
        loss_vec.append(tloss)

    acc_vec, loss_vec = np.array(acc_vec), np.array(loss_vec)

    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
    
    writer.add_scalar('Loss/Avg/{}'.format(args.dataset), loss_vec.dot(weights), t)
    writer.add_scalar('Loss/Std/{}'.format(args.dataset), loss_vec.std(), t)
    writer.add_scalar('Accuracy/Avg/{}'.format(args.dataset), acc_vec.dot(weights), t)
    writer.add_scalar('Accuracy/Std/{}'.format(args.dataset), acc_vec.std(), t)

    writer.add_scalar('Loss/Generalization/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Accuracy/Generalization/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}-{:.4f}, Test Accuracy: {:.4f}-{:.4f},".format(t, loss_vec.mean(), loss_vec.std(), acc_vec.mean(), acc_vec.std()))
    

# %%
