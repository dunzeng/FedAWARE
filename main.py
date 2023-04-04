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
from fedlab.utils.functional import evaluate, get_best_gpu, AverageMeter


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

from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.models.mlp import MLP
from mode import UniformSampler

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
        loss_ = AverageMeter()
        acc_ = AverageMeter()
        for id in tqdm(id_list):
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, loss_, acc_)
            self.cache.append(pack)
        return loss_, acc_

    def train(self, model_parameters, train_loader, loss_, acc_):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output, 1)
                loss_.update(loss.item())
                acc_.update(torch.sum(predicted.eq(target)).item(), len(target))

        return [self.model_parameters]

class Server_MomentumGradientCache(SyncServerHandler):
    def setup_optim(self, sampler, alpha, args):  
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler
        self.lr = args.glr

        self.args = args

        self.C = args.C # gradient clipping
        self.momentum = [torch.zeros_like(self.model_parameters) for _ in range(self.n)]
        self.alpha = alpha
        self.solver = MinNormSolver
        self.stats = {"count":np.zeros(self.n)}

    def momentum_update(self, gradients, indices):
        for grad, idx in zip(gradients, indices):
            self.momentum[idx] = (1-self.alpha)*self.momentum[idx] + self.alpha*grad
        
        # norms = np.max((np.array([torch.norm(grad, p=2, dim=0).item() for grad in self.momentum])/self.C, np.ones_like(self.num_clients)), axis=0)
        norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in self.momentum])
        norm_momentum = [self.momentum[i]/n for i, n in enumerate(norms)]

        sol, val = self.solver.find_min_norm_element_FW(norm_momentum)
        print("FW solver - val {},\n lambda: {}".format(val, str(sol)))
        # self.stats["count"] += sol>0
        # sol = sol/sol.sum()
        assert sol.sum()-1 < 1e-5
        return sol, norm_momentum
    
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
        indices, _ = self.sampler.last_sampled
        sol, norm_momentum = self.momentum_update(gradient_list, indices)
        self.sampler.update(sol)
        # indices = np.arange(args.num_clients)
        # norms = np.array([torch.norm(self.momentum[i], p=2, dim=0).item() for i in indices])
        # norm_momentum = [self.momentum[i]/norms[i] for i in indices]
        estimates = Aggregators.fedavg_aggregate(norm_momentum, sol)

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
    parser.add_argument('-solver', type=str, default="fedavg")
    parser.add_argument('-freq', type=int, default=10)
    parser.add_argument('-dseed', type=int, default=0) # data seed

    parser.add_argument('-a', type=float, default=0.0)
    parser.add_argument('-b', type=float, default=0.0)
    parser.add_argument('-dir', type=float, default=0.1)
    parser.add_argument('-C', type=float, default=1.0)
    parser.add_argument('-query_freq', type=int, default=20)

    # MGDA
    parser.add_argument('-alpha', type=float, default=1)
    parser.add_argument('-beta', type=float, default=1)
    return parser.parse_args()

args = parse_args()

# if args.sampler in ['arbi', 'independent', 'optimal']:
args.k = int(args.num_clients*args.sample_ratio)

# format
dataset = args.dataset
if args.dataset == "synthetic":
    dataset = "synthetic_{}_{}".format(args.a, args.b)

run_time = time.strftime("%m-%d-%H:%M")
pareto = "pareto"

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

    weights = np.array([len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)])
    weights = weights/weights.sum()
elif args.dataset == "mnist":
    model = MLP(784,10)
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
    
    weights = np.array([len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)])
    weights = weights/weights.sum()

    test_data = torchvision.datasets.MNIST(root="./datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor())
    gen_test_loader = DataLoader(test_data, batch_size=1024)
else: 
    assert False

trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = Server_MomentumGradientCache(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
    
probs = np.ones(args.num_clients)/args.num_clients
sampler = UniformSampler(args.num_clients, probs)
handler.num_clients = trainer.num_clients
handler.setup_optim(sampler, args.alpha, args)

t = 0

json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

while handler.if_stop is False:
    print("running..")
    # server side
    broadcast = handler.downlink_package
    if t%args.query_freq==0:
        sampled_clients = handler.sample_clients(args.num_clients)
    else:
        sampled_clients = handler.sample_clients(args.k)

    # client side
    train_loss, train_acc = trainer.local_process(broadcast, sampled_clients)
    full_info = trainer.uplink_package
    
    for pack in full_info:
        handler.load(pack)

    t += 1
    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
    
    writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
    writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)

    writer.add_scalar('Test/loss/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Test/accuracy/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, train_loss.avg,  train_acc.avg, tacc, tloss))
    torch.save(handler.stats, os.path.join(path, "stats.pkl"))