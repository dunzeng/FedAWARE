import sys
sys.path.append("./")

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

from fedlab.utils.functional import evaluate, setup_seed, AverageMeter
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer

from scipy.special import softmax
import time
from copy import deepcopy

from mode import FedAvgSerialClientTrainer, UniformSampler, solver, gradient_diversity

from torch.utils.tensorboard import SummaryWriter

from min_norm_solvers import MinNormSolver, gradient_normalizers
from settings import get_settings, get_heterogeneity

METHOD = "FedAvg"

class Server_MomentumGradientCache(SyncServerHandler):
    def setup_optim(self, sampler, args):  
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler

        self.args = args
        self.lr = args.glr
        self.k = args.k
        self.method = args.method
        self.solver = MinNormSolver
    
    @property
    def num_clients_per_round(self):
        return self.round_clients
           
    def sample_clients(self, k):
        clients = self.sampler.sample(k)
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients
        
    def global_update(self, buffer):
        gradient_list = [torch.sub(self.model_parameters, ele[0]) for ele in buffer]
        indices, _ = self.sampler.last_sampled
        estimates = Aggregators.fedavg_aggregate(gradient_list, self.args.weights[indices])

        serialized_parameters = self.model_parameters - self.lr*estimates
        SerializationTool.deserialize_model(self._model, serialized_parameters)

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
            dataset = self.dataset.get_dataset(id)
            self.batch_size, self.epochs = get_heterogeneity(args, len(dataset))
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            #data_loader = self.dataset.get_dataloader(id, self.batch_size)
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
    parser.add_argument('-dataset', type=str, default="synthetic")
    parser.add_argument('-partition', type=str, default="dir") # dir, path
    parser.add_argument('-dir', type=float, default=0.1)
    parser.add_argument('-preprocess', type=bool, default=False)
    parser.add_argument('-seed', type=int, default=0) # run seed
    parser.add_argument('-dseed', type=int, default=0) # data seed
    parser.add_argument('-method', type=str, default="fedavg")

    parser.add_argument('-agnostic', type=int, default=0)
    return parser.parse_args()

args = parse_args()

setup_seed(args.seed)
args.k = int(args.num_clients*args.sample_ratio)

# format
dataset = args.dataset

run_time = time.strftime("%m-%d-%H:%M:%S")
base_dir = "logs/"
dir = "./{}/{}/DataSeed{}_RunSeed{}_NUM{}_BS{}_LR{}_EP{}_K{}_T{}".format(base_dir, dataset, args.dseed, args.seed, args.num_clients, args.batch_size, args.lr, args.epochs, args.k, args.com_round)
log = "Setting_Agnostic{}_{}_{}".format(args.agnostic, METHOD, run_time)

path = os.path.join(dir, log)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

if args.method == "fedavg":
    probs = np.ones(args.num_clients)/args.num_clients
    sampler = UniformSampler(args.num_clients, probs)
    
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

while handler.if_stop is False:
    print("running..")
    # server side
    broadcast = handler.downlink_package
    sampled_clients = handler.sample_clients(args.k)

    # client side
    train_loss, train_acc = trainer.local_process(broadcast, sampled_clients)
    full_info = trainer.uplink_package
    
    gradient_list = [torch.sub(handler.model_parameters, ele[0]) for ele in full_info]
    diversity = gradient_diversity(gradient_list)
    writer.add_scalar('Metric/Diversity/{}'.format(args.dataset), diversity, t)

    for pack in full_info:
        handler.load(pack)

    t += 1
    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
    
    writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
    writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)

    writer.add_scalar('Test/loss/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Test/accuracy/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, train_loss.avg,  train_acc.avg, tacc, tloss))