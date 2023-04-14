import sys
sys.path.append("./")

import numpy as np
import json
import os
import argparse

import torch
from torch import nn
from tqdm import tqdm
from fedlab.utils.functional import evaluate, setup_seed, AverageMeter
from fedlab.contrib.algorithm.fednova import FedNovaServerHandler
from fedlab.utils.aggregator import Aggregators
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer

import time
from torch.utils.data import DataLoader
from mode import UniformSampler, gradient_diversity

from torch.utils.tensorboard import SummaryWriter
from min_norm_solvers import MinNormSolver, gradient_normalizers
from settings import get_settings, get_logs, parse_args, get_heterogeneity



class FedNovaServerHandler_(FedNovaServerHandler):
    def setup_optim(self, sampler, args):   
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler

        self.args = args
        self.lr = args.glr
        self.k = args.k
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
        taus = np.array([elem[1].item() for elem in buffer])
        gradient_list = [torch.sub(self.model_parameters, ele[0]) for ele in buffer]
        indices, _ = self.sampler.last_sampled

        tau_eff = (taus*self.args.weights[indices]).sum()
        reweights = (tau_eff/taus)*self.args.weights[indices]
        delta = Aggregators.fedavg_aggregate(gradient_list, reweights)

        self.set_model(self.model_parameters -  delta)
    
class FedNovaSerialClientTrainer_(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        loss_, acc_ = AverageMeter(), AverageMeter()
        for id in id_list:
            dataset = self.dataset.get_dataset(id)
            self.batch_size, self.epochs = get_heterogeneity(args, len(dataset))
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            pack = self.train(model_parameters, data_loader, loss_, acc_)
            tau = [torch.Tensor([len(data_loader) * self.epochs])]
            pack += tau
            self.cache.append(pack)
        return loss_, acc_
    
args = parse_args()
args.method = "fednova"
args.k = int(args.num_clients*args.sample_ratio)
setup_seed(args.seed)

path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

# trainer
trainer = FedNovaSerialClientTrainer_(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = FedNovaServerHandler_(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
handler.num_clients = trainer.num_clients
sampler = UniformSampler(args.num_clients)
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
    
    # diversity
    gradient_list = [torch.sub(handler.model_parameters, ele[0]) for ele in full_info]
    diversity = gradient_diversity(gradient_list)
    writer.add_scalar('Metric/Diversity/{}'.format(args.dataset), diversity, t)
    
    norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in gradient_list])
    writer.add_scalar('Metric/MaxNorm/{}'.format(args.dataset), max(norms), t)
    writer.add_scalar('Metric/MinNorm/{}'.format(args.dataset), min(norms), t)
    
    for pack in full_info:
        handler.load(pack)

    t += 1
    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
    
    writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
    writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)

    writer.add_scalar('Test/loss/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Test/accuracy/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, 0,  0, tacc, tloss))