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
from fedlab.contrib.algorithm.fedprox import FedProxSerialClientTrainer, FedProxServerHandler
import time
from torch.utils.data import DataLoader

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from settings import get_settings, get_logs, parse_args, get_heterogeneity
from utils import UniformSampler, gradient_diversity, get_gradient_diversity


class FedProxSerialClientTrainer_(FedProxSerialClientTrainer):
    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        loss_, acc_ = AverageMeter(), AverageMeter()
        
        for id in tqdm(id_list):
            dataset = self.dataset.get_dataset(id)
            self.batch_size, self.epochs = get_heterogeneity(args, len(dataset))
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            #data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, self.mu, loss_, acc_)
            self.cache.append(pack)
        return loss_, acc_

    def train(self, model_parameters, train_loader, mu,  loss_, acc_) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        self.set_model(model_parameters)
        frz_model = deepcopy(self._model)
        frz_model.eval()

        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(
                        self.device)

                output = self._model(data)
                l1 = self.criterion(output, target)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * mu * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output, 1)
                loss_.update(loss.item())
                acc_.update(torch.sum(predicted.eq(target)).item(), len(target))

        return [self.model_parameters]

class FedProxServerHandler_(FedProxServerHandler):
    def setup_optim(self, sampler, args):
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler

        self.args = args
        self.lr = args.glr
        self.k = args.k
    
    @property
    def num_clients_per_round(self):
        return self.round_clients
           
    def sample_clients(self, k):
        clients = self.sampler.sample(k)
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients

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


args = parse_args()
args.method = "fedprox"
args.k = int(args.num_clients*args.sample_ratio)

setup_seed(args.seed)
# format
path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

# trainer
trainer = FedProxSerialClientTrainer_(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr, args.mu)
trainer.setup_dataset(dataset)

# server-sampler
handler = FedProxServerHandler_(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
handler.num_clients = trainer.num_clients
sampler = UniformSampler(args.num_clients)
handler.setup_optim(sampler, args)

t = 0
while handler.if_stop is False:
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

    if t==0 or (t+1)%args.freq == 0:
        tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
        
        writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
        writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)

        writer.add_scalar('Test/loss/{}'.format(args.dataset), tloss, t)
        writer.add_scalar('Test/accuracy/{}'.format(args.dataset), tacc, t)

        print("Round {}, Loss {:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, 0,  0, tacc, tloss))
    t += 1

writer.close()