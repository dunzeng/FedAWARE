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
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.algorithm.scaffold import ScaffoldSerialClientTrainer, ScaffoldServerHandler
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from settings import get_settings, parse_args, get_logs, get_heterogeneity
from utils import UniformSampler, gradient_diversity

class ScaffoldServerHandler_(ScaffoldServerHandler):
    def setup_optim(self, sampler, args):
        super().setup_optim(args.glr)
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

class ScaffoldSerialClientTrainer_(SGDSerialClientTrainer):
    def setup_optim(self, epochs, batch_size, lr):
        super().setup_optim(epochs, batch_size, lr)
        self.cs = [None for _ in range(self.num_clients)]

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        global_c = payload[1]
        loss_, acc_ = AverageMeter(), AverageMeter()
        for id in id_list:
            dataset = self.dataset.get_dataset(id)
            self.batch_size, self.epochs = get_heterogeneity(args, len(dataset))
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            pack = self.train(id, model_parameters, global_c, data_loader, loss_, acc_)
            self.cache.append(pack)
        return loss_, acc_
    
    def train(self, id, model_parameters, global_c, train_loader, loss_, acc_):
        self.set_model(model_parameters)
        frz_model = model_parameters

        if self.cs[id] is None:
            self.cs[id] = torch.zeros_like(model_parameters)

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()

                grad = self.model_gradients
                grad = grad - self.cs[id] + global_c
                idx = 0
                for parameter in self._model.parameters():
                    layer_size = parameter.grad.numel()
                    shape = parameter.grad.shape
                    #parameter.grad = parameter.grad - self.cs[id][idx:idx + layer_size].view(parameter.grad.shape) + global_c[idx:idx + layer_size].view(parameter.grad.shape)
                    parameter.grad.data[:] = grad[idx:idx+layer_size].view(shape)[:]
                    idx += layer_size

                self.optimizer.step()

                _, predicted = torch.max(output, 1)
                loss_.update(loss.item())
                acc_.update(torch.sum(predicted.eq(target)).item(), len(target))

        dy = self.model_parameters - frz_model
        dc = -1.0 / (self.epochs * len(train_loader) * self.lr) * dy - global_c
        self.cs[id] += dc
        return [dy, dc]

args = parse_args()
args.method = "scaffold"
args.k = int(args.num_clients*args.sample_ratio)
setup_seed(args.seed)


path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

# trainer
trainer = ScaffoldSerialClientTrainer_(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = ScaffoldServerHandler_(model=model,
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
    gradient_list = [ele[0] for ele in full_info]
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