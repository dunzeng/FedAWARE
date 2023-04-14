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
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer, FedAvgServerHandler
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from settings import get_settings, get_logs, parse_args, get_heterogeneity
from mode import UniformSampler, gradient_diversity


class FedOptServerHandler_(FedAvgServerHandler):
    def setup_optim(self, sampler, args):
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler

        
        self.args = args
        self.lr = args.glr
        self.k = args.k

        # momentum
        self.beta1 = self.args.beta1
        self.beta2 = self.args.beta2
        self.option = self.args.option
        self.tau = self.args.tau
        self.momentum = torch.zeros_like(self.model_parameters)
        self.vt = torch.zeros_like(self.model_parameters)
        assert self.option in ["adagrad", "yogi", "adam"]

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
    
    def global_update(self, buffer):
        gradient_list = [torch.sub(ele[0], self.model_parameters) for ele in buffer]
        indices, _ = self.sampler.last_sampled
        delta = Aggregators.fedavg_aggregate(gradient_list, self.args.weights[indices])
        self.momentum = self.beta1*self.momentum + (1-self.beta1)*delta

        delta_2 = torch.pow(delta, 2)
        if self.option == "adagrad":
            self.vt += delta_2
        elif self.option == "yogi":
            self.vt = self.vt - (1-self.beta2)*delta_2*torch.sign(self.vt-delta_2)
        else:
            # adam
            self.vt = self.beta2*self.vt + (1-self.beta2)*delta_2

        serialized_parameters = self.model_parameters + self.lr*self.momentum/(torch.sqrt(self.vt) + self.tau)
        self.set_model(serialized_parameters)



args = parse_args()
args.method = "fedopt"
args.k = int(args.num_clients*args.sample_ratio)
setup_seed(args.seed)

path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

# trainer
trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = FedOptServerHandler_(model=model,
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