import sys
sys.path.append("./")

import numpy as np
import json
import os

import random
from tqdm import tqdm
import sys
import torch
from torch import nn, softmax
from torch.utils.data import DataLoader

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer

from fedlab.utils.functional import evaluate, setup_seed, AverageMeter
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer
from fedlab.contrib.algorithm.feddyn import FedDynSerialClientTrainer, FedDynServerHandler


from utils import FedAvgSerialClientTrainer, UniformSampler, solver, gradient_diversity, get_gradient_diversity

from torch.utils.tensorboard import SummaryWriter

from min_norm_solvers import MinNormSolver, gradient_normalizers
from settings import get_settings, get_heterogeneity, get_logs, parse_args


    
class FedDynSerialClientTrainer_(FedDynSerialClientTrainer):
    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        loss_, acc_ = AverageMeter(), AverageMeter()
        
        for id in tqdm(id_list):
            dataset = self.dataset.get_dataset(id)
            self.batch_size, self.epochs = get_heterogeneity(args, len(dataset))
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            #data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, data_loader, loss_, acc_)
            self.cache.append(pack)
        return loss_, acc_
    
    def train(self, id, model_parameters, train_loader, loss_, acc_):
        if self.L[id] is None:
            self.L[id] = torch.zeros_like(model_parameters)

        L_t = self.L[id]
        frz_parameters = model_parameters

        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                l1 = self.criterion(output, target)
                l2 = torch.dot(L_t, self.model_parameters)
                l3 = torch.sum(torch.pow(self.model_parameters - frz_parameters,2))
                
                loss = l1 - l2 + 0.5 * self.alpha * l3

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output, 1)
                loss_.update(loss.item())
                acc_.update(torch.sum(predicted.eq(target)).item(), len(target))
                
        self.L[id] = L_t - self.alpha * (self.model_parameters-frz_parameters)
        return [self.model_parameters]

args = parse_args()
args.method = "feddyn"
args.k = int(args.num_clients*args.sample_ratio)
setup_seed(args.seed)

path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

probs = np.ones(args.num_clients)/args.num_clients
sampler = UniformSampler(args.num_clients, probs)
    
trainer = FedDynSerialClientTrainer_(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr, args.alpha_dyn)
trainer.setup_dataset(dataset)

# server-sampler
handler = FedDynServerHandler(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
    
handler.num_clients = trainer.num_clients
handler.setup_optim(args.alpha_dyn)

t = 0

while handler.if_stop is False:
    # server side
    broadcast = handler.downlink_package
    sampled_clients = sampler.sample(args.k)

    # client side
    train_loss, train_acc = trainer.local_process(broadcast, sampled_clients)
    full_info = trainer.uplink_package
    
    gradient_list = [torch.sub(handler.model_parameters, ele[0]) for ele in full_info]
    diversity = gradient_diversity(gradient_list)
    writer.add_scalar('Metric/Diversity/{}'.format(args.dataset), diversity, t)

    norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in gradient_list])
    writer.add_scalar('Metric/MaxNorm/{}'.format(args.dataset), max(norms), t)
    writer.add_scalar('Metric/MinNorm/{}'.format(args.dataset), min(norms), t)

    for pack in full_info:
        handler.load(pack)

    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
    
    writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
    writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)

    writer.add_scalar('Test/loss/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Test/accuracy/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, train_loss.avg,  train_acc.avg, tacc, tloss))
    t += 1