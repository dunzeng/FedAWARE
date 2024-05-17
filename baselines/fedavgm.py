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

from utils import FedAvgSerialClientTrainer, UniformSampler, gradient_diversity, get_gradient_diversity, FeedbackSampler

from torch.utils.tensorboard import SummaryWriter
from settings import get_settings, get_heterogeneity, get_logs, parse_args

from utils import FedAWARE_Projector, agnews_evaluate
from agnews_dataset import get_AGNEWs_testloader

class FedAvgMServerHandler(SyncServerHandler):
    def setup_optim(self, sampler, args):  
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler

        self.args = args
        self.lr = args.glr
        self.k = args.k
        self.method = args.method
        self.momentum = torch.zeros_like(self.model_parameters)
        self.beta = args.fedm_beta

        if self.args.projection:
            self.projector = FedAWARE_Projector(self.n, self.args.alpha, self.model_parameters)

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
        self.momentum = self.beta*self.momentum + estimates
        estimates = self.momentum

        if self.args.projection:
            print("projecting...")
            self.projector.momentum_update(gradient_list, indices)
            if self.sampler.explored:
                gdm_estimates = self.projector.compute_estimates()
                estimates = self.projector.projection(estimates, gdm_estimates)

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

    def setup_optim(self, epochs, batch_size, lr, args):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.args = args
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        loss_, acc_ = AverageMeter(), AverageMeter()
        
        for id in tqdm(id_list):
            dataset = self.dataset.get_dataset(id)
            self.batch_size, self.epochs = get_heterogeneity(args, len(dataset))
            # data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, loss_, acc_)
            self.cache.append(pack)
        return loss_, acc_

    def train(self, model_parameters, train_loader, loss_, acc_): 
        self.set_model(model_parameters)
        self._model.train()
        
        if self.args.dataset == "agnews":
            for _ in range(self.epochs):
                for data in train_loader:
                    if self.cuda:
                        label, input_ids, mask = data['label'], data["input_ids"], data["attention_mask"]
                        input_ids = torch.Tensor(input_ids)
                        mask = torch.Tensor(mask)
                        label = torch.Tensor(label).to(dtype=torch.long)

                        input_ids = input_ids.to(device=self.device, dtype=torch.long)
                        mask = torch.Tensor(mask).to(device=self.device, dtype=torch.long)
                        target = label.to(device=self.device, dtype=torch.long)

                    output = self.model(input_ids, mask)["logits"]
                    loss = self.criterion(output, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    _, predicted = torch.max(output, 1)
                    loss_.update(loss.item())
                    acc_.update(torch.sum(predicted.eq(target)).item(), len(target))
        else:
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


args = parse_args()
args.method = "fedavgm"
args.k = int(args.num_clients*args.sample_ratio)
setup_seed(args.seed)

path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights


probs = np.ones(args.num_clients)/args.num_clients
sampler = FeedbackSampler(args.num_clients, probs)

trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr, args)
trainer.setup_dataset(dataset)

# server-sampler
handler = FedAvgMServerHandler(model=model,
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

    norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in gradient_list])
    writer.add_scalar('Metric/MaxNorm/{}'.format(args.dataset), max(norms), t)
    writer.add_scalar('Metric/MinNorm/{}'.format(args.dataset), min(norms), t)

    for pack in full_info:
        handler.load(pack)

    if t==0 or (t+1)%args.freq == 0:
        if args.dataset == "agnews":
            gen_test_loader = get_AGNEWs_testloader()
            tloss, tacc = agnews_evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
        else:
            tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
        
        writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
        writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)

        writer.add_scalar('Test/loss/{}'.format(args.dataset), tloss, t)
        writer.add_scalar('Test/accuracy/{}'.format(args.dataset), tacc, t)

        print("Round {}, Loss {:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, train_loss.avg,  train_acc.avg, tacc, tloss))
    t += 1

writer.close()