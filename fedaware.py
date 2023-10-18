
import numpy as np
import json
import os
import argparse
import random

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from fedlab.utils import functional as F
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu, AverageMeter

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer


from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer

from torch.utils.tensorboard import SummaryWriter
from min_norm_solvers import MinNormSolver, gradient_normalizers

from utils import UniformSampler, gradient_diversity, FeedbackSampler, get_gradient_diversity

from settings import get_settings, get_logs, parse_args, get_heterogeneity

from fedlab.utils import SerializationTool

class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def setup_optim(self, epochs, batch_size, lr, optim='sgd'):
        super().setup_optim(epochs, batch_size, lr)

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


class Server_MomentumGradientCache(SyncServerHandler):
    def setup_optim(self, sampler, alpha, args):  
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler
        self.lr = args.glr

        self.args = args

        self.momentum = [torch.zeros_like(self.model_parameters) for _ in range(self.n)]
        self.alpha = alpha
        self.solver = MinNormSolver
        self.stats = {"count":np.zeros(self.n)}

    def momentum_update(self, gradients, indices):
        for grad, idx in zip(gradients, indices):
            self.momentum[idx] = (1-self.alpha)*self.momentum[idx] + self.alpha*grad
        
        # norms = np.max((np.array([torch.norm(grad, p=2, dim=0).item() for grad in self.momentum])/self.C, np.ones_like(self.num_clients)), axis=0)
        norms = [torch.norm(grad, p=2, dim=0).item() for grad in self.momentum]
        norms = np.array([1 if item==0 else item for item in norms])
        
        # norm_momentum = norms
        norm_momentum = [self.momentum[i]/n for i, n in enumerate(norms)]
        sol, val = self.solver.find_min_norm_element_FW(norm_momentum)
        print("FW solver - val {} density {}, \n lambda: {}".format(val, (sol>0).sum(), str(sol)))
        self.stats["count"] += sol>0
        # sol = sol/sol.sum()
        assert sol.sum()-1 < 1e-5
        return sol, norm_momentum
    
    @property
    def num_clients_per_round(self):
        return self.round_clients
           
    def sample_clients(self, k, startup=0):
        clients = self.sampler.sample(k, startup)
        
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients
        
    def global_update(self, buffer):
        # print("Theta {:.4f}, Ws {}".format(self.theta, self.ws))
        gradient_list = [torch.sub(self.model_parameters, ele[0]) for ele in buffer]
        indices, _ = self.sampler.last_sampled
        
        if self.sampler.explored:
            sol, norm_momentum = self.momentum_update(gradient_list, indices)
            self.sampler.update(sol) # feedback
            estimates = Aggregators.fedavg_aggregate(norm_momentum, sol)

            serialized_parameters = self.model_parameters - self.lr*estimates
            SerializationTool.deserialize_model(self._model, serialized_parameters)
        else:
            for grad, idx in zip(gradient_list, indices):
                self.momentum[idx] = (1-self.alpha)*self.momentum[idx] + self.alpha*grad
            parameters = [ele[0] for ele in buffer]
            aggregated_parameters = Aggregators.fedavg_aggregate(parameters, args.weights[indices])
            SerializationTool.deserialize_model(self._model, aggregated_parameters)
        # indices = np.arange(args.num_clients)
        # norms = np.array([torch.norm(self.momentum[i], p=2, dim=0).item() for i in indices])
        # norm_momentum = [self.momentum[i]/norms[i] for i in indices]

args = parse_args()
args.method = "ours"
args.k = int(args.num_clients*args.sample_ratio)
setup_seed(args.seed)

path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

# client-trainer
trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = Server_MomentumGradientCache(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
    
probs = np.ones(args.num_clients)/args.num_clients
sampler = FeedbackSampler(args.num_clients, probs)
handler.num_clients = trainer.num_clients
handler.setup_optim(sampler, args.alpha, args)

t = 0
while handler.if_stop is False:
    print("running..")
    # server side
    broadcast = handler.downlink_package

    if t == 0:
        sampled_clients = handler.sample_clients(args.k, args.startup)
    else:
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

    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader)
    
    writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
    writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)

    writer.add_scalar('Test/loss/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Test/accuracy/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}, Accuracy: {:.4f}, Generalization: {:.4f}-{:.4f}".format(t, train_loss.avg,  train_acc.avg, tacc, tloss))
    torch.save(handler.stats, os.path.join(path, "stats.pkl"))
    t += 1
