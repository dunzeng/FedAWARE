
import numpy as np
import json
import os

from tqdm import tqdm
import torch
from torch import nn
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import evaluate, AverageMeter
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer


from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer

from torch.utils.tensorboard import SummaryWriter
from min_norm_solvers import MinNormSolver
from utils import UniformSampler, gradient_diversity, FeedbackSampler, FedAWARE_Projector, agnews_evaluate
from settings import get_settings, get_logs, parse_args, get_heterogeneity
from agnews_dataset import get_AGNEWs_testloader


def projection(va, vb):
    # project va to the direction of vb
    d_proj = (torch.dot(va, vb) / torch.dot(vb, vb)) * vb
    return d_proj

class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def setup_optim(self, epochs, batch_size, lr, momentum, args=None):
        self.args = args
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)

        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        loss_ = AverageMeter()
        acc_ = AverageMeter()

        for id in tqdm(id_list):
            dataset = self.dataset.get_dataset(id)
            self.batch_size, self.epochs = get_heterogeneity(args, len(dataset))
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

        self.warmup = False
        self.t=0

        self.projector = FedAWARE_Projector(self.n, self.args.alpha, self.model_parameters)

    def momentum_update(self, gradients, indices):
        for grad, idx in zip(gradients, indices):
            self.momentum[idx] = (1-self.alpha)*self.momentum[idx] + self.alpha*grad
        
    def compute_lambda(self, vectors):
        sol, val = self.solver.find_min_norm_element_FW(vectors)
        print("FW solver - val {} density {}".format(val, (sol>0).sum()))
        assert sol.sum()-1 < 1e-5
        return sol
    
    @property
    def num_clients_per_round(self):
        return self.round_clients
           
    def sample_clients(self, k, startup=0):
        clients = self.sampler.sample(k, startup)
        
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients
        
    def global_update(self, buffer):
        gradient_list = [torch.sub(self.model_parameters, ele[0]) for ele in buffer]
        indices, _ = self.sampler.last_sampled

        estimates = Aggregators.fedavg_aggregate(gradient_list, self.args.weights[indices])
        self.projector.momentum_update(gradient_list, indices)
            
        if self.sampler.explored:
            estimates = self.projector.compute_estimates()
            # self.sampler.update(self.projector.feedback) # acceleration trick
            if self.args.projection: 
                d_fedavg = Aggregators.fedavg_aggregate(self.projector.momentum, self.args.weights)
                estimates = self.projector.projection(d_fedavg, estimates)
        
        serialized_parameters = self.model_parameters - self.lr*estimates
        self.set_model(serialized_parameters)
        self.t += 1

args = parse_args()
args.method = "fedaware"
args.k = int(args.num_clients*args.sample_ratio)
setup_seed(args.seed)

path = get_logs(args)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

model, dataset, weights, gen_test_loader = get_settings(args)
args.weights = weights

# client-trainer
trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr, args.local_momentum, args)
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
    # server side
    broadcast = handler.downlink_package

    if t == 0:
        sampled_clients = handler.sample_clients(args.k, args.startup) # zero-error momentum initialization
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

    if t%args.freq == 0:
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
        torch.save(handler.stats, os.path.join(path, "stats.pkl"))
    t += 1

writer.close()
torch.save(handler._model.state_dict(), os.path.join(path, "model.pth"))