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

from model import ToyCifarNet, LinearReg
from scipy.special import softmax
import time
from copy import deepcopy

from min_norm_solvers import MinNormSolver, gradient_normalizers

def gradient_diversity(gradients, weights=None):
    if weights is None:
        weights = 1/np.ones(len(gradients))
    else:
        weights = weights/weights.sum()
    # norms = [torch.norm(grad, p=2, dim=0).item() for grad in gradients]
    norms = [torch.norm(grad, p=2, dim=0).item()**2 for grad in gradients]
    d = Aggregators.fedavg_aggregate(gradients)
    d_norm = torch.norm(d, p=2, dim=0).item()
    diversity = np.sqrt(sum(norms)/len(norms))/d_norm
    return diversity

def get_gradient_diversity(gradients, global_gradients):
    norms = [torch.norm(grad, p=2, dim=0).item() for grad in gradients]
    diversity = torch.sqrt(sum(norms)/len(norms)/torch.norm(global_gradients, p=2, dim=0)).item()
    return diversity

class FeedbackSampler:
    def __init__(self, n, probs=None):
        self.name = "uniform"
        self.n = n
        self.p = probs if probs is not None else np.ones(n)/float(n)
        self.explore = [i for i in range(n)]
        random.shuffle(self.explore)
        self.explored = False

    def sample(self, k, startup=0):
        if startup:
            k = self.n
        if len(self.explore) > 0 and not self.explored:
            sampled = self.explore[0:k]
            self.explore = list(set(self.explore) - set(sampled))
            self.last_sampled = sampled, self.p[sampled]
            if len(self.explore)==0:
                self.explored = True
            return np.sort(np.array(sampled))
        else:
            nonzero_entries = sum(self.p > 0)
            if nonzero_entries > k:
                sampled = np.random.choice(self.n, k, p=self.p, replace=False)
            else:
                sampled = np.random.choice(self.n, nonzero_entries, p=self.p, replace=False)
                remains = np.setdiff1d(np.arange(self.n), sampled)
                uniform_sampled = np.random.choice(remains, k-nonzero_entries, replace=False)
                sampled = np.concatenate((sampled, uniform_sampled))
            self.last_sampled = sampled, self.p[sampled]
            return np.sort(sampled)
        
    def update(self, probs, beta=1):
        self.p = (1-beta)*self.p + beta*probs

class UniformSampler:
    def __init__(self, n, probs=None):
        self.name = "uniform"
        self.n = n
        self.p = probs if probs is not None else np.ones(n)/float(n)

    def sample(self, k):
        if k == self.n:
            self.last_sampled = np.arange(self.n), self.p
            return np.arange(self.n)
        else:
            sampled = np.random.choice(self.n, k, p=self.p, replace=False)
            self.last_sampled = sampled, self.p[sampled]
            return np.sort(sampled)
        
    def update(self, probs, beta=1):
        self.p = (1-beta)*self.p + beta*probs
        print(self.p)

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
    
class FedAWARE_Projector():
    def __init__(self, n, alpha, model_parameters) -> None:
        self.n = n
        self.momentum = [torch.zeros_like(model_parameters) for _ in range(n)]
        self.alpha = alpha
        self.solver = MinNormSolver
        self.feedback = None

    def momentum_update(self, gradients, indices):
        for grad, idx in zip(gradients, indices):
            self.momentum[idx] = (1-self.alpha)*self.momentum[idx] + self.alpha*grad
    
    def projection(self, va, vb):
        # project va to the direction of vb
        d_proj = (torch.dot(va, vb) / torch.dot(vb, vb)) * vb
        return d_proj

    def compute_estimates(self):
        norms = [torch.norm(grad, p=2, dim=0).item() for grad in self.momentum]
        norms = np.array([1 if item==0 else item for item in norms])
        momentum = [self.momentum[i]/n for i, n in enumerate(norms)]
        sol = self.compute_lambda(momentum)
        self.feedback=sol
        gdm_estimates = Aggregators.fedavg_aggregate(momentum, sol)
        return gdm_estimates

    def compute_lambda(self, vectors):
        sol, val = self.solver.find_min_norm_element_FW(vectors)
        print("FW solver - val {} density {}".format(val, (sol>0).sum()))
        # self.stats["count"] += sol>0
        # sol = sol/sol.sum()
        assert sol.sum()-1 < 1e-5
        return sol
    

class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def agnews_evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy.
    
    Returns:
        (loss.sum, acc.avg)
    """
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            label, input_ids, mask = data['label'], data["input_ids"], data["attention_mask"]
            input_ids = torch.Tensor(input_ids)
            mask = torch.Tensor(mask)
            label = torch.Tensor(label).to(dtype=torch.long)

            input_ids = input_ids.to(device=gpu, dtype=torch.long)
            mask = torch.Tensor(mask).to(device=gpu, dtype=torch.long)
            target = label.to(device=gpu, dtype=torch.long)

            outputs = model(input_ids, mask)["logits"]
            loss = criterion(outputs, target)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(target)).item(), len(target))

    return loss_.sum, acc_.avg
