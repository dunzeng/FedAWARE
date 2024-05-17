import sys
sys.path.append("./")
from tqdm import tqdm
import sys
import torch
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer

from fedlab.utils.functional import AverageMeter
from settings import get_heterogeneity


class SerialClientTrainer(SGDSerialClientTrainer):
    def setup_optim(self, epochs, batch_size, lr, args=None):
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
            self.batch_size, self.epochs = get_heterogeneity(self.args, len(dataset))
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
