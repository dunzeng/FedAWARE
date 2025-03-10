
import numpy as np
import torch
from fedlab.utils.aggregator import Aggregators
from min_norm_solvers import MinNormSolver
from tqdm import tqdm

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
