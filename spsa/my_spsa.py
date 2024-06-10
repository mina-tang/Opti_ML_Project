
from pytorch_optim_training_manager import train_manager
import torch
import torchvision
import torchvision.transforms as transforms
import models
import os
import torch
import torch.optim as optim
import sys
sys.path.append("../")
sys.path.append("spsa/")


class SPSA(optim.Optimizer):
    def __init__(self, params, lr = 0.1, c = 0.1, alpha = 0.602, gamma = 0.101):
        defaults = dict(lr = lr, c = c, alpha = alpha, gamma = gamma, t = 0)
        super(SPSA, self).__init__(params, defaults)

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            c = group["c"]
            alpha = group["alpha"]
            gamma = group["gamma"]
            t = group["t"] + 1
            group["t"] = t

            for p in group["params"]:
                if p.grad is None:
                    continue
                perturbation = torch.randint(0, 2, size = p.data.size(), dtype=p.data.dtype, device = p.data.device) * 2 - 1
            
                a_t = lr / (t ** alpha)
                c_t = c / (t ** gamma)

                p.data.add_(c_t * perturbation)
                positive_loss = closure()
                
                p.data.sub_(2 * c_t * perturbation)
                negative_loss = closure()
                
                p.data.add_(c_t * perturbation)

                g_hat = (positive_loss - negative_loss) / (2.0 * c_t * perturbation)

                p.data.add_(-a_t * g_hat)       
        return loss
