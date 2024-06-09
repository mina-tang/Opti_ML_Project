
from pytorch_optim_training_manager import train_manager
import torch
import torchvision
import torchvision.transforms as transforms
import models
import os
import torch
import torch.optim as optim
import sys
sys.path.append('../')
sys.path.append('spsa/')


class SPSA(optim.Optimizer):
    def __init__(self, params, lr = 0.1, c = 0.1, alpha = 0.6, gamma = 0.101, t = 0):
        defaults = dict(lr = lr, c = c, alpha = alpha, gamma = gamma, t = t)
        super(SPSA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SPSA, self).__setstate__(state)

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            alpha = group['alpha']
            gamma = group['gamma']
            t = group['t']
            for p in group['params']:
                if p.grad is None:
                    continue

                d = p.grad.data
                delta = 2 * torch.randint(0, 2, size = d.size(), dtype = torch.float32) - 1
                
                #Scale
                a = c / (t + 1) ** gamma

                #SPSA step
                p.data.add_(-lr * (a * delta.sign() + alpha) * d)

        return loss
