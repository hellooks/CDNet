import torch
import numpy as np
import torch.optim as optim
def createLossAndOptimizer(net, learning_rate, loss, scheduler_step, scheduler_gamma, cube):
    loss = LossFunc(loss)
    optimizer = optim.Adam([{'params': net.parameters(), 'lr':learning_rate},
                            {'params': cube,'lr':learning_rate}], lr = learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma= scheduler_gamma)
    return loss, optimizer, scheduler

def createLossAndOptimizer_ablation(net, learning_rate, loss, scheduler_step, scheduler_gamma):
    loss = LossFunc(loss)
    optimizer = optim.Adam([{'params': net.parameters(), 'lr':learning_rate}], lr = learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma= scheduler_gamma)
    return loss, optimizer, scheduler

class LossFunc(torch.nn.Module):
    def __init__(self, lossname='mse',margin=1.0):
        super(LossFunc, self).__init__()
        self.margin = margin
        self.lossname = lossname
        if self.lossname not in ['mae', 'mse', 'mae_mse']:
            raise AssertionError ("Please Specify Valid Loss Fuction!!!!!")

    def forward(self, output, label):
        output = torch.squeeze(output)
        if self.lossname=='mae':
            return torch.abs(output - label).mean()  
        elif self.lossname=='mse':
            return torch.mean((output - label) ** 2)  
        elif self.lossname == 'mae_mse':
            return torch.abs(output - label).mean() + torch.mean((output - label) ** 2)
        else:
            raise AssertionError ("Please Specify Valid Loss Fuction!!!!!")