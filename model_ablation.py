import torch
import math
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from torchvision import models
from cnn_finetune import make_model

class BCNN(nn.Module):
    def __init__(self, thresh=1e-8, is_vec=True, input_dim=64):
        super(BCNN, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.output_dim = input_dim * input_dim

    def _bilinearpool(self, x):
        batchSize, dim, h, w = x.data.shape
        x = x.reshape(batchSize, dim, h * w)
        x = 1. / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        x = self._bilinearpool(x)
        x = self._signed_sqrt(x)
        if self.is_vec:
            x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        return x
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)
    elif classname.find('Gdn1d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)
    else:
        pass
    
class Siamese(nn.Module):
    def __init__(self, backbone = 'resnet18'):
        super(Siamese, self).__init__()
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.prenet = nn.Sequential(*list(model.children())[:-1])
            n_dim = 512
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=True)
            self.prenet = nn.Sequential(*list(model.children())[:-1])
            n_dim = 512
        elif backbone == 'resnext101':
            model =make_model( 'resnext101_64x4d', num_classes=1000, pretrained=True)
            self.prenet = nn.Sequential(*list(model.children())[:-1])
            n_dim = 2048
        self.fc_layer = nn.Linear(n_dim, 1)
        self.fc_layer.apply(weights_init)

    def forward(self, x, y):
        x = self.prenet(x)
        y = self.prenet(y)
        comb = torch.abs(x.view(x.shape[0], -1)-y.view(y.shape[0], -1))
        return self.fc_layer(comb)

if __name__ == '__main__':
    model = Siamese(backbone='resnet18')
    x1 = torch.rand(8, 3, 128, 128)
    x2 = torch.rand(8, 3, 128, 128)
    print(model)
    out = model(x1, x2)
    print(out)
    print(out.shape)