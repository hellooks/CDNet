import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torch.autograd import Function
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_uniform_(m.weight.data)
    else:
        pass

class Siamese(nn.Module):
    def __init__(self,config):
        super(Siamese, self).__init__()
        self.conv1x1 = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv11x11 = nn.Conv2d(3, 32, kernel_size=11, stride=1, padding=5, dilation=3, bias=False)
        self.conv_m1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv_m2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv_m3 = nn.Conv2d(16, 12, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.apply(weights_init)

    def forward_once(self, x):
        upsample = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',align_corners=True)
        x1 = self.conv1x1(x)
        x2 = self.conv11x11(x)
        x = torch.cat([x1, upsample(x2)], dim=1)
        x = self.conv_m3(self.relu(self.conv_m2(self.relu(self.conv_m1(self.relu(x))))))
        return x

    def forward(self, x, y, matrix):
        matrix.to(x.device)
        matrix = torch.abs(matrix)
        matrix = (matrix.t() + matrix) / 2
        output_x = self.forward_once(x)
        output_y = self.forward_once(y)
        output = torch.abs(output_x - output_y).view(output_x.shape[0], output_x.shape[1], -1)
        # bs*48*H*W -> bs*48*P -> bs*P*48 -> bs*p*1*48
        output = output.view(output.shape[0], output.shape[1], -1).transpose(dim0=1, dim1=2).unsqueeze(2)
        score = torch.sqrt(1e-8 + torch.matmul(torch.matmul(output, matrix), output.transpose(dim0=-2, dim1=-1)))
        score=score.squeeze(2)
        score=score.squeeze(2)
        return torch.mean(score, dim=1)
if __name__ == "__main__":
        model = Siamese()
        x = torch.rand(2,3,224, 224)
        y = torch.rand(2,3,224, 224)
        output=model(x,y)
        print(output.shape)