import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
EPS = 1e-8

class crossentropy_vector(nn.Module):
    def __init__(self):
        super(crossentropy_vector, self).__init__()

    def forward(self, y_pred, y_true):
        f=nn.Softmax(dim=1)
        loss = - (f(y_true) * torch.log(f(y_pred)+1e-6) + (1-f(y_true)) * torch.log(1-f(y_pred)+1e-6))
        return (torch.sum(loss, dim=1)).mean()

def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

class FocalBCELoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# https://github.com/TakaraResearch/Pytorch-1D-Wasserstein-Statistical-Loss/blob/master/pytorch_stats_loss.py
def wasserstein_1d(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(cdf_loss(tensor_a,tensor_b,p=1))


def cdf_loss(tensor_a,tensor_b,p=2):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss



# class JSDLoss(nn.Module):

#     def __init__(self):
#         super(JSDLoss,self).__init__()

#     def forward(self, batch_size, f_real, f_synt):
#         assert f_real.size()[1] == f_synt.size()[1]

#         f_num_features = f_real.size()[1]
#         identity = autograd.Variable(torch.eye(f_num_features)*0.1, requires_grad=False)

#         # if use_cuda:
#         identity = identity.cuda()

#         f_real_mean = torch.mean(f_real, 0, keepdim=True)
#         f_synt_mean = torch.mean(f_synt, 0, keepdim=True)

#         dev_f_real = f_real - f_real_mean.expand(batch_size,f_num_features)
#         dev_f_synt = f_synt - f_synt_mean.expand(batch_size,f_num_features)

#         f_real_xx = torch.mm(torch.t(dev_f_real), dev_f_real)
#         f_synt_xx = torch.mm(torch.t(dev_f_synt), dev_f_synt)

#         cov_mat_f_real = (f_real_xx / batch_size) - torch.mm(f_real_mean, torch.t(f_real_mean)) + identity
#         cov_mat_f_synt = (f_synt_xx / batch_size) - torch.mm(f_synt_mean, torch.t(f_synt_mean)) + identity

#         cov_mat_f_real_inv = torch.inverse(cov_mat_f_real)
#         cov_mat_f_synt_inv = torch.inverse(cov_mat_f_synt)

#         temp1 = torch.trace(torch.add(torch.mm(cov_mat_f_synt_inv, cov_mat_f_real), torch.mm(cov_mat_f_real_inv, cov_mat_f_synt)))
#         temp1 = temp1.view(1,1)
#         temp2 = torch.mm(torch.mm((f_synt_mean - f_real_mean), (cov_mat_f_synt_inv + cov_mat_f_real_inv)), torch.t(f_synt_mean - f_real_mean))
#         loss_g = torch.add(temp1, temp2).mean()

#         return loss_g
class JSDLoss(nn.Module):

    def __init__(self):
        super(JSDLoss,self).__init__()

    def forward(self, batch_size, f_real, f_synt):
        assert f_real.size()[1] == f_synt.size()[1]

        f_num_features = f_real.size()[1]
        identity = autograd.Variable(torch.eye(f_num_features)*0.1, requires_grad=False)

        # if use_cuda:
        identity = identity.cuda()

        f_real_mean = torch.mean(f_real, 0, keepdim=True)
        f_synt_mean = torch.mean(f_synt, 0, keepdim=True)

        dev_f_real = f_real - f_real_mean.expand(batch_size,f_num_features)
        dev_f_synt = f_synt - f_synt_mean.expand(batch_size,f_num_features)

        f_real_xx = torch.mm(torch.t(dev_f_real), dev_f_real)
        f_synt_xx = torch.mm(torch.t(dev_f_synt), dev_f_synt)

        cov_mat_f_real = (f_real_xx / batch_size) - torch.mm(f_real_mean, torch.t(f_real_mean)) + identity
        cov_mat_f_synt = (f_synt_xx / batch_size) - torch.mm(f_synt_mean, torch.t(f_synt_mean)) + identity

        cov_mat_f_real_inv = torch.inverse(cov_mat_f_real).float()
        cov_mat_f_synt_inv = torch.inverse(cov_mat_f_synt).float()

        temp1 = torch.trace(torch.add(torch.mm(cov_mat_f_synt_inv.float(), cov_mat_f_real.float()), torch.mm(cov_mat_f_real_inv.float(), cov_mat_f_synt.float())))
        temp1 = temp1.view(1,1)
        temp2 = torch.mm(torch.mm((f_synt_mean.double() - f_real_mean.double()), (cov_mat_f_synt_inv.double() + cov_mat_f_real_inv.double())), torch.t(f_synt_mean.double() - f_real_mean.double()))
        loss_g = torch.add(temp1, temp2).mean()

        return loss_g
class mmd_loss(nn.Module):
    def __init__(self):
        super(mmd_loss, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)#将source,target按列方向合并
        #将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0-total1)**2).sum(2)
        #调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        #高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        #得到最终的核矩阵
        return sum(kernel_val)#/len(kernel_val)

    def forward(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
        kernels = self.guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        #根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss#因为一般都是n==m，所以L矩阵一般不加入计算

class L2_Loss(torch.nn.Module):

    def __init__(self):
        super(L2_Loss, self).__init__()

    def forward(self, y, y_var, g, g_var):
        k = g.size()[1]
        y_var = y_var + EPS
        g_var = g_var + EPS
        y_std = torch.sqrt(y_var)
        g_std = torch.sqrt(g_var)

        log_c = -(torch.sum(torch.log(g_std), dim=-1, keepdim=True) + torch.log(y_std))
        c1 = 0.5 * (torch.sum(1 / g_var, dim=-1, keepdim=True) + (1 / y_var))
        c2 = (torch.sum(g / g_var, dim=-1, keepdim=True) + y / y_var)
        c3 = 0.5 * (torch.sum(g ** 2 / g_var, dim=-1, keepdim=True) + y ** 2 / y_var)

        loss = -(log_c.mean() - 0.5 * torch.log(c1).mean() + (c2 ** 2 / (4 * c1)).mean() - c3.mean()) / k

        return loss
class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class Binary_Loss(torch.nn.Module):

    def __init__(self):
        super(Binary_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, p, g, alpha, beta):
        log_a = g * torch.log(alpha) + (1 - g) * torch.log(1 - alpha)
        log_a = torch.sum(log_a, dim=1, keepdim=True)
        a = torch.exp(log_a)

        log_b = (1 - g) * torch.log(beta) + g * torch.log(1 - beta)
        log_b = torch.sum(log_b, dim=1, keepdim=True)
        b = torch.exp(log_b)
        bin_loss = -torch.mean(torch.log(a * p + b * (1 - p))) / (g.size()[1])
        # cr_loss = (self.ce_loss(cr1, d1.squeeze(dim=1)) + self.ce_loss(cr2, d2.squeeze(dim=1)))/2
        # loss = bin_loss + cr_loss
        # return loss,bin_loss,cr_loss
        return bin_loss

class Cr_Loss(torch.nn.Module):
    def __init__(self):
        super(Cr_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, cr1, cr2, d1, d2):
        cr_loss = (self.ce_loss(cr1, d1.squeeze(dim=1)) + self.ce_loss(cr2, d2.squeeze(dim=1)))/2
        return cr_loss

class KLD_loss(torch.nn.Module):

    def __init__(self):
        super(KLD_loss, self).__init__()
    def forward(self, mean, var):
        # KLD = -0.5*torch.sum(1+var-mean.pow(2)-var.exp())/(mean.size()[1]*mean.size()[0])
        # KLD = -0.5*torch.sum(1+var-mean.pow(2)-var.exp())/(mean.size()[0])
        KLD = torch.mean(-0.5 * torch.sum(1 + var - mean ** 2 - var.exp(), dim = 1), dim = 0)

        return KLD
        
class Refined_Binary_Loss(torch.nn.Module):

    def __init__(self, eg=True):
        super(Refined_Binary_Loss, self).__init__()
        self.eg = eg

    def forward(self, p, g1, g2, alpha, beta):

        n1, k1 = g1.size()
        log_a = g1 * torch.log(alpha) + (1 - g1) * torch.log(1 - alpha)
        log_a = torch.sum(log_a, dim=1)
        a = torch.exp(log_a)

        log_b = (1 - g1) * torch.log(beta) + g1 * torch.log(1 - beta)
        log_b = torch.sum(log_b, dim=1)
        b = torch.exp(log_b)

        loss1 = -torch.mean(torch.log(a * p[:n1, 0] + b * p[:n1, 1]) -
                            torch.log(torch.sum(p[:n1, :], dim=1))) / k1

        n2, k2 = g2.size()
        log_a = g2 * torch.log(alpha[:, -k2:]) + (1 - g2) * torch.log(1 - alpha[:, -k2:])
        log_a = torch.sum(log_a, dim=1)
        a = torch.exp(log_a)

        log_b = (1 - g2) * torch.log(beta[:, -k2:]) + g2 * torch.log(1 - beta[:, -k2:])
        log_b = torch.sum(log_b, dim=1)
        b = torch.exp(log_b)

        loss2 = -torch.mean(torch.log(a * p[n1:, 0] + b * p[n1:, 1]) -
                            torch.log(torch.sum(p[n1:, :], dim=1)))

        if self.eg:
            reg = -(p[n1:, 0] * torch.log(p[n1:, 0]) + p[n1:, 1] * torch.log(p[n1:, 1])) / \
                    torch.sum(p[n1:, :], dim=1) + torch.log(torch.sum(p[n1:, :], dim=1))
            reg = torch.mean(reg)

            loss2 = (loss2 + reg * (k1-k2)) / k1
        else:
            loss2 = loss2 / k2
        return (loss1 + loss2) / 2


if __name__ == "__main__":
    net = models.vgg19()
    print(net)
    print(net.features[0])

