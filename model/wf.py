import torch
import xitorch
import torchdiffeq
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, k, x):
        tt = torch.ones_like(x[:, :1, :, :]) * k
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.tanh = nn.Tanh()
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, k, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.tanh(out)
        out = self.conv1(k, out)
        out = self.norm2(out)
        out = self.tanh(out)
        out = self.conv2(k, out)
        out = self.norm3(out)

        return out

# class FODEncoder(nn.Module):
#
#     def __init__(self, odefunc):
#         super(FODEncoder, self).__init__()
#         self.odefunc = odefunc
#
#     def forward(self, x, krange, t=1.):
#         self.target_func = lambda k, x: torch.mul(torch.sin(6.28 * t * k).view(-1, 1, 1, 1), self.odefunc(k, x))
#         krange = torch.tensor(krange).float()
#         krange = krange.type_as(x)
#         out = -(1 / 6.28) * odeint(self.target_func, x, krange, adjoint_params=(self.odefunc.parameters()), rtol=1e-3,
#                                    atol=1e-3)
#         return out[1]
#     @property
#     def nfe(self):
#         return self.odefunc.nfe
#
#     @nfe.setter
#     def nfe(self, value):
#         self.odefunc.nfe = value

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class FODE(nn.Module):

    def __init__(self, fn):
        super(FODE, self).__init__()
        self.fn = fn

    def forward(self, x, t=10.):
        t_range = torch.tensor([0., t])
        t_range = t_range.type_as(x)
        out = odeint(self.fn, x, t_range, rtol=1e-3,
                                   atol=1e-3)
        return out[1]
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])



    data = datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train)


    train_loader = DataLoader(
        data, batch_size=batch_size,shuffle=True
    )
    return train_loader

if __name__ == '__main__':
    epochsize = 100
    lr = 1e-3

    model = FODE(fn(ODEfunc(1), ODEfunc(1)))
    mse = torch.nn.MSELoss()
    loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = torch.load("../data/toyset.pth")["L11R6"]

    HL=data[0]
    HD=data[1]
    HR=data[2]
    HL01=data[3]
    HR01=data[4]
    energy=data[-2]
    conductance=data[-1]

    x = torch.zeros(HL.shape[-1]+HR.shape[-1]+HD.shape[-1],HL.shape[-1]+HR.shape[-1]+HD.shape[-1])
    x[:HL.shape[-1],:HL.shape[-1]] = HL[0,0,:,:]
    x[HL.shape[-1]+HD.shape[-1]:, HL.shape[-1]+HD.shape[-1]:] = HR[0, 0, :, :]
    x[HL.shape[-1]:HL.shape[-1]+HD.shape[-1], HL.shape[-1]:HL.shape[-1]+HD.shape[-1]] = HD[0, 0, :, :]
    x[:HL.shape[-1],HL.shape[-1]:2*HL.shape[-1]] = HL01[0,0,:,:]
    x[HL.shape[-1]:2 * HL.shape[-1], :HL.shape[-1]] = HL01[0, 0, :, :].T
    x[-HR.shape[-1]*2:-HR.shape[-1],-HR.shape[-1]:] = HR01[0,0,:,:]
    x[-HR.shape[-1]:,-HR.shape[-1] * 2:-HR.shape[-1]] = HR01[0, 0, :, :].T

    count = 0
    for epoch in range(epochsize):

        for sample, label in loader:
            count += 1
            optimizer.zero_grad()
            out = model(sample, t=10.)
            loss = mse(out, sample)

            loss.backward()
            optimizer.step()

            if count % 1 == 0:

                plt.clf()
                plt.matshow(out[0].detach().view(28,28))
                plt.show()

            # if count % 10 == 0:
            #     image = sample[0]
            #     plt.ion()
            #     for t in range(0,20):



