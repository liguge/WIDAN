import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, frequency, eps=0.3, mode='sigmoid'):
        super(Laplace_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.fre = frequency
        self.a_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, self.kernel_size - 1, steps=int(self.kernel_size))

    def Laplace(self, p):
        # m = 1000
        # ep = 0.03
        # # tal = 0.1
        # f = 80
        w = 2 * torch.pi * self.fre
        # A = 0.08
        q = torch.tensor(1 - pow(0.03, 2))

        if self.mode == 'vanilla':
            return ((1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))) * (torch.sin(w * (p - 0.1))))

        if self.mode == 'maxmin':
            a = (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))
            return (a-a.min())/(a.max()-a.min())

        if self.mode == 'sigmoid':
            return (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))

        if self.mode == 'softmax':
            return (1/math.e) * torch.exp(F.softmax((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)), dim=-1)) * (
                torch.sin(w * (p - 0.1)))

        if self.mode == 'tanh':
            return (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).tanh()) * (torch.sin(w * (p - 0.1)))

        if self.mode == 'atan':
            return (1/math.e) * torch.exp((2 / torch.pi) * (((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))).atan()) * (
                torch.sin(w * (p - 0.1)))

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        return self.Laplace(p1).view(self.out_channels, 1, self.kernel_size)
    
class Harmonice_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, frequency, eps=0.3, mode='sigmoid'):
        super(Laplace_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.fre = frequency
        self.a_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, self.kernel_size - 1, steps=int(self.kernel_size))

    def Laplace(self, p):
        # m = 1000
        # ep = 0.03
        # # tal = 0.1
        # f = 80
        w = torch.pi * self.fre
        # A = 0.08
        q = torch.tensor(1 - pow(0.03, 2))

        if self.mode == 'vanilla':
            return (torch.sin(0.03/q*2*w(p-0.1))-torch.sin(0.03/q*w(p-0.1)))/ (0.03/q*w(p-0.1))

        if self.mode == 'maxmin':
            a = (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))
            return (a-a.min())/(a.max()-a.min())

        if self.mode == 'sigmoid':
            return (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))

        if self.mode == 'softmax':
            return (1/math.e) * torch.exp(F.softmax((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)), dim=-1)) * (
                torch.sin(w * (p - 0.1)))

        if self.mode == 'tanh':
            return (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).tanh()) * (torch.sin(w * (p - 0.1)))

        if self.mode == 'atan':
            return (1/math.e) * torch.exp((2 / torch.pi) * (((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))).atan()) * (
                torch.sin(w * (p - 0.1)))

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        return self.Laplace(p1).view(self.out_channels, 1, self.kernel_size)


class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3, mode='sigmoid'):
        super(Morlet_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))

    def Morlet(self, p, c):

        if self.mode == 'vanilla':
            return c * torch.exp((-torch.pow(p, 2) / 2)) * torch.cos(5 * p)

        if self.mode == 'sigmoid':
            return c * torch.exp((-torch.pow(p, 2) / 2).sigmoid()) * torch.cos(5 * p)

        if self.mode == 'maxmin':
            a = c * torch.exp((-torch.pow(p, 2) / 2).sigmoid()) * torch.cos(5 * p)
            return (a-a.min())/(a.max()-a.min())

        if self.mode == 'softmax':
            return c * torch.exp(F.softmax(-torch.pow(p, 2) / 2, dim=-1)) * torch.cos(5 * p)

        if self.mode == 'tanh':
            return c * torch.exp((-torch.pow(p, 2) / 2).tanh()) * torch.cos(5 * p)

        if self.mode == 'atan':
            return c * torch.exp((2 / torch.pi) * ((-torch.pow(p, 2) / 2).atan())) * torch.cos(5 * p)

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)
        c = (pow(torch.pi, 0.25)) / torch.sqrt(self.a_ + 0.01)  # worth exploring    1e-3  & D = C / self.a_.cuda()
        # C = (pow(pi, 0.25)) / (torch.sqrt(self.a_) + 1e-12)
        # C = (pow(pi, 0.25)) / (torch.sqrt(self.a_) + self.eps)
        Morlet_right = self.Morlet(p1, c)
        Morlet_left = self.Morlet(p2, c)
        return torch.cat([Morlet_left, Morlet_right], dim=1).view(self.out_channels, 1, self.kernel_size)


class Mexh_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3, mode='sigmoid'):
        super(Mexh_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))

    def Mexh(self, p):
        # p = 0.04 * p  # 将时间转化为在[-5,5]这个区间内
        if self.mode == 'vanilla':
            return ((2 / pow(3, 0.5) * (pow(torch.pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp((-torch.pow(p, 2) / 2)))

        if self.mode == 'sigmoid':
            return ((2 / pow(3, 0.5) * (pow(torch.pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp(
                (-torch.pow(p, 2) / 2).sigmoid()))

        if self.mode == 'softmax':
            return ((2 / pow(3, 0.5) * (pow(torch.pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp(
                F.softmax(-torch.pow(p, 2) / 2, dim=-1)))

        if self.mode == 'tanh':
            return ((2 / pow(3, 0.5) * (pow(torch.pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp(
                (-torch.pow(p, 2) / 2).tanh()))

        if self.mode == 'atan':
            return (2 / torch.pi) * ((2 / pow(3, 0.5) * (pow(torch.pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp(
                (-torch.pow(p, 2) / 2))).atan()

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)
        Mexh_right = self.Mexh(p1)
        Mexh_left = self.Mexh(p2)
        return torch.cat([Mexh_left, Mexh_right], dim=1).view(self.out_channels, 1, self.kernel_size)  # 40x1x250


class Gaussian_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3, mode='sigmoid'):
        super(Gaussian_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(1, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))

    def Gaussian(self, p):

        if self.mode == 'vanilla':
            return -((1 / (pow(2 * torch.pi, 0.5))) * p * (torch.exp(((-torch.pow(p, 2)) / 2))))

        if self.mode == 'sigmoid':
            return -((1 / (pow(2 * torch.pi, 0.5))) * p * (torch.exp(((-torch.pow(p, 2)) / 2).sigmoid())))

        if self.mode == 'softmax':
            return -((1 / (pow(2 * torch.pi, 0.5))) * p * (torch.exp(F.softmax((-torch.pow(p, 2)) / 2, dim=-1))))

        if self.mode == 'tanh':
            return -((1 / (pow(2 * torch.pi, 0.5))) * p * (torch.exp(((-torch.pow(p, 2)) / 2).tanh())))

        if self.mode == 'atan':
            return -((1 / (pow(2 * torch.pi, 0.5))) * p * (torch.exp((2 / torch.pi) * (((-torch.pow(p, 2)) / 2).atan()))))
        # y = D * torch.exp(-torch.pow(p, 2))
        # F0 = (2./torch.pi)**(1./4.) * torch.exp(-torch.pow(p, 2))
        # y = -2 / (3 ** (1 / 2)) * (-1 + 2 * p ** 2) * F0
        # y = (2./torch.pi)**(1./4.) * torch.exp(-torch.pow(p, 2))
        # y = -2 / (3 ** (1 / 2)) * (-1 + 2 * p ** 2) * y
        # y = -((1 / (pow(2 * torch.pi, 0.5))) * p * torch.exp((-torch.pow(p, 2)) / 2))

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)
        Gaussian_right = self.Gaussian(p1)
        Gaussian_left = self.Gaussian(p2)
        return torch.cat([Gaussian_left, Gaussian_right], dim=1).view(self.out_channels, 1,
                                                                      self.kernel_size)


class Shannon_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3):
        super(Shannon_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))

    def Shannon(self, p):
        return (torch.sin(torch.pi * (p - 0.5)) - torch.sin(2 * torch.pi * (p - 0.5))) / (torch.pi * (p - 0.5))

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)
        Shannon_right = self.Shannon(p1)
        Shannon_left = self.Shannon(p2)
        return torch.cat([Shannon_left, Shannon_right], dim=1).view(self.out_channels, 1,
                                                                    self.kernel_size)


class Sin_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3, mode='sigmoid'):
        super(Sin_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))
        # self.time_disc_right = torch.linspace(0, 1, steps=int((self.kernel_size / 2)))
        # self.time_disc_left = torch.linspace(-1, 0, steps=int((self.kernel_size / 2)))

    def Sin(self, p):
        # y = (torch.sin(2 * torch.pi * (p - 0.5)) - torch.sin(torch.pi * (p - 0.5))) / (torch.pi * (p - 0.5))
        return torch.sin(p) / (p + 1e-12)

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)  #
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)  #
        Shannon_right = self.Sin(p1)
        Shannon_left = self.Sin(p2)
        return torch.cat([Shannon_left, Shannon_right], dim=1).view(self.out_channels, 1,
                                                                    self.kernel_size)
class Morlet_fast1(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3, mode='sigmoid'):
        super(Morlet_fast1, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc = torch.linspace((-self.kernel_size / 2), (self.kernel_size / 2) - 1, steps=int(self.kernel_size))

    def Morlet(self, p, c):

        if self.mode == 'vanilla':
            return c * torch.exp((-torch.pow(p, 2) / 2)) * torch.cos(5 * p)

        if self.mode == 'sigmoid':
            return c * torch.exp((-torch.pow(p, 2) / 2).sigmoid()) * torch.cos(5 * p)

        if self.mode == 'maxmin':
            a = c * torch.exp((-torch.pow(p, 2) / 2).sigmoid()) * torch.cos(5 * p)
            return (a-a.min())/(a.max()-a.min())

        if self.mode == 'softmax':
            return c * torch.exp(F.softmax(-torch.pow(p, 2) / 2, dim=-1)) * torch.cos(5 * p)

        if self.mode == 'tanh':
            return c * torch.exp((-torch.pow(p, 2) / 2).tanh()) * torch.cos(5 * p)

        if self.mode == 'atan':
            return c * torch.exp((2 / torch.pi) * ((-torch.pow(p, 2) / 2).atan())) * torch.cos(5 * p)

    def forward(self):
        # a = self.time_disc - self.b_
        # print(a.size())
        p = (self.time_disc - self.b_) / (self.a_ + self.eps)
        c = (pow(torch.pi, 0.25)) / torch.sqrt(self.a_ + 0.01)  # worth exploring    1e-3  & D = C / self.a_.cuda()
        # C = (pow(pi, 0.25)) / (torch.sqrt(self.a_) + 1e-12)
        # C = (pow(pi, 0.25)) / (torch.sqrt(self.a_) + self.eps)
        Morlet = self.Morlet(p, c)
        return Morlet.view(self.out_channels, 1, self.kernel_size)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    input = torch.randn(2, 1, 1024).cuda()
    weight = Morlet_fast1(out_channels=64, kernel_size=300, eps=0.3, mode='vanilla').forward().cuda()
    weight_t = weight.cpu().detach().numpy()
    y = weight_t[20, :, :].squeeze()
    x = np.linspace(0, 300, 300)
    plt.plot(x, y)
    plt.savefig('wahaha.tiff', format='tiff', dpi=600)
    plt.show()