import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
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



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, out_channel=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=64, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self._initialize_weights()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for name, can in self.named_children():
            if name == 'conv1':
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        if m.kernel_size == (64,):
                            m.weight.data = Laplace_fast(out_channels=64, kernel_size=64, eps=0.1, frequency=64000,
                                                         mode='sigmoid').forward()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


# convnet without the last layer
class resnet18_features(nn.Module):
    def __init__(self, pretrained=False):
        super(resnet18_features, self).__init__()
        self.model_resnet18 = resnet18(pretrained)
        self.__in_features = 512


    def forward(self, x):
        x = self.model_resnet18(x)
        return x

    def output_num(self):
        return self.__in_features