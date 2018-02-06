#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:50:40 2018

@author: sensetime
"""
from torch.autograd import Variable
import torchvision.transforms as transforms
import weights 
import math
import torch
import torch.nn as nn
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,requires_grad=False):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0], is_downsample=True)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.fc1 = nn.Linear(12544, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 256)
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, 212)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_downsample=False):
        downsample = None
        if stride != 1 or is_downsample:
            downsample = nn.Sequential(
                nn.Conv2d(planes/stride, planes,
                          kernel_size=1, stride=stride, bias=True)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.layer1(x)
        l1 = x
        x = self.layer2(x)
        l2 = x
        x = self.layer3(x)
        l3 = x
        x = self.layer4(x)
        l4 = x

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc3(x)

        return [l1,l2,l3,l4]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet18']))
    return model

def get_model():
    pre_model = weights.weights
    pre_model.load_state_dict(torch.load('./weights.pth'))
    pre_model.eval()
    model = resnet18()
    assert(len(pre_model.state_dict().items())==len(model.state_dict().items()))
    for (name1, item1), (name2, item2) in zip(pre_model.state_dict().items(), model.state_dict().items()):
        assert(item1.size()==item2.size())
    for (name1, item1), (name2, item2) in zip(pre_model.state_dict().items(), model.state_dict().items()):
        model.state_dict()[name2].copy_(item1)
    return model