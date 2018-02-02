e#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:45:49 2018

@author: sensetime
"""

import utils
from transformer_net import TransformerNet
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import numpy as np
from PIL import Image


content_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/0_ori.jpg'
style_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/starry_night_crop.jpg'
styled_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/4.jpg'
black_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/init.jpg'
im_size = 256
batch_size = 1


transformernet = TransformerNet()


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

content_image = Image.open(content_file)
style_image = Image.open(style_file)
styled_image = Image.open(styled_file)
black_image = Image.open(black_file)

transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

mse_loss = torch.nn.MSELoss()
styled_image = transform(styled_image)
content_image = transform(content_image)
style_image = transform(style_image)
black_image = transform(black_image)

content_image = Variable(content_image.expand(1,3,im_size,im_size))
style_image = Variable(style_image.expand(1,3,im_size,im_size))
styled_image = Variable(styled_image.expand(1,3,im_size,im_size))
black_image = Variable(black_image.expand(1,3,im_size,im_size))

black_image = transformernet(content_image)






content_image = utils.normalize_batch(content_image)
black_image = utils.normalize_batch(black_image)
styled_image = utils.normalize_batch(styled_image)
style_image =utils.normalize_batch(style_image)

vgg = Vgg16()
feature_x = vgg(content_image)
feature_y = vgg(styled_image)
print('content loss')
print(mse_loss(feature_x[1].detach(),feature_y[1].detach())*1e5)

feature_style = vgg(style_image)
gmx = [utils.gram_matrix(x) for x in feature_style]
gmy = [utils.gram_matrix(x) for x in feature_y]
style_loss = 0
for i in range(len(gmx)):
    print(float(mse_loss(gmx[i].detach(),gmy[i].detach())))
    style_loss += mse_loss(gmx[i].detach(),gmy[i].detach())
print('style loss')
print(float(style_loss*1e10))



























'''
style_model = TransformerNet()
style_model = torch.nn.DataParallel(style_model)
style_model.load_state_dict(torch.load('../saved_models/Ex2_waves.model'))
'''