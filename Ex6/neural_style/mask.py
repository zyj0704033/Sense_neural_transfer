#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:16:09 2018

@author: sensetime
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import utils
from transformer_net import TransformerNet
from PIL import Image
def masktransform(mask_img,color):
    """
    :type mask_img: Image Mode:RGB
    :type color: list[list[RGB]]
    :rtype: Image
    """
    mask_img = np.asarray(mask_img)
    w,h,c = mask_img.shape

    mask_out = np.zeros((w,h,c)).astype("uint8")
    for col in color:
        for i in range(w):
            for j in range(h):
                if mask_img[i][j][0]==col[0] and mask_img[i][j][1]==col[1] and mask_img[i][j][2]==col[2]:
                    mask_out[i][j][0] = 0
                    mask_out[i][j][1] = 0
                    mask_out[i][j][2] = 1
    return Image.fromarray(mask_out,mode="RGB")
    '''
    mask_out = np.zeros((w,h)).astype("uint8")
    for col in color:
        for i in range(w):
            for j in range(h):
                if mask_img[i][j][0]==col[0] and mask_img[i][j][1]==col[1] and mask_img[i][j][2]==col[2]:
                    mask_out[i][j] = 1
        '''
    return Image.fromarray(mask_out)

content_image = utils.load_image('0_ori.jpg')
mask_image = utils.load_image('mask.png')

content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0)
content_image = content_image.cuda()
content_image = Variable(content_image, volatile=True)

style_model = TransformerNet()
style_model = torch.nn.DataParallel(style_model,device_ids=[0])
style_model.load_state_dict(torch.load('/home/sensetime/Desktop/Sense_neural_transfer/fast_neural_style/saved_models/Ex2_3features.model'))
output = style_model(content_image)
output = output.cpu()
output_data = output.data[0]
img = output_data.clone().clamp(0, 255).numpy()
img = img.transpose(1, 2, 0).astype("uint8")
img = Image.fromarray(img)
img.show()

mask_image = utils.load_image('mask.png')
mask_image = masktransform(mask_image,[[255,0,0]])
print(mask_image.mode)
print(mask_image.size)
mask_image.show()
mask_image = content_transform(mask_image)
mask_image = mask_image.unsqueeze(0)
mask_image = mask_image.cuda()
mask_image = Variable(mask_image, volatile=True)
output = style_model(mask_image)
output = output.cpu()
output_data = output.data[0]
img = output_data.clone().clamp(0, 255).numpy()
img = img.transpose(1, 2, 0).astype("uint8")
img = Image.fromarray(img)
img.show()

