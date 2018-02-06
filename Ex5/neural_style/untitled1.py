#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:07:20 2018

@author: sensetime
"""
import face_alignment_net
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


model = face_alignment_net.get_model()






content_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/0_ori.jpg'
style_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/starry_night_crop.jpg'
styled_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/4.jpg'
black_file = '/home/sensetime/Desktop/Vgg_Feature_Visual/pictures/init.jpg'
im_size = 168
batch_size = 1

content_image = Image.open(content_file)
style_image = Image.open(style_file)
black_image = Image.open(black_file)
styled_image = Image.open(styled_file)

preprocess = transforms.Compose([transforms.Scale(168),
                                              transforms.CenterCrop(112),
                                              transforms.ToTensor()])
    
#model = model.cuda()    
def distribution(image1,network):
    """
        :type image1:Image
        :type network:torch.nn.Module,used to extract features
    """
    
    transform = transforms.Compose([transforms.Scale(im_size),transforms.CenterCrop(112),transforms.Grayscale(),transforms.ToTensor()])
    image1 = transform(image1)
    mean_ = torch.mean(image1)
    std_ = torch.std(image1)
    image1 = (image1-mean_)/std_
    vgg = network
    a,b,c = image1.size()
    image1 = torch.autograd.Variable(image1.expand(1,a,b,c))
    image1_featuremap = vgg.forward(image1)
    dis_list = []
    for i in range(len(image1_featuremap)):
        print("layer%d:" %i)
        image1_feature = image1_featuremap[i].data.cpu().numpy()
        image1_feature = image1_feature[0,:,:,:]
        a,b,c = image1_feature.shape
        image1_feature = image1_feature.reshape((a,b*c))
        l = []
        for j in range(b*c):
            l.append(np.dot(image1_feature[:,j].T,image1_feature[:,j]))
        dis_list.append(l)
        #plt.figure()
        #sns.distplot(l)
        '''
        plt.figure()
        plt.hist(l,bins=50)
        '''
        #print(image1_feature.shape)
    return dis_list
lc = distribution(content_image,model)
ls = distribution(style_image,model)
lsd = distribution(styled_image,model)
name_list = ['layer1', 'layer2', 'layer3', 'layer4']
for i in range(1,3):
    plt.figure()
    plt.title(name_list[i])
    sns.kdeplot(ls[i],label='style image')
    sns.kdeplot(lsd[i],label='styled content',color='g') 
    sns.kdeplot(lc[i],label='content image')
