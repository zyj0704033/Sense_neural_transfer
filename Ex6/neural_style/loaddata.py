import os
import utils
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from vgg import Vgg16
from torch.autograd import Variable
from PIL import Image
import numpy as np
from vgg import maskpooling
image_dir = '/home/sensetime/Desktop/Opencv_test/5--Car_Accident'
mask_dir = '/home/sensetime/Desktop/Opencv_test/mask'
l = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir,f))]
for i in l:
    print(i) 
transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor(),transforms.Lambda(lambda x: x.mul(255))])
datasets = utils.Mydataset(image_dir,mask_dir,[[[255,255,255]],[[255,0,0]]],transform)
dataloder = DataLoader(datasets,batch_size=2,drop_last=True)
MaskPooling = maskpooling()
for b,i in enumerate(dataloder):
    print(b)
    print(torch.max(i[1][0]))
    print(i[1][0].shape)
    Variable(i[1])
    a = MaskPooling(Variable(i[1][0]))
    for k in a:
        print(k.shape)

style_image = utils.load_image('/home/sensetime/Desktop/Opencv_test/5--Car_Accident/tar62.png')
vgg = Vgg16(requires_grad=False)
style_transform = transforms.Compose([
	transforms.Resize(256),                                                                                   
	transforms.CenterCrop(256), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
    ])
style = style_transform(style_image)
print(style.shape)
style = style.repeat(4, 1, 1, 1)
print(style.shape)
style_v = Variable(style)
style_v = utils.normalize_batch(style_v)
features_style = vgg(style_v)
for i in features_style:
    print(i.data.shape)