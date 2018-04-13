import torch
from PIL import Image
from torch.autograd import Variable
import torch.utils.data as utils_data
import os
import random
import numpy as np


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean)
    batch = batch / Variable(std)
    return batch

def masktransform(self,mask_img,color):
    '''
    :type mask_img: Image Mode:RGB
    :type color: list[list[RGB]]
    :rtype: Image
    '''
    mask_img = np.asarray(mask_img)
    w,h,c = mask_img.shape
    mask_out = np.zeros((w,h)).astype("uint8")
    for col in color:
        for i in range(w):
            for j in range(h):
                if mask_img[i][j][0]==col[0] and mask_img[i][j][1]==col[1] and mask_img[i][j][2]==col[2]:
                    mask_out[i][j] = 1
    return Image.fromarray(mask_out)




class Mydataset(utils_data.Dataset):
    def __init__(self, image_dir, mask_dir,color_sets, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.color_sets = color_sets
        self.transform = transform
        self.mask_file_list = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir,f))]
        random.shuffle(self.mask_file_list)

    def __getitem__(self, index):
        mask_filename = os.path.join(self.mask_dir,self.mask_file_list[index])
        image_filename = os.path.join(self.image_dir,self.mask_file_list[index])
        image = Image.open(image_filename)
        mask = Image.open(mask_filename)
        masks = []
        for color in self.color_sets:
            masks.append(self.masktransform(mask,color))
        if self.transform:
            image = self.transform(image)
            masks = [self.transform(i) for i in masks]
        return image,masks




    def __len__(self):
        return len(self.mask_file_list)

    def masktransform(self,mask_img,color):
        '''
        :type mask_img: Image Mode:RGB
        :type color: list[list[RGB]]
        :rtype: Image
        '''
        mask_img = np.asarray(mask_img)
        w,h,c = mask_img.shape
        mask_out = np.zeros((w,h)).astype("uint8")
        for col in color:
            for i in range(w):
                for j in range(h):
                    if mask_img[i][j][0]==col[0] and mask_img[i][j][1]==col[1] and mask_img[i][j][2]==col[2]:
                        mask_out[i][j] = 1
        return Image.fromarray(mask_out)