# =========================================
# Time: 2020-11-17
# Author: wangping
# Function: dataloader for cnn_recognition
# =========================================

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from random import randint, sample
import random

TRAIN_DATA_PATH = "data"
TEST_DATA_PATH = "data"
EXT = ".png"

# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).split()[0]     # open as gray image
    
class MyDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        img_list = glob.glob(TRAIN_DATA_PATH + '/*')
        # print(img_list)
        imgs = []
        
        for single_list in img_list:
            single_list = glob.glob(single_list + "/*")
            for item in single_list:
                label = item.split("/")[-2]
                imgs.append((item, label))

        imgs = sorted(imgs)
        random.shuffle(imgs)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        try:
            fn, label = self.imgs[index]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            label = int(label)
        except:
            fn, label = self.imgs[0]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            label = int(label)
        ##print(fn,label)
        return img, label

    def __len__(self):
        return len(self.imgs)


class MyDataset_test(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        img_list = glob.glob(TEST_DATA_PATH + '/*')
        imgs = []
        
        for single_list in img_list:
            single_list = glob.glob(single_list + "/*")
            for item in single_list:
                label = item.split("/")[-2]
                imgs.append((item, label))

        imgs = sorted(imgs)
        random.shuffle(imgs)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        try:
            fn, label = self.imgs[index]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            label = int(label)
        except:
            fn, label = self.imgs[0]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs)