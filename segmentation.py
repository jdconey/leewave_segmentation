# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:38:18 2021

@author: mm16jdc
"""

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
import random
import cv2
from PIL import Image

train_base = 'C:/Users/mm16jdc/Documents/CEDA_satellites/pytorch_mask_test/images_2/train/'
val_base = 'C:/Users/mm16jdc/Documents/CEDA_satellites/pytorch_mask_test/images_2/valid/'
#images = os.listdir(base+'images')
#masks = os.listdir(base+'masks')





    ######################################
class WavesDataset(Dataset):
    def __init__(self, folder_path,transform=None):
        self.folder_path = folder_path
        super(WavesDataset, self).__init__()
        self.img_files = os.listdir(folder_path+'images')
        self.mask_files = os.listdir(folder_path+'masks')
        self.transform = transform

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
          #  print(self.folder_path+'images/'+img_path)
          #  print(self.folder_path+'masks/'+mask_path)
            data = cv2.imread(self.folder_path+'images/'+img_path,0)
            label = cv2.imread(self.folder_path+'masks/'+mask_path,0)
            
            if self.transform:
                data2 = self.transform(data)
                label2 = self.transform(label)
            
            #return torch.from_numpy(data2).float(), torch.from_numpy(label2).float()
            return(data2,label2)

    def __len__(self):
        return len(self.img_files)  

def stats(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            loss = nn.CrossEntropyLoss(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            running_loss += loss
            n += 1
    return running_loss/n, correct/total 

    
trnsfrms = transforms.Compose([
           # transforms.Resize(100,100),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0],std=[1]),
            ])

train_set = WavesDataset(
    train_base,
    transform = trnsfrms
    )

val_set = WavesDataset(
        val_base,
        transform=trnsfrms
        )


train_loader = DataLoader(
    train_set,
    batch_size=1,
    shuffle=True,
    num_workers=2,
)

test_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=1, # Forward pass only so batch size can be larger
    shuffle=False,
    num_workers=0
)

net = nn.Sequential(
    nn.Conv2d(1,16, kernel_size=5, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(16,32, kernel_size=5, padding=0,),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    
    # complete the rows of this specificaton here
    
    nn.Flatten(),    # convert from 2-D feature map and channels to a 1-D vector (within minibatch)
    nn.Linear(301088, 10000),    # replace the empty argument with the size of the 1-D vector
   # nn.ReLU(),
    #nn.Linear(128,1600)
)

#for param in net.parameters():
#    print(param.shape)

print('hello!')

net = UNet(dimensions=1)
optimizer = optim.RMSprop(net.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
criterion = nn.CrossEntropyLoss()
epoch_number=10
statsrec2 = np.zeros((3,epoch_number))

for epoch in range(epoch_number):  # loop over the dataset multiple times

    running_loss = 0.0
    n = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels_flattened = torch.reshape(labels,[1,1,160000])
         # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward, backward, and update parameters
        outputs = net(inputs)
        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        # accumulate loss
        running_loss += loss.item()
        n += 1
    ltrn = running_loss/n
    ltst, atst = stats(test_loader, net)
    statsrec2[:,epoch] = (ltrn, ltst, atst)
    print(f"epoch: {epoch} training loss: {ltrn: .3f}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")