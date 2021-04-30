# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:56:13 2021

@author: mm16jdc
"""

import torch

import torch.nn.functional as F
import torch.nn as nn


from torch.utils import data
import xarray
import numpy
import os
from transformations import Compose, Resize, DenseTarget
from transformations import MoveAxis, Normalize01, AlbuSeg2d
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib
import albumentations
from unet2 import UNet
from trainer import Trainer, get_IoU
import tqdm

import wandb
wandb.init(project="ukv_leewaves")

root = pathlib.Path('C:/Users/mm16jdc/Documents/ukv_data/data/segmentation_full/')
xmin=275
xmax=787
ymin=250
ymax=762


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class IoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)       
        inputs = torch.argmax(inputs, dim=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = xarray.open_dataarray(input_ID), numpy.load(target_ID)
        x = x[ymin:ymax,xmin:xmax].values
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

class SegmentationDataSet2(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = xarray.open_dataarray(img_name), numpy.load(tar_name)
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = xarray.open_dataarray(input_ID), numpy.load(target_ID)
            x = x[ymin:ymax,xmin:xmax].values

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

'''
inputs = os.listdir(root+'lw_paths/')
targets = os.listdir(root+'masks/')

training_dataset = SegmentationDataSet(inputs=inputs,
                                       targets=targets,
                                       transform=None)

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=2,
                                      shuffle=True)
x, y = next(iter(training_dataloader))

'''



# root directory
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames
# input and target files
inputs = get_filenames_of_path(root / 'lw_paths')
targets = get_filenames_of_path(root / 'masks2')
# training transformations and augmentations
transforms_training = Compose([
    # Resize(input_size=(128, 128, 3), target_size=(128, 128)),
    AlbuSeg2d(albu=albumentations.HorizontalFlip(p=0.5)),
    DenseTarget(),
    MoveAxis(),
    Normalize01()
])

# validation transformations
transforms_validation = Compose([
    # Resize(input_size=(128, 128, 3), target_size=(128, 128)),
    DenseTarget(),
    MoveAxis(),
    Normalize01()
])
# random seed
random_seed = 42
# split dataset into training set and validation set
train_size = 0.9  # 80:20 split

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)
# dataset training
dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training)

# dataset validation
dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=4,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=4,
                                   shuffle=True)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

'''
model = UNet(in_channels=2,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=1)
'''

model = UNet(n_classes=2, padding=True, up_mode='upsample')

# criterion
#criterion = torch.nn.CrossEntropyLoss()
#criterion= IoULoss()
criterion = CrossEntropy2d()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# trainer
wandb.watch(model, log_freq=100)


trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=10,
                  epoch=0,
                  notebook=False)
# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

torch.save({
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': [training_losses,validation_losses],
            'lr_rates':lr_rates
            }, 'model_2021-04-29.pt')

torch.save(model, 'little_model_2021-04-30.pt')