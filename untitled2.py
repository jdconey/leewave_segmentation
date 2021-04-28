# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:56:13 2021

@author: mm16jdc
"""

import torch
from torch.utils import data
import xarray
import numpy
import os

root = 'C:/Users/mm16jdc/Documents/ukv_data/data/segmentation_full/'
xmin=275
xmax=787
ymin=250
ymax=762


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
        x, y = xarray.open_dataarray(root+'lw_paths/'+input_ID), numpy.load(root+'masks/'+target_ID)
        x = x[ymin:ymax,xmin:xmax].values
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

inputs = os.listdir(root+'lw_paths/')
targets = os.listdir(root+'masks/')

training_dataset = SegmentationDataSet(inputs=inputs,
                                       targets=targets,
                                       transform=None)

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=2,
                                      shuffle=True)
x, y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')