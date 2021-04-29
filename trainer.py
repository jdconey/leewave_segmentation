# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:32:01 2021

@author: mm16jdc
"""

import numpy as np
import torch
import wandb
from sklearn.metrics import jaccard_score as jsc
import pickle

EPS = 1e-6
#slightly modified
def get_IoU(pred, target, n_classes=2):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)
  print(pred.shape)
  print(target.shape)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.nanmean(ious)

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        accuracies = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            x = x.unsqueeze(1)  # if torch tensor
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            print(out.shape,target.shape)
            pickle.dump(out,open("out.pkl","wb"))
            pickle.dump(target,open("target.pkl","wb"))
           # print(out.shape)
           # print(target.shape)
        #    accuracies.append(get_IoU(out,target))
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
         #   print(accuracies[-1],type(accuracies[-1]))
            batch_iter.set_description(f'Training: (loss {loss_value:.4f}; accuracy {accuracies[-1]:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.training_accuracy.append(np.mean(accuracies))
        wandb.log({"training_loss": self.training_loss[-1]})
        wandb.log({"training_accuracy": self.training_accuracy[-1]})
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        val_accuracies = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            x = x.unsqueeze(1)  # if torch tensor

            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                
      #          val_accuracies.append(get_IoU(out,target))

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f}); accuracy {val_accuracies[-1]:.4f})')

        self.validation_loss.append(np.mean(valid_losses))
        self.validation_accuracy.append(np.mean(val_accuracies))

        wandb.log({"validation_loss": self.validation_loss[-1]})
        wandb.log({"validation_accuracy": self.validation_accuracy[-1]})


        batch_iter.close()