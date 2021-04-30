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
def get_IoU(predo, target, n_classes=2):
    pred = torch.argmax(predo, dim=1)  # perform argmax to generate 1 channel
    pred = pred.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    pred = np.squeeze(pred)  # remove batch dim and channel dim -> [H, W]
    #pred = torch.from_numpy(pred)
    #pred = re_normalize(pred)  # scale it to the range [0-255]
    target = target.cpu().numpy()
      # pred = pred.view(-1)
      # target = target.view(-1)
    #print('pred',pred.shape)
    #print(target.shape)
 #   pickle.dump(target,open("target.pkl","wb"))
 #   pickle.dump(pred,open("pred.pkl","wb"))
 #   pickle.dump(predo,open("predo.pkl","wb"))

    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return np.nanmean(iou_score)

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
            loss = self.criterion(out,target)
          #  loss = self.criterion(torch.argmax(out, dim=1), target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
          #  print(out.shape,target.shape)
          #  pickle.dump(out,open("out.pkl","wb"))
          #  pickle.dump(target,open("target.pkl","wb"))
           # print(out.shape)
           # print(target.shape)
            #accuracies.append(get_IoU(out,target))
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
         #   print(accuracies[-1],type(accuracies[-1]))
            #batch_iter.set_description(f'Training: (loss {loss_value:.4f}; accuracy {accuracies[-1]:.4f})')  # update progressbar
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        #self.training_accuracy.append(1 - np.mean(accuracies))
        wandb.log({"training_loss": self.training_loss[-1]})
        #wandb.log({"training_accuracy": self.training_accuracy[-1]})
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
      #  val_accuracies = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            x = x.unsqueeze(1)  # if torch tensor

            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out,target)
                #loss = self.criterion(torch.argmax(out, dim=1), target)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                
                #val_accuracies.append(get_IoU(out,target))

                #batch_iter.set_description(f'Validation: (loss {loss_value:.4f}; accuracy {val_accuracies[-1]:.4f})')
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))
    #    self.validation_accuracy.append(np.mean(val_accuracies))

        wandb.log({"validation_loss": self.validation_loss[-1]})
      #  wandb.log({"validation_accuracy": self.validation_accuracy[-1]})


        batch_iter.close()