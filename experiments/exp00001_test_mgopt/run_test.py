#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:39:54 2022

@author: gjsaave
"""

import numpy as np
import sys
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import os
import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import math

sys.path.append("../../")
from models import LinearFC, CustomMSE
from datasets import PeaksData
from mgopt_methods import make_dropout_dict, make_grad_dict, mgopt

#Exp Use batch size 1 for this experiment
torch.set_printoptions(precision=4)
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

epochs = 10
num_features = 2
num_classes = 5
num_hidden_nodes = 64
batch_size = 5000
momentum = 0.9
weight_decay = 1e-4
base_lr = 0.001
num_layers=3

levels = 1
alpha = 2

drop_rates = make_dropout_dict(levels)
#Dict to store gradients by mgopt level, mgopt step (or optimization problem), and NN layer
#FIXME test this extensively. Make sure grads aren't being overwritten by recursive calls. g2_bar might be weird
wgrads, bgrads = make_grad_dict(levels, num_layers)

data_path = "/Users/gjsaave/Desktop/peaks_data"
train_data = PeaksData(train_or_val="train", data_path=data_path)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_data = PeaksData(train_or_val="val", data_path=data_path)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

# #fine level model has no dropout
model = LinearFC(num_layers, num_features, num_classes, num_hidden_nodes, drop_rate=drop_rates[str(levels)])
criterion = CustomMSE(num_layers=num_layers, vx_term=False)
optimizer = torch.optim.SGD(params=model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

def train(loader, model, criterion, optimizer, wgrads, bgrads, levels, drop_rates, base_lr, momentum, weight_decay, alpha, num_layers):
    correct = 0
    total = 0
    acc = 0

    model.train()
    for i, (input, target) in enumerate(loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        model, output, loss, wgrads, bgrads = mgopt(model, input_var, target_var, criterion, optimizer, wgrads, bgrads, levels, drop_rates, base_lr, momentum, weight_decay, alpha, num_layers=num_layers)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.max(1, keepdim=True).indices).sum().item()
        total += batch_size
        acc = correct / total
        #print("Train Accuracy: ", acc)
        
    return acc, loss, model, wgrads, bgrads


def validation(val_loader, model, criterion):
    correct = 0
    total = 0
    acc = 0

    model.eval()
        
    for i, (input, target) in enumerate(val_loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        with torch.no_grad():
            output = model(input_var.float())
            loss = criterion(output, target_var.float(), None, None, None, None) #None since we are not using <v,x> term
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.max(1, keepdim=True).indices).sum().item()
            total += batch_size
            acc = correct / total
            #print("Test Accuracy: ", acc)
            
    return acc, loss


train_accs = []
val_accs = []
train_losses = []
val_losses = []

#FIXME Right now we assume N0=1 i.e., we use 1 mini-batch. It might make sense to use several mini-batches
for epoch in range(epochs):
    _acc, _loss, model, wgrads, bgrads = train(train_loader, model, criterion, optimizer, wgrads, bgrads, levels, drop_rates, base_lr, momentum, weight_decay, alpha, num_layers)
    train_accs.append(_acc)
    train_losses.append(_loss)
    _acc, _loss = validation(val_loader, model, criterion)
    val_accs.append(_acc)
    val_losses.append(_loss)
    
plt.plot(list(range(epochs)), train_accs, label="train")
plt.plot(list(range(epochs)), val_accs, label="val")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("2 Fully connected linear layers. Peaks data.")
plt.legend()
plt.show()

plt.plot(list(range(epochs)), train_losses, label="train")
plt.plot(list(range(epochs)), val_losses, label="val")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("2 Fully connected linear layers. Peaks data.")
plt.legend()
plt.show()


