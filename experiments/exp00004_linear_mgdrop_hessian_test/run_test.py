#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 04:33:39 2022

@author: gjsaave
"""

import numpy as np
import sys
import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

sys.path.append("../../")
from models import LinearFC, CustomMSE
from datasets import PeaksData
from mgopt_methods import MGDrop
from logging_utils import save_results, save_mgdrop_stuff
from train_test_methods import train_mgdrop, validation_mgdrop


torch.set_printoptions(precision=4)
torch.backends.cudnn.enabled = False

iters = 1 #make np.inf if you want to do normal training
epochs = 1
num_features = 2
num_classes = 5
num_hidden_nodes = 4
momentum = 0
weight_decay = 0
base_lr = 0.001

num_layers=2
levels = 1
alpha = 2
bls_num_test_points = 10
compute_hessian = True
clear_grads = False
store_weights = False



data_path = "/Users/gjsaave/Desktop/peaks_data"

for batch_size in [1]:
    random_seed = 4
    torch.manual_seed(random_seed)
    
    exp_folder = "seed" + str(random_seed) + "_batchsize" + str(batch_size) + "_momentum" + str(momentum) + "_weightdecay" + str(weight_decay) + "_numlayers" + str(num_layers) + "_lr" + str(base_lr) + "_iters" + str(iters)
    
    train_data = PeaksData(train_or_val="train", data_path=data_path)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_data = PeaksData(train_or_val="val", data_path=data_path)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    
    output_savepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), exp_folder, "output")
    if not os.path.exists(output_savepath):
        os.makedirs(output_savepath)
        
    plots_savepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), exp_folder, "plots")
    if not os.path.exists(plots_savepath):
        os.makedirs(plots_savepath)
    
    mgdrop = MGDrop(num_layers, levels, alpha, compute_hessian, clear_grads, store_weights, bls_num_test_points)
    
    # #fine level model has no dropout
    model = LinearFC(num_layers, num_features, num_classes, num_hidden_nodes, drop_rate=mgdrop.drop_rates[str(levels)])
    criterion = CustomMSE(num_layers=num_layers, vx_term=False)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    train_accs_all_iters = []
    val_accs_all_iters = []
    
    for epoch in range(epochs):
        _acc, _loss, model, train_accs_all_iters = train_mgdrop(train_loader, model, criterion, optimizer, batch_size, mgdrop, levels, base_lr, momentum, weight_decay, train_accs_all_iters, iters)
        train_accs.append(_acc)
        train_losses.append(_loss.item())
        print("epoch: ", epoch)
        print("train acc: ", _acc)
        
        _acc, _loss, val_accs_all_iters = validation_mgdrop(val_loader, model, criterion, batch_size, val_accs_all_iters, iters)
        val_accs.append(_acc)
        val_losses.append(_loss.item())
        print("val acc: ", _acc)
        
    
    save_results(output_savepath, epochs, train_accs, val_accs, train_losses, val_losses, train_accs_all_iters, val_accs_all_iters)
    save_mgdrop_stuff(output_savepath, mgdrop)
        
    plt.plot(list(range(epochs)), train_accs, label="train")
    plt.plot(list(range(epochs)), val_accs, label="val")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Peaks data.")
    plt.legend()
    plt.savefig(os.path.join(plots_savepath, "acc_per_epoch"), dpi=300, bbox_inches="tight")
    plt.clf()
    
    plt.plot(list(range(epochs*iters)), train_accs_all_iters, label="train")
    plt.plot(list(range(epochs*iters)), val_accs_all_iters, label="val")
    plt.xlabel("forward/backward passes")
    plt.ylabel("accuracy")
    plt.title("Peaks data.")
    plt.legend()
    plt.savefig(os.path.join(plots_savepath, "acc_per_iters"), dpi=300, bbox_inches="tight")
    plt.clf()
    
    plt.plot(list(range(epochs)), train_losses, label="train")
    plt.plot(list(range(epochs)), val_losses, label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Peaks data.")
    plt.legend()
    plt.savefig(os.path.join(plots_savepath, "loss"), dpi=300, bbox_inches="tight")
    plt.clf()


