#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 05:01:31 2022

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
from logging_utils import save_results
import time


def train_mgdrop(loader, model, criterion, optimizer, batch_size, mgdrop, levels, base_lr, momentum, weight_decay, train_accs_all_iters, iters=np.inf):
    correct = 0
    total = 0
    acc = 0

    model.train()
    for i, (input, target) in enumerate(loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        model, optimizer, output, loss = mgdrop.mgopt(model, input_var, target_var, criterion, optimizer, levels, base_lr, momentum, weight_decay)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.max(1, keepdim=True).indices).sum().item()
        total += batch_size
        acc = correct / total
        train_accs_all_iters.append(acc)
        
        if i%50 == 0:
            print("Iteration, Train Accuracy", i, " ", acc)
        
        #if we've iterated iters times then break out
        if i == iters-1:
            break
        
        
    return acc, loss, model, train_accs_all_iters


def validation_mgdrop(val_loader, model, criterion, batch_size, val_accs_all_iters, iters=np.inf):
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
            val_accs_all_iters.append(acc)
            
        #if we've iterated iters times then break out
        if i == iters-1:
            break
        
    return acc, loss, val_accs_all_iters


def train(train_loader, model, criterion, optimizer, batch_size, train_acc_all_iters, num_evals=1, iters=np.inf):
    correct = 0
    total = 0
    acc = 0
    
    model.train()
    for i, (input, target) in enumerate(train_loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        for i in range(num_evals):
            optimizer.zero_grad()
            output = model(input_var.float())
            loss = criterion(output, target_var.float(), None, None, None, None)
            loss.backward()
            optimizer.step()
        #print("Train Loss: ", loss.item())
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.max(1, keepdim=True).indices).sum().item()
        total += batch_size
        acc = correct / total
        #print("Train Accuracy: ", acc)
        train_acc_all_iters.append(acc)
        
        #if we've iterated iters times then break out
        if i == iters-1:
            break
        
    return acc, loss, model, train_acc_all_iters

def validation(val_loader, model, criterion, batch_size, val_acc_all_iters, iters=np.inf):
    correct = 0
    total = 0
    acc = 0

    model.eval()
    # if dropout:
    #     model.apply(apply_dropout)
        
    for i, (input, target) in enumerate(val_loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        with torch.no_grad():
            output = model(input_var.float())
            loss = criterion(output, target_var.float(), None, None, None, None)
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.max(1, keepdim=True).indices).sum().item()
            total += batch_size
            acc = correct / total
            #print("Test Accuracy: ", acc)
            val_acc_all_iters.append(acc)
        
        #if we've iterated iters times then break out
        if i == iters-1:
            break
            
    return acc, loss, val_acc_all_iters


def get_target_indices(dataset, target):
    if dataset == "peaks":
        target_indices = target.data.max(1, keepdim=True).indices
    elif dataset == "mnist" or dataset == "cifar10":
        target_indices = target.data.unsqueeze(1)

    return target_indices


def get_correct_var_type(dataset, target_var):
    if dataset == "peaks":
        target_var_correct_type = target_var.float()
    elif dataset == "mnist" or dataset == "cifar10":
        target_var_correct_type = target_var.long()

    return target_var_correct_type

def train_and_val(train_loader, val_loader, model, criterion, optimizer, batch_size, train_accs_all_iters, val_accs_all_iters, val_every_n_iters, full_idx, num_evals=1, dataset="peaks", device="cpu", val_iters_list=[]):
    train_correct = 0
    train_total = 0
    train_acc = 0
    
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var)
        
        for j in range(num_evals):
            optimizer.zero_grad()
            train_output = model(input_var.float())
            train_loss = criterion(train_output, target_var_correct_type, None, None, None, None)
            train_loss.backward()
            optimizer.step()
        #print("Train Loss: ", loss.item())
        
        train_pred = train_output.data.max(1, keepdim=True)[1]
        target_indices = get_target_indices(dataset, target)
        train_correct += train_pred.eq(target_indices).sum().item()
        train_total += input_var.shape[0]
        train_acc = train_correct / train_total
        #print("Train Accuracy: ", acc)
        train_accs_all_iters.append(train_acc)

        #Evaluate entire validation set every n iters or when training epoch is over
        if full_idx % val_every_n_iters == 0 or i == len(train_loader)-1:
            val_iters_list.append(full_idx)
            val_correct = 0
            val_total = 0
            val_acc = 0

            model.eval()

            for i, (input, target) in enumerate(val_loader):
                input, target = input.to(device), target.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)
                target_var_correct_type = get_correct_var_type(dataset, target_var)     

                with torch.no_grad():
                    val_output = model(input_var.float())
                    val_loss = criterion(val_output, target_var_correct_type, None, None, None, None)

                    val_pred = val_output.data.max(1, keepdim=True)[1]
                    target_indices = get_target_indices(dataset, target)
                    val_correct += val_pred.eq(target_indices).sum().item()
                    val_total += input_var.shape[0]
                    val_acc = val_correct / val_total

            val_accs_all_iters.append(val_acc)
            model.train()

        full_idx += 1
        
    return train_acc, train_loss, val_acc, val_loss, train_accs_all_iters, val_accs_all_iters, model, full_idx, val_iters_list


def train_and_val_mgdrop(train_loader, val_loader, model, criterion, optimizer, batch_size, train_accs_all_iters, val_accs_all_iters, val_every_n_iters, mgdrop, levels, base_lr, momentum, weight_decay, full_idx, dataset="peaks", device="cpu", N_c=1, val_iters_list=[]):
    train_correct = 0
    train_total = 0
    train_acc = 0
    
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var)

        model, optimizer, train_output, train_loss = mgdrop.mgopt(model, input_var, target_var_correct_type, criterion, optimizer, levels, base_lr, momentum, weight_decay, N_c)
        
        train_pred = train_output.data.max(1, keepdim=True)[1]
        target_indices = get_target_indices(dataset, target)
        train_correct += train_pred.eq(target_indices).sum().item()
        train_total += input_var.shape[0]
        train_acc = train_correct / train_total
        #print("Train Accuracy: ", acc)
        train_accs_all_iters.append(train_acc)

        #Evaluate entire validation set every n iters or when training epoch is over
        if full_idx % val_every_n_iters == 0 or i == len(train_loader)-1:
            val_iters_list.append(full_idx)
            val_correct = 0
            val_total = 0
            val_acc = 0

            model.eval()

            for i, (input, target) in enumerate(val_loader):
                input, target = input.to(device), target.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)
                target_var_correct_type = get_correct_var_type(dataset, target_var)

                with torch.no_grad():
                    val_output = model(input_var.float())
                    val_loss = criterion(val_output, target_var_correct_type, None, None, None, None)

                    val_pred = val_output.data.max(1, keepdim=True)[1]
                    target_indices = get_target_indices(dataset, target)
                    val_correct += val_pred.eq(target_indices).sum().item()
                    val_total += input_var.shape[0]
                    val_acc = val_correct / val_total

            val_accs_all_iters.append(val_acc)
            model.train()

        full_idx += 1
    return train_acc, train_loss, val_acc, val_loss, train_accs_all_iters, val_accs_all_iters, model, full_idx, val_iters_list


def train_and_val_smgdrop(train_loader, val_loader, model, criterion, optimizer, batch_size, train_accs_all_iters, val_accs_all_iters, val_every_n_iters, mgdrop, levels, base_lr, momentum, weight_decay, full_idx, dataset="peaks", device="cpu", N_c=1, val_iters_list=[]):
    train_correct = 0
    train_total = 0
    train_acc = 0
    len_loader = len(train_loader.dataset)
    
    model.train()
    for i, (input, target) in enumerate(train_loader):
        data_indices = get_data_indices(i, len_loader,batch_size, levels)
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var)

        data_idx = 0
        model, optimizer, train_output, train_loss, _ = mgdrop.smgopt(model, input_var, target_var_correct_type, criterion, optimizer, levels, base_lr, momentum, weight_decay, data_indices, data_idx, N_c)

        with torch.no_grad():
            train_output = model(input_var.float())
            train_loss = criterion(train_output, target_var_correct_type, None, None, None, None)
        
            train_pred = train_output.data.max(1, keepdim=True)[1]
            target_indices = get_target_indices(dataset, target)
            train_correct += train_pred.eq(target_indices).sum().item()
            train_total += input_var.shape[0]
            train_acc = train_correct / train_total
            
        train_accs_all_iters.append(train_acc)

        #Evaluate entire validation set every n iters or when training epoch is over
        if full_idx % val_every_n_iters == 0 or i == len(train_loader)-1:
            val_iters_list.append(full_idx)
            val_correct = 0
            val_total = 0
            val_acc = 0

            model.eval()

            for i, (input, target) in enumerate(val_loader):
                input, target = input.to(device), target.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)
                target_var_correct_type = get_correct_var_type(dataset, target_var)

                with torch.no_grad():
                    val_output = model(input_var.float())
                    val_loss = criterion(val_output, target_var_correct_type, None, None, None, None)

                    val_pred = val_output.data.max(1, keepdim=True)[1]
                    target_indices = get_target_indices(dataset, target)
                    val_correct += val_pred.eq(target_indices).sum().item()
                    val_total += input_var.shape[0]
                    val_acc = val_correct / val_total

            val_accs_all_iters.append(val_acc)
            model.train()

        full_idx += 1
    return train_acc, train_loss, val_acc, val_loss, train_accs_all_iters, val_accs_all_iters, model, full_idx, val_iters_list


def get_data_indices(i, dataset_size, batch_size, levels):
    data_indices = []
    step = int(batch_size/(2*levels + 1))
    total = batch_size

    #If the batch is too small we just use the same indices for whole iteration og smgdrop
    if (i+1)*batch_size > dataset_size:
        for k in range(0, total, step):
            data_indices.append((0, -1))

    else:
        last_k = 0
        for k in range(0, total, step):
            data_indices.append((k, k+step))
            last_k = k
            
    return data_indices


def time_train_passes(train_loader, val_loader, model, criterion, optimizer, batch_size):

    forward_times = []
    backward_times = []
    optimizer_times = []
    loss_times = []
    
    model.train()
    for i, (input, target) in enumerate(train_loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        optimizer.zero_grad()
        start = time.time()
        train_output = model(input_var.float())
        forward_times.append(time.time() - start)
        start = time.time()
        train_loss = criterion(train_output, target_var.float(), None, None, None, None)
        loss_times.append(time.time() - start)
        

        start = time.time()
        train_loss.backward()
        backward_times.append(time.time() - start)
        start = time.time()
        optimizer.step()
        optimizer_times.append(time.time() - start)

    return forward_times, backward_times, optimizer_times, loss_times


def dropout_coarsening_test(train_loader, val_loader, model, criterion, optimizer, batch_size, train_accs_all_iters, val_accs_all_iters, val_accs_all_iters_full, val_every_n_iters, num_evals=1, dataset="peaks", device="cpu", num_drops=100):
    train_correct = 0
    train_total = 0
    train_acc = 0
    
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var)
        
        for j in range(num_evals):
            optimizer.zero_grad()
            train_output = model(input_var.float())
            train_loss = criterion(train_output, target_var_correct_type, None, None, None, None)
            train_loss.backward()
            optimizer.step()
        #print("Train Loss: ", loss.item())
        
        train_pred = train_output.data.max(1, keepdim=True)[1]
        target_indices = get_target_indices(dataset, target)
        train_correct += train_pred.eq(target_indices).sum().item()
        train_total += batch_size
        train_acc = train_correct / train_total
        #print("Train Accuracy: ", acc)
        train_accs_all_iters.append(train_acc)

        #Evaluate entire validation set every n iters or when training epoch is over
        if i % val_every_n_iters == 0 or i == len(train_loader)-1:
            val_acc_per_drop = []
            for d in range(num_drops):
                val_correct = 0
                val_total = 0
                val_acc = 0

                model.train()
                #This forces test time dropout
                #model.apply(apply_dropout)

                for j, (input, target) in enumerate(val_loader):
                    input, target = input.to(device), target.to(device)
                    input_var = torch.autograd.Variable(input)
                    target_var = torch.autograd.Variable(target)
                    target_var_correct_type = get_correct_var_type(dataset, target_var)     

                    with torch.no_grad():
                        val_output = model(input_var.float())
                        val_loss = criterion(val_output, target_var_correct_type, None, None, None, None)

                        val_pred = val_output.data.max(1, keepdim=True)[1]
                        target_indices = get_target_indices(dataset, target)
                        val_correct += val_pred.eq(target_indices).sum().item()
                        val_total += batch_size
                        val_acc = val_correct / val_total

                val_acc_per_drop.append(val_acc)

            val_accs_all_iters.append(val_acc_per_drop)
            model.train()

        if i % val_every_n_iters == 0 or i == len(train_loader)-1:
            val_correct = 0
            val_total = 0
            val_acc = 0

            model.eval()
            #This forces test time dropout
            #model.apply(apply_dropout)

            for j, (input, target) in enumerate(val_loader):
                input, target = input.to(device), target.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)
                target_var_correct_type = get_correct_var_type(dataset, target_var)     

                with torch.no_grad():
                    val_output = model(input_var.float())
                    val_loss = criterion(val_output, target_var_correct_type, None, None, None, None)

                    val_pred = val_output.data.max(1, keepdim=True)[1]
                    target_indices = get_target_indices(dataset, target)
                    val_correct += val_pred.eq(target_indices).sum().item()
                    val_total += batch_size
                    val_acc = val_correct / val_total

            val_accs_all_iters_full.append(val_acc)
                
            model.train()

    return train_acc, train_loss, val_acc, val_loss, train_accs_all_iters, val_accs_all_iters, val_accs_all_iters_full, model


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
