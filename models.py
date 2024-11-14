#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:06:02 2022

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
import torch.nn.functional as F

import matplotlib.pyplot as plt
import math
from collections import OrderedDict

class LinearFC_2layer(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden_nodes, drop_rate, freeze=False):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_hidden_nodes)
        self.fc2 = nn.Linear(num_hidden_nodes, num_classes)
        #self.drop1 = nn.Dropout(p=drop_rate)
        self.drop1 = DropoutCustom(p=drop_rate, freeze=freeze)
        
    def forward(self, x):
        out = self.fc1(x)
        out_drop = self.drop1(out)
        out_drop = self.fc2(out_drop)
        
        return out_drop
    
    
class LinearFC(nn.Module):
    def __init__(self, num_layers, num_features, num_classes, num_hidden_nodes, drop_rate, freeze=False):
        super().__init__()

        self.total_num_layers = num_layers
        self.drop_indices = []
        self.fc_indices = []
        self.conv_indices = []
        self.layers_to_coarsen = []
        self.layers_to_coarsen_conv = []
        
        m = OrderedDict()
        m["fc0"] = nn.Linear(num_features, num_hidden_nodes)
        m["drop0"] = DropoutCustom(p=drop_rate, freeze=freeze)
        self.drop_indices.append(0)
        self.fc_indices.append(0)
        self.layers_to_coarsen.append(0)
    
        for l in range(1,num_layers-1):
            m["fc" + str(l)] = nn.Linear(num_hidden_nodes, num_hidden_nodes)
            m["drop" + str(l)] = DropoutCustom(p=drop_rate, freeze=freeze)
            self.drop_indices.append(l)
            self.fc_indices.append(l)
            self.layers_to_coarsen.append(l)
            
        m["fc" + str(num_layers-1)] = nn.Linear(num_hidden_nodes, num_classes)
        m["drop" + str(num_layers-1)] = DropoutCustom(p=0, freeze=freeze)
        self.fc_indices.append(num_layers-1)
        self.drop_indices.append(num_layers-1)
            
        self.model = nn.ModuleDict(m)
        
    def forward(self, x):
        #This flattens the input if it is an image
        if len(x.shape) == 4:
            out = torch.flatten(x, start_dim=1)
        else:
            out = x
            
        out = x
        for l in self.model.keys():
            out = self.model[l](out)
        
        return out


class NonLinearFC(nn.Module):
    def __init__(self, num_layers, num_features, num_classes, num_hidden_nodes, drop_rate, freeze=False):
        super().__init__()

        self.total_num_layers = num_layers
        self.drop_indices = []
        self.fc_indices = []
        self.conv_indices = []
        self.layers_to_coarsen = []
        self.layers_to_coarsen_conv = []
        
        m = OrderedDict()
        m["fc0"] = nn.Linear(num_features, num_hidden_nodes)
        m["drop0"] = DropoutCustom(p=drop_rate, freeze=freeze)
        m["relu0"] = nn.ReLU()
        self.drop_indices.append(0)
        self.fc_indices.append(0)
        self.layers_to_coarsen.append(0)
        
        for l in range(1,num_layers-1):
            m["fc" + str(l)] = nn.Linear(num_hidden_nodes, num_hidden_nodes)
            m["drop" + str(l)] = DropoutCustom(p=drop_rate, freeze=freeze)
            m["relu" + str(l)] = nn.ReLU()
            self.drop_indices.append(l)
            self.fc_indices.append(l)
            self.layers_to_coarsen.append(l)
            
        m["fc" + str(num_layers-1)] = nn.Linear(num_hidden_nodes, num_classes)
        m["drop" + str(num_layers-1)] = DropoutCustom(p=0, freeze=freeze)
        self.fc_indices.append(num_layers-1)
        self.drop_indices.append(num_layers-1)
        self.model = nn.ModuleDict(m)
        
    def forward(self, x):
        #This flattens the input if it is an image
        if len(x.shape) == 4:
            out = torch.flatten(x, start_dim=1)
        else:
            out = x
 
        for l in self.model.keys():
            out = self.model[l](out)
        
        return out


class NonLinearCNN(nn.Module):
    def __init__(self, num_layers, num_features, num_classes, num_hidden_nodes, drop_rate, num_layers_conv, in_channels, out_channels_init, kernel_size, padding, im_size, drop_rate_conv, dilation=1, stride=1, freeze=False):
        super().__init__()

        self.num_layers = num_layers
        self.num_layers_conv = num_layers_conv
        self.num_hidden_nodes = num_hidden_nodes
        self.total_num_layers = num_layers + num_layers_conv
        self.break_counter = 0 #This is a hack to make the forward loop
        self.max_pool_kernel = 2
        self.drop_indices = []
        self.fc_indices = []
        self.conv_indices = []
        self.layers_to_coarsen = []
        self.layers_to_coarsen_conv = []

        m = OrderedDict()
        m["fc0"] = nn.Conv2d(in_channels, out_channels_init, kernel_size=kernel_size, padding=padding)
        m["maxpool0"] = nn.MaxPool2d(self.max_pool_kernel)
        m["drop0"] = DropoutConvCustom(p=0, freeze=freeze)
        m["relu0"] = nn.ReLU()
        self.conv_indices.append(0)
        self.drop_indices.append(0)
        self.break_counter += 3

        out_channels_cur = out_channels_init
        #Note that fc is not an accurate identifier for the conv layers but I'm using it here to make the code easier
        for l in range(1, num_layers_conv):
            m["fc" + str(l)] = nn.Conv2d(out_channels_cur, 2*out_channels_cur, kernel_size=kernel_size, padding=padding)
            m["drop" + str(l)] = DropoutConvCustom(p=drop_rate_conv, freeze=freeze)
            m["maxpool" + str(l)] = nn.MaxPool2d(self.max_pool_kernel)
            m["relu" + str(l)] = nn.ReLU()

            self.drop_indices.append(l)
            self.conv_indices.append(l)
            self.layers_to_coarsen_conv.append(l)
            self.break_counter += 4
            
            out_channels_cur = 2*out_channels_cur

        #This computes the downsampled image size. It is very fragile. If you change the way the layers are organized or any params like dilation, the max pool padding, and stride then you should check it
        if padding == "valid":
            im_size_end = im_size
            padding_amount = 0
            max_pool_padding = 0
            max_pool_dilation = 1
            for l in range(0, num_layers_conv):
                im_size_end = int((im_size_end + 2*padding_amount - dilation*(kernel_size - 1)-1)/stride + 1)
                im_size_end = int((im_size_end + 2*max_pool_padding - max_pool_dilation*(self.max_pool_kernel - 1)-1)/self.max_pool_kernel + 1)
        elif padding == "same":
            im_size_end = im_size
            max_pool_padding = 0
            max_pool_dilation = 1
            for l in range(0, num_layers_conv):
                im_size_end = int((im_size_end + 2*max_pool_padding - max_pool_dilation*(self.max_pool_kernel - 1)-1)/self.max_pool_kernel + 1)

        flatten_size = im_size_end*im_size_end*out_channels_cur
        m["fc" + str(num_layers_conv)] = nn.Linear(flatten_size, num_hidden_nodes)
        m["drop" + str(num_layers_conv)] = DropoutCustom(p=drop_rate, freeze=freeze)
        m["relu" + str(num_layers_conv)] = nn.ReLU()
        self.drop_indices.append(num_layers_conv)
        self.fc_indices.append(num_layers_conv)
        self.layers_to_coarsen.append(num_layers_conv)
        
        for l in range(num_layers_conv+1, num_layers_conv + num_layers-1):
            m["fc" + str(l)] = nn.Linear(num_hidden_nodes, num_hidden_nodes)
            m["drop" + str(l)] = DropoutCustom(p=drop_rate, freeze=freeze)
            m["relu" + str(l)] = nn.ReLU()
            self.drop_indices.append(l)
            self.fc_indices.append(l)
            self.layers_to_coarsen.append(l)
            
        m["fc" + str(num_layers_conv + num_layers-1)] = nn.Linear(num_hidden_nodes, num_classes)
        m["drop" + str(num_layers_conv + num_layers - 1)] = DropoutCustom(p=0, freeze=freeze)
        self.fc_indices.append(num_layers_conv + num_layers - 1)
        self.drop_indices.append(num_layers_conv + num_layers - 1)
        self.model = nn.ModuleDict(m)
        

    def forward(self, x):
        model_keys = list(self.model.keys())
        out = x
        for l in range(0, self.break_counter):
            k = model_keys[l]
            out = self.model[k](out)

        out = torch.flatten(out, 1)

        for l in range(self.break_counter, len(model_keys)):
            k = model_keys[l]
            out = self.model[k](out)

        out = F.log_softmax(out)
        #out = out.type(torch.LongTensor)
        
        return out


class CNNDropHardcoded(torch.nn.Module):
    def __init__(self):
        self.total_num_layers=2
        super(CNNDropHardcoded, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        #self.conv2_drop = DropoutConvCustom(p=0.5, freeze=False)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print("conv1 weights ", self.conv1.weight)
        #x0 = self.conv2_drop(x)
        x0 = x
        x1 = self.conv1(x0)
        x2 = F.relu(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        print("conv x1 ", x1.shape)
        #print("conv x1 ", x1[0, :, 0, 0])
        #print("x0 ", x0)
        print("conv x3 ", x3.shape)
        print("conv x4 ", x4.shape)
        # print("---------------------------")
        # print("conv x2 ", x2.shape)
        # print("conv x2 ", x2[0, :, 0, 0])
        sys.exit()
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    
class DropoutCustom(torch.nn.Module):
    
    def __init__(self, p, freeze):
        """
        freeze: when set to True dropout will use the same self.drop_mat from previous forward call
        """
        super().__init__()
        self.p = p
        self.drop_mat = None
        self.freeze = freeze
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")
            
        self.scale_factor = (1 / (1 - self.p))
            
            
    def forward(self, x):
        if self.training and not self.freeze:
            self.drop_mat = torch.empty(x.size()[1]).uniform_(0, 1) >= self.p
            x = x.mul(self.drop_mat) * self.scale_factor

        elif self.training and self.freeze:
            x = x.mul(self.drop_mat) * self.scale_factor
            
        return x


class DropoutConvCustom(torch.nn.Module):

    def __init__(self, p, freeze):
        """
        freeze: when set to True dropout will use the same self.drop_mat from previous forward call
        """
        super().__init__()
        self.p = p
        self.drop_mat = None
        self.freeze = freeze
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")

        self.scale_factor = (1 / (1 - self.p))


    def forward(self, x):
        if self.training and not self.freeze:
            self.drop_mat = torch.empty(x.size()[1]).uniform_(0, 1) >= self.p
            self.drop_mat = self.drop_mat.unsqueeze(1)
            self.drop_mat = self.drop_mat.unsqueeze(2)
            self.drop_mat = self.drop_mat.unsqueeze(3)
            self.drop_mat = torch.reshape(self.drop_mat, (1, -1, 1, 1))
            self.drop_mat = self.drop_mat.repeat((x.size()[0], 1, x.size()[2], x.size()[3]))
            x = x.mul(self.drop_mat) * self.scale_factor

        elif self.training and self.freeze:
            x = x.mul(self.drop_mat) * self.scale_factor

        return x
    

class CustomMSE(nn.Module):
    def __init__(self, num_layers, vx_term):
        super(CustomMSE, self).__init__()
        #self.criterion = criterion
        self.vx_term = vx_term
        self.num_layers = num_layers
        self.mse_loss = torch.nn.MSELoss()
        
    def MSE(self, output, target):
        #loss = torch.mean((output - target)**2)
        loss = self.mse_loss(output, target)
        return loss

    def forward(self, output, target, x, wgrads, bgrads, level):
        #if we don't want an <v_bar, x_bar> term in the criterion
        if not self.vx_term:
            loss = self.MSE(output, target)
            
        #subtract <v, x> term from criterion
        else:  
            #Note that v was compute on the previous level so we use level + 1
            v_trans_x = 0
            for l in range(self.num_layers):
                #Element-wise multiplication of v and x
                v_times_x_w = torch.mul(wgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1],  x.model["fc"+str(l)].weight)
                v_times_x_w_sum = torch.sum(v_times_x_w)
                v_times_x_b = torch.mul(bgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1], x.model["fc"+str(l)].bias)
                v_times_x_b_sum = torch.sum(v_times_x_b)
                v_trans_x = torch.add(v_trans_x, torch.add(v_times_x_w_sum, v_times_x_b_sum))
                            
            #FIXME with more than 2 levels this may need to be a recursive call
            loss = self.MSE(output, target) - v_trans_x      

        return loss


class CustomNLL(nn.Module):
    def __init__(self, num_layers, vx_term, dataset="peaks"):
        super(CustomNLL, self).__init__()
        #self.criterion = criterion
        self.vx_term = vx_term
        self.num_layers = num_layers
        self.dataset = dataset
        
    def nll(self, output, target):
        #Make target not one hot encoded
        if self.dataset == "peaks":
            _target = (target == 1).nonzero(as_tuple=False)[:,-1]
        else:
            _target = target

        loss = F.nll_loss(output, _target)
        return loss

    def forward(self, output, target, x, wgrads, bgrads, level):
        #if we don't want an <v_bar, x_bar> term in the criterion
        if not self.vx_term:
            loss = self.nll(output, target)
            
        #subtract <v, x> term from criterion
        else:  
            #Note that v was compute on the previous level so we use level + 1
            v_trans_x = 0
            for l in range(self.num_layers):
                #Element-wise multiplication of v and x
                v_times_x_w = torch.mul(wgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1],  x.model["fc"+str(l)].weight)
                v_times_x_w_sum = torch.sum(v_times_x_w)
                v_times_x_b = torch.mul(bgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1], x.model["fc"+str(l)].bias)
                v_times_x_b_sum = torch.sum(v_times_x_b)
                v_trans_x = torch.add(v_trans_x, torch.add(v_times_x_w_sum, v_times_x_b_sum))
                
            
            #FIXME with more than 2 levels this may need to be a recursive call
            loss = self.nll(output, target) - v_trans_x      

        return loss


class CustomCrossEntropy(nn.Module):
    def __init__(self, num_layers, vx_term, dataset="peaks"):
        super(CustomCrossEntropy, self).__init__()
        #self.criterion = criterion
        self.vx_term = vx_term
        self.num_layers = num_layers
        self.dataset = dataset
        self.ce_loss = nn.CrossEntropyLoss()
        
    def ce(self, output, target):
        #Make target not one hot encoded
        if self.dataset == "peaks":
            _target = (target == 1).nonzero(as_tuple=False)[:,-1]
        else:
            _target = target

        loss = self.ce_loss(output, _target)
        return loss

    def forward(self, output, target, x, wgrads, bgrads, level):
        #if we don't want an <v_bar, x_bar> term in the criterion
        if not self.vx_term:
            loss = self.ce(output, target)
            
        #subtract <v, x> term from criterion
        else:  
            #Note that v was compute on the previous level so we use level + 1
            v_trans_x = 0
            for l in range(self.num_layers):
                #Element-wise multiplication of v and x
                v_times_x_w = torch.mul(wgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1],  x.model["fc"+str(l)].weight)
                v_times_x_w_sum = torch.sum(v_times_x_w)
                v_times_x_b = torch.mul(bgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1], x.model["fc"+str(l)].bias)
                v_times_x_b_sum = torch.sum(v_times_x_b)
                v_trans_x = torch.add(v_trans_x, torch.add(v_times_x_w_sum, v_times_x_b_sum))
                
            
            #FIXME with more than 2 levels this may need to be a recursive call
            loss = self.ce(output, target) - v_trans_x                 

        return loss

    
def create_optimizer(model, opt_method, base_lr, momentum, weight_decay):
    if opt_method == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_method == "adagrad":
        optimizer = torch.optim.Adagrad(params=model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif opt_method == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif opt_method == "rmsprop":
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def create_criterion(loss_method, total_num_layers, vx_term, dataset):
    if loss_method == "mse":
        criterion = CustomMSE(num_layers=total_num_layers, vx_term=vx_term)
    elif loss_method == "nll":
        criterion = CustomNLL(num_layers=total_num_layers, vx_term=vx_term, dataset=dataset)
    elif loss_method == "ce":
        criterion = CustomCrossEntropy(num_layers=total_num_layers, vx_term=vx_term, dataset=dataset)

    return criterion
