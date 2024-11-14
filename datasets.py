#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:05:22 2022

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
import torchvision

import matplotlib.pyplot as plt
import math

class PeaksData(Dataset):
    
    def __init__(self, train_or_val, data_path):
        if train_or_val == "train":
            full_data_path = os.path.join(data_path, "features_training.dat")
            self.data = np.genfromtxt(full_data_path)
            full_data_path = os.path.join(data_path, "labels_training.dat")
            self.labels = np.genfromtxt(full_data_path)
        elif train_or_val == "val":
            full_data_path = os.path.join(data_path, "features_validation.dat")
            self.data = np.genfromtxt(full_data_path)
            full_data_path = os.path.join(data_path, "labels_validation.dat")
            self.labels = np.genfromtxt(full_data_path)
            
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_mnist_data(data_path, batch_size, debug=False, data_subset=1):
    trds = torchvision.datasets.MNIST(data_path, train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    if debug:
        indices = list(range(10))
        trds = torch.utils.data.Subset(trds, indices)

    if data_subset < 1.0:
        sub_indices = np.random.randint(0, len(trds), size=int(len(trds)*data_subset))
        trds = torch.utils.data.Subset(trds, sub_indices)

    train_loader = torch.utils.data.DataLoader(trds, batch_size=batch_size, shuffle=True)

    teds = torchvision.datasets.MNIST(data_path, train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    if debug:
        teds = torch.utils.data.Subset(teds, indices)
    
    test_loader = torch.utils.data.DataLoader(teds, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_cifar10_data(data_path, batch_size, debug=False):
    trds = torchvision.datasets.CIFAR10(data_path, train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    if debug:
        indices = list(range(10))
        trds = torch.utils.data.Subset(trds, indices)

    train_loader = torch.utils.data.DataLoader(trds, batch_size=batch_size, shuffle=True)

    teds = torchvision.datasets.CIFAR10(data_path, train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    if debug:
        teds = torch.utils.data.Subset(teds, indices)
    
    test_loader = torch.utils.data.DataLoader(teds, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
