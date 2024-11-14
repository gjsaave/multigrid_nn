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
import argparse
import time

sys.path.append("../../")
from models import LinearFC, NonLinearFC, CustomMSE, NonLinearCNN, CNNDropHardcoded, create_optimizer, create_criterion
from datasets import PeaksData, get_mnist_data
from mgopt_methods import MGDrop
from logging_utils import save_results, save_mgdrop_stuff, save_args
from train_test_methods import train_and_val, train_and_val_mgdrop
from helper_methods import str2bool

start = time.time()

torch.set_printoptions(precision=4)
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='xena code')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=1, metavar='N')
parser.add_argument('--datapath', type=str, default="/tmp", metavar='N',
                    help='where to find the numpy datasets')
parser.add_argument('--savepath', type=str, default="/tmp", metavar='N',
                    help='where to save off output')
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--num_hidden_nodes', type=int, default=8, help="num hidden")
parser.add_argument('--momentum', type=float, default=0, help="monentum")
parser.add_argument('--weight_decay', type=float, default=0, help="weight decay")
parser.add_argument('--num_layers', type=int, default=2, help="")
parser.add_argument('--num_evals', type=int, default=1, help="")
parser.add_argument('--iters', type=int, default=1000000000, help="number of forward/backward passes before terminating an epoch. Use np.inf if you want to run through all examples in an epoch.")
parser.add_argument('--val_every_n_iters', type=int, default=1000000000, help="get validation accuracy after this many data evaluations. If np.inf then it will evaluate at the end of every epoch as normal.")
parser.add_argument('--drop_rate', type=float, default=0.5, help="")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--levels', type=int, default=1, help="num levels in MGOpt. level is indexed from 0 to args.levels inclusive")
parser.add_argument('--alpha', type=int, default=2, help="Used in the MGOpt line search. Left and right boundary of line search are -alpha and alpha respectively.")
parser.add_argument('--bls_num_test_points', type=int, default=1000, help="Number of test points in the line search between -alpha and alpha")
parser.add_argument('--compute_hessian', type=str2bool, default=False, help="compute the loss hessian for every forward/backward pass")
parser.add_argument('--clear_grads', type=str2bool, default=True, help="clear the gradient storage dictionary after every forward/backward pass. Helps reduce memory usage")
parser.add_argument('--store_weights', type=str2bool, default=False, help="store the weights before and after updates for every forward/backward pass.")
parser.add_argument('--model_type', type=str, default="linear", metavar='N',
                    help='which model to use. Options are linear, nonlinear.')
parser.add_argument('--opt_method', type=str, default="sgd", metavar='N',
                    help='which optimizer to use. choices are sgd, adam, adagrad, rmsprop')

parser.add_argument('--dataset', type=str, default="peaks", metavar="N", help="which dataset to use. Note you also need to set the datapath arg")
parser.add_argument('--num_layers_conv', type=int, default=2, metavar='N', help="number of convolution layers before FCN")
parser.add_argument('--in_channels', type=int, default=1, metavar='N', help="number of channels for input images")
parser.add_argument('--out_channels_init', type=int, default=10, metavar='N', help="number of channels for inner conv layers. Each successive layer doubles this number")
parser.add_argument('--kernel_size', type=int, default=5, metavar='N', help="conv kernel size")
parser.add_argument('--alg', type=str, default="baseline", metavar="N", help="use mgdrop or baseline")
parser.add_argument('--loss_method', type=str, default="mse", metavar='N',
                    help='which criterion to use. choices are mse, nll')

args = parser.parse_args()

iters = args.iters
val_every_n_iters = args.val_every_n_iters
epochs = args.epochs
batch_size = args.batch_size
num_features = args.num_features
num_classes = args.num_classes
num_hidden_nodes = args.num_hidden_nodes
momentum = args.momentum
weight_decay = args.weight_decay
base_lr = args.lr
num_layers=args.num_layers
opt_method=args.opt_method
num_layers_conv = args.num_layers_conv
in_channels = args.in_channels
out_channels_init = args.out_channels_init
kernel_size = args.kernel_size
loss_method = args.loss_method

#Baseline-only params
num_evals =args.num_evals 
drop_rate = args.drop_rate 

#MGDrop params
levels = args.levels
alpha = args.alpha
bls_num_test_points = args.bls_num_test_points
compute_hessian = args.compute_hessian
clear_grads = args.clear_grads
store_weights = args.store_weights

data_path = args.datapath
random_seed = args.seed
torch.manual_seed(random_seed)
model_type = args.model_type
dataset = args.dataset
alg = args.alg
    
savepath = args.savepath
output_parent = "exp_output_" + str(alg)

#Whenever you add in a new parameter you must add the string to both mgdrop and baseline exp_folder
if alg == "baseline":
    exp_folder = "seed" + str(random_seed) + "_batchsize" + str(batch_size) + "_momentum" + str(momentum) + "_weightdecay" + str(weight_decay) + "_numlayers" + str(num_layers) + "_lr" + str(base_lr) + "_iters" + str(iters) + "_numevals" + str(num_evals) + "_droprate" + str(drop_rate)  + "_modeltype" + str(model_type) + "_numhiddennodes" + str(num_hidden_nodes) + "_valeveryniters" + str(val_every_n_iters)  + "_om" + str(opt_method) + "_ds" + str(dataset) + "_nc" + str(num_layers_conv) + "_ic" + str(in_channels) + "_oc" + str(out_channels_init) + "_ks" + str(kernel_size) + "_lm" + str(loss_method)
elif alg == "mgdrop":
    exp_folder = "seed" + str(random_seed) + "_batchsize" + str(batch_size) + "_momentum" + str(momentum) + "_weightdecay" + str(weight_decay) + "_numlayers" + str(num_layers) + "_lr" + str(base_lr) + "_iters" + str(iters) + "_levels" + str(levels) + "_modeltype" + str(model_type) + "_hessian" + str(compute_hessian) + "_cleargrads" + str(clear_grads) + "_storeweights" + str(store_weights) + "_blsnumtestpoints" + str(bls_num_test_points)  + "_numhiddennodes" + str(num_hidden_nodes)  + "_valeveryniters" + str(val_every_n_iters) + "_om" + str(opt_method) + "_ds" + str(dataset) + "_nc" + str(num_layers_conv) + "_ic" + str(in_channels) + "_oc" + str(out_channels_init) + "_ks" + str(kernel_size) + "_lm" + str(loss_method)

# if dataset == "peaks":
#     train_data = PeaksData(train_or_val="train", data_path=data_path)
#     train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
#     val_data = PeaksData(train_or_val="val", data_path=data_path)
#     val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
# elif dataset == "mnist":
#     train_loader, val_loader = get_mnist_data(data_path, batch_size)
    
output_savepath = os.path.join(savepath, output_parent, exp_folder, "output")
if not os.path.exists(output_savepath):
    os.makedirs(output_savepath)
    
plots_savepath = os.path.join(savepath, output_parent, exp_folder, "plots")
if not os.path.exists(plots_savepath):
    os.makedirs(plots_savepath)

_drop_rate = None
mgdrop = None
if alg == "baseline":
    _drop_rate = drop_rate
elif alg == "mgdrop":
    mgdrop = MGDrop(num_layers, levels, alpha, compute_hessian, clear_grads, store_weights, bls_num_test_points, dataset=dataset, opt_method=opt_method, loss_method=loss_method)
    _drop_rate = mgdrop.drop_rates[str(levels)]
    
if model_type == "linear":
    model = LinearFC(num_layers, num_features, num_classes, num_hidden_nodes, drop_rate=_drop_rate)
    
elif model_type == "nonlinear":
    model = NonLinearFC(num_layers, num_features, num_classes, num_hidden_nodes, drop_rate=_drop_rate)

elif model_type == "nonlinear_cnn":
    model = NonLinearCNN(num_layers, num_features, num_classes, num_hidden_nodes, _drop_rate, num_layers_conv, in_channels, out_channels_init, kernel_size)

elif model_type == "cnndrophardcoded":
    model = CNNDropHardcoded()

total_num_layers = model.total_num_layers
criterion = create_criterion(loss_method, total_num_layers, vx_term=False, dataset=dataset)
optimizer = create_optimizer(model, opt_method, base_lr, momentum, weight_decay)

train_accs = []
val_accs = []
train_losses = []
val_losses = []
train_accs_all_iters = []
val_accs_all_iters = []

def train(epoch):
    model.float()
    model.train()
    #for batch_idx, (data, target) in enumerate(train_loader):
    data = np.ones(shape=(1, 4, 12, 12))
    data = torch.from_numpy(data)
    #data = data.type(torch.DoubleTensor)
    #print("data", data.float())
    output = model(data.float())

    
    # loss = F.nll_loss(output, target)
    # loss.backward()
    # optimizer.step()
    # if batch_idx % log_interval == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))
    #   train_losses.append(loss.item())
    #   train_counter.append(
    #     (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    #   torch.save(network.state_dict(), '/results/model.pth')
    #   torch.save(optimizer.state_dict(), '/results/optimizer.pth')
    

train(0)

save_results(output_savepath, epochs, train_accs, val_accs, train_losses, val_losses, train_accs_all_iters, val_accs_all_iters)
save_args(output_savepath, args)
if alg == "mgdrop":
    save_mgdrop_stuff(output_savepath, mgdrop)

plt.plot(list(range(epochs)), train_accs, label="train")
plt.plot(list(range(epochs)), val_accs, label="val")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Peaks data.")
plt.legend()
plt.savefig(os.path.join(plots_savepath, "acc_per_epoch"), dpi=300, bbox_inches="tight")
plt.clf()

plt.plot(list(range(len(train_accs_all_iters))), train_accs_all_iters, label="train")
plt.plot(list(range(len(val_accs_all_iters))), val_accs_all_iters, label="val")
plt.xlabel("iterations")
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

end = time.time()
print("Total run time: ", end - start)