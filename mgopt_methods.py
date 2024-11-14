#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:11:29 2022

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
import time

sys.path.append("./")
from models import CustomMSE, create_optimizer, create_criterion
from helper_methods import full_hessian


class MGDrop():
    
    def __init__(self, total_num_layers, levels, alpha, compute_hessian, clear_grads, store_weights, bls_num_test_points=10, dataset="peaks", opt_method="sgd", loss_method="mse", model_type="nonlinear", drop_rate=0.5, drop_rate_conv=0, device="cpu", vx_term=True, same_drop_all_levels_fc=False, same_drop_all_levels_conv=False):
        """
        wgrads, bgrads: stores the gradients for weight and biases
        clear_grads: clears out the gradient dictionary after every iteration to save on memory
        """
        self.total_num_layers = total_num_layers
        self.levels= levels
        self.alpha = alpha
        self.compute_hessian = compute_hessian
        self.hessian_dict = self.make_hessian_dict(levels)
        self.layerwise_hessian_indices_dict = self.make_hessian_dict(levels)
        self.store_weights = store_weights
        self.weight_dict = self.make_weight_dict(levels, total_num_layers)
        self.bias_dict = self.make_weight_dict(levels, total_num_layers)
        self.drop_rates = self.make_dropout_dict(levels, drop_rate, same_drop_all_levels_fc)
        self.drop_rates_conv = self.make_dropout_dict(levels, drop_rate_conv, same_drop_all_levels_conv)
        self.wgrads, self.bgrads = self.make_grad_dict(levels, total_num_layers)
        self.clear_grads = clear_grads
        self.bls_num_test_points = bls_num_test_points
        self.dataset = dataset
        self.opt_method = opt_method
        self.loss_method = loss_method
        self.model_type = model_type
        self.device = device
        self.vx_term = vx_term

    def make_weight_dict(self, levels, num_layers):
        """
        creates a dictionary that stores the weights at different levels and different steps of the MGOPT algorithm. The terms x1, x2_bar, etc. refer to the different steps in the Nash algorithm. Note we use a similar naming convention to the grad dictionaries. "before" and "after" refer to before or after the optimization step for this particular forward/backward pass. The dictionary will store the weights from all forward/backward passes

        """
        weight_dict = {}
        for i in range(levels+1):
            weight_dict["level" + str(i)] = {}
            for x in ["x0", "x1", "x1_bar", "x2_bar", "x3"]:
                weight_dict["level" + str(i)][x] = {}
                for l in range(num_layers):
                    weight_dict["level" + str(i)][x]["fc" + str(l)] = {}   
                    for p in ["before", "after"]:
                        weight_dict["level" + str(i)][x]["fc" + str(l)][p] = []
                
        return weight_dict
        
    
    def make_hessian_dict(self, levels):
        """
        creates a dictionary that stores the hessian of the loss at different levels and different steps of the MGOPT algorithm. The terms x1, x2_bar, etc. refer to the different steps in the Nash algorithm. Note we use a similar naming convention to the grad dictionaries. The dictionary will store the weights from all forward/backward passes

        """
        hessian_dict = {}
        for i in range(levels+1):
            hessian_dict["level" + str(i)] = {}
            for x in ["x0", "x1", "x1_bar", "x2_bar", "x3"]:
                hessian_dict["level" + str(i)][x] = []
                
        return hessian_dict
        
        
    def make_grad_dict(self, levels, num_layers):
        wgrads = {}
        bgrads = {}
        layer_dict = {}
        for l in range(num_layers):
            layer_dict["fc" + str(l)] = []
            
        for i in range(levels+1):      
            wgrads["level" + str(i)] = {}
            bgrads["level" + str(i)] = {}
            
            #Naming convention is based on steps from Nash MGOPT. These rep gradients from different parts of algorithm
            #g1, g1_bar, v_bar are from Step 3.
            #Note that the layer for each quantity (and v_bar) indicates the inputs into the nodes for that layers 
            #e.g. wgrads[0][v_bar]["fc1"] are the v_bar for the inputs to fc1 and "fc2" are the v_bar for the inputs to fc2
            #g2_bar is from recursive call in Step 4. Note this can also represent gradient from Step 1 if if we use levels=0
            #g3 is from last step
            #e2 is error from step 5
            for g in ["g0", "g1", "g1_bar", "v_bar", "g2_bar", "g3", "e2"]: 
                wgrads["level" + str(i)][g] = copy.deepcopy(layer_dict)
                bgrads["level" + str(i)][g] = copy.deepcopy(layer_dict)
                
        return wgrads, bgrads
    
    
    def make_dropout_dict(self, levels, drop_rate, same_drop_all_levels):
        """
        creates dictionary that controls dropouut values for each level of mgopt
        Finest level has 0 dropout and next coarse level has 0.5. 
        Otherwise each level has dropout of (dropout val at next finest level + (1 - dropout val at next finest level) * 0.5) 
        e.g., for levels=2 we will have {"2": 0, "1" : 0.5, "0": 0.75}
        This should have the effect of each coarsening creating a network with approximately half the 
        nodes of the current network
        """
        # if levels == 0:
        #     drop_rates = {str(levels): 0}
        # else:
        #     drop_rates = {str(levels): 0, str(levels-1): drop_rate}

        # for i in range(levels-2, -1, -1):
        #     if drop_rate != 0:
        #         drop_rates[str(i)] = (1 - drop_rates[str(i+1)])*0.5 + drop_rates[str(i+1)]
        #     else:
        #         drop_rates[str(i)] = 0

        if same_drop_all_levels:
            drop_rates = {str(levels): drop_rate}

            for i in range(levels-1, -1, -1):
                drop_rates[str(i)] = drop_rate

        else:
            drop_rates = {str(levels): drop_rate}

            for i in range(levels-1, -1, -1):
                drop_rates[str(i)] = (1 - drop_rates[str(i+1)])*0.5 + drop_rates[str(i+1)]


        return drop_rates

            
    def forward_backward(self, model, criterion, optimizer, input_var, target_var, level, grad_str, x_str, opt_step):
        """
        Does a forward, backward pass and saves the gradients into a dictionary
        """
        
        model.train()
        optimizer.zero_grad()
        _input = input_var.float()
        output = model(_input)
        loss = criterion(output, target_var, model, self.wgrads, self.bgrads, level)

        # if level == 0:
        #     criterion2 = create_criterion(self.loss_method, model.total_num_layers, vx_term=False, dataset=self.dataset)
        #     loss2 = criterion2(output, target_var, model, self.wgrads, self.bgrads, level)

        #     print("loss 2 ", loss2)
        
        #save weights and biases
        if self.store_weights:
            for l in range(self.total_num_layers):
                self.weight_dict["level" + str(level)][x_str]["fc" + str(l)]["before"].append(copy.deepcopy(model.model["fc" + str(l)].weight))
                self.bias_dict["level" + str(level)][x_str]["fc" + str(l)]["before"].append(copy.deepcopy(model.model["fc" + str(l)].bias))
           
        #save off hessian
        if self.compute_hessian:
            params_list = list(model.named_parameters())
            h, lh = full_hessian(params_list, criterion, output, target_var, model, self.wgrads, self.bgrads, level)
            self.hessian_dict["level" + str(level)][x_str].append(copy.deepcopy(h))
            self.layerwise_hessian_indices_dict["level" + str(level)][x_str].append(copy.deepcopy(lh))

        # if level == 1:
        #     loss.backward()
        # else:
        #     loss2.backward()

        loss.backward()
           
        #Save off grads
        for l in range(self.total_num_layers):
            self.wgrads["level" + str(level)][grad_str]["fc" + str(l)].append(copy.deepcopy(model.model["fc" + str(l)].weight.grad))
            self.bgrads["level" + str(level)][grad_str]["fc" + str(l)].append(copy.deepcopy(model.model["fc" + str(l)].bias.grad))

        # print("drop ", model.model["drop0"].drop_mat)
        # print("weights ", model.model["fc0"].weight)
        # print("grads ", model.model["fc0"].weight.grad)

        # if level == 0:
        #     print("grads ", model.model["fc1"].weight.grad)
        #     print("vbar ", self.wgrads["level1"]["v_bar"]["fc1"][-1])
        #     sys.exit()

        
        #FIXME this introduces numerical error on x2_bar updates. We may need to explicitly zero out updates afterward
        if opt_step:      
            optimizer.step()
        
        if self.store_weights:
            for l in range(self.total_num_layers):
                self.weight_dict["level" + str(level)][x_str]["fc" + str(l)]["after"].append(copy.deepcopy(model.model["fc" + str(l)].weight))
                self.bias_dict["level" + str(level)][x_str]["fc" + str(l)]["after"].append(copy.deepcopy(model.model["fc" + str(l)].bias))
            
        return model, output, loss
    
    
    def compute_error(self, x1_bar, x2_bar, level):
        """
        assumes we are simply injecting coarse grid error. There is no interpolation
        e2 contains a list of the errors for each layer. The index of each element in 
        the list is the training iteration and the element is the error i.e index 0 is error for first iteration and
        -1 is the error for the current iteration
        """
        
        #FIXME do i need to use deepcopy here? PyTorch doesn't support it here automatically for some reason
        for l in range(self.total_num_layers):
            self.wgrads["level" + str(level)]["e2"]["fc" + str(l)].append((x2_bar.model["fc" + str(l)].weight - x1_bar.model["fc" + str(l)].weight))
            self.bgrads["level" + str(level)]["e2"]["fc" + str(l)].append((x2_bar.model["fc" + str(l)].bias - x1_bar.model["fc" + str(l)].bias))

    
    def format_drop_mat(self, drop_mat):
        """
        This method reformats the drop matrix to be 1d. This is necessary because the conv drop matrix is multiple dimensions
        """

        if len(list(drop_mat.shape)) == 4:
            drop_mat = copy.deepcopy(drop_mat[0,:,0,0])

        return drop_mat

            
    def compute_vbar(self, x1_bar, level):
        """
        compute vbar for all layers. Note that dropout for a layer effects both the inputs for the current layer as well as the inputs for the next layer. Thus, when coarsening via I_h_H we must explicitly coarsen the current layer as well as the next layer.
        """
        v_bar_w = {}
        v_bar_b = {}
        drop_indices = x1_bar.drop_indices
        all_indices = x1_bar.conv_indices + x1_bar.fc_indices
        ##For debugging
        # for l in range(num_layers):
        #     print("level ", l)
        #     print("g1", wgrads["level" + str(level)]["g1"]["fc"+ str(l)][-1])
        #     print("g1 bar ", wgrads["level" + str(level)]["g1_bar"]["fc" + str(l)][-1])
        #     print("----------------------------------------")
        
        #Coarsen weights based on dropout's effect on the inputs for the current layer
        #Assumes no dropout on last layer
        for l in all_indices[:-1]:
            drop_mat = self.format_drop_mat(x1_bar.model["drop" + str(l)].drop_mat)
            I_h_H_prod_g1 = np.asarray(copy.deepcopy(self.wgrads["level" + str(level)]["g1"]["fc"+ str(l)][-1]))
           
            for i in range(len(drop_mat)):
                #if node is dropped out, zero out grads associated with it
                if not drop_mat[i]:
                    I_h_H_prod_g1[i] = 0    
            self.wgrads["level" + str(level)]["v_bar"]["fc" + str(l)].append(copy.deepcopy(self.wgrads["level" + str(level)]["g1_bar"]["fc" + str(l)][-1] - I_h_H_prod_g1))
            
        #Coarsen weights based on dropout's effect on the inputs for the next layer
        for l in all_indices[1:]:
            drop_mat = self.format_drop_mat(x1_bar.model["drop" + str(l-1)].drop_mat)
            I_h_H_prod_g1 = np.asarray(copy.deepcopy(self.wgrads["level" + str(level)]["g1"]["fc"+ str(l)][-1]))
            for i in range(len(drop_mat)):
                #if node is dropped out, zero out grads associated with it
                if not drop_mat[i]:
                    I_h_H_prod_g1[:, i] = 0  
            self.wgrads["level" + str(level)]["v_bar"]["fc" + str(l)].append(copy.deepcopy(self.wgrads["level" + str(level)]["g1_bar"]["fc" + str(l)][-1] - I_h_H_prod_g1))
            
        #Coarsen biases
        #Assumes no dropout on last layer
        for l in all_indices[:]:
            drop_mat = self.format_drop_mat(x1_bar.model["drop" + str(l)].drop_mat)
            I_h_H_prod_g1 = np.asarray(copy.deepcopy(self.bgrads["level" + str(level)]["g1"]["fc" + str(l)][-1]))
            for i in range(len(drop_mat)):
                #if node is dropped out, zero out grads associated with it
                if not drop_mat[i]:
                    I_h_H_prod_g1[i] = 0  
            self.bgrads["level" + str(level)]["v_bar"]["fc" + str(l)].append(copy.deepcopy(self.bgrads["level" + str(level)]["g1_bar"]["fc" + str(l)][-1] - I_h_H_prod_g1))

        #Since we don't coarsen the last layer there is no change in the biases. So we use zeros for v_bar.
        # self.bgrads["level" + str(level)]["v_bar"]["fc" + str(fc_indices[-1])].append(copy.deepcopy(np.zeros(shape = np.asarray(self.bgrads["level" + str(level)]["g1_bar"]["fc" + str(fc_indices[-1])][-1]).shape)))
        
        
    def basic_line_search(self, x1, input_var, target_var, criterion, level, alpha_left, alpha_right, num_test_points):
        """
        Takes in interval left, right and creates num_test_points number of points
        """
        x2 = copy.deepcopy(x1)
        best_x2 = None
        
        if num_test_points == 0:
            for l in range(self.total_num_layers):
                w = self.wgrads["level" + str(level)]["e2"]["fc" + str(l)][-1]
                b = self.bgrads["level" + str(level)]["e2"]["fc" + str(l)][-1]
                x2.model["fc" + str(l)].weight = torch.nn.Parameter(x1.model["fc" + str(l)].weight + w)
                x2.model["fc" + str(l)].bias = torch.nn.Parameter(x1.model["fc" + str(l)].bias + b)

            best_x2 = x2

        else:
            grid = np.linspace(alpha_left, alpha_right, num=num_test_points)
            best_obj = float("inf")
            for i in range(len(grid)):
                alpha = grid[i]

                for l in range(self.total_num_layers):
                    w = self.wgrads["level" + str(level)]["e2"]["fc" + str(l)][-1]
                    b = self.bgrads["level" + str(level)]["e2"]["fc" + str(l)][-1]
                    x2.model["fc" + str(l)].weight = torch.nn.Parameter(x1.model["fc" + str(l)].weight + alpha * w)
                    x2.model["fc" + str(l)].bias = torch.nn.Parameter(x1.model["fc" + str(l)].bias + alpha * b)

                output = x2(input_var.float())
                obj = criterion(output, target_var, x2, self.wgrads, self.bgrads, level)

                #FIXME note that this will not work with loss functions that allow negative objective values
                if obj < best_obj:
                    best_x2 = copy.deepcopy(x2)
                    best_obj = obj
                    #print("alpha", alpha)
                    #print("best obj", best_obj)

                
        return best_x2
    
    
    def make_x1bar(self, x1, drop_rate, drop_rate_conv):
        """
        create x1_bar model.
    
        """
        x1_bar = copy.deepcopy(x1)
        for l in x1_bar.layers_to_coarsen_conv:
            x1_bar.model["drop" + str(l)].p = drop_rate_conv
        for l in x1_bar.layers_to_coarsen:
            x1_bar.model["drop" + str(l)].p = drop_rate
            
        return x1_bar
    
    
    def make_x2bar(self, x1_bar, freeze):
        """
        create x2_bar model.  Assumes that "freeze" is the same for every layer
    
        """
        x2_bar = copy.deepcopy(x1_bar)
        for l in x2_bar.drop_indices:
            x2_bar.model["drop" + str(l)].freeze = freeze
            
        return x2_bar

    
    def mgopt(self, x, input_var, target_var, criterion, optimizer, level, base_lr, momentum, weight_decay, N_c):
        """
        drop_rates: dictionary with drop rate (or coarsening factor) per level e.g., {"0": 0.5, "1": 0}
        wgrads, bgrads: holds gradients for weights and biases for each level and layer
        level: highest is fine level, lowest is coarse
        """
        
        #If coarsest level, train network with dropout, no <v,x> correction
        if level == 0:
            #Note we use grad_str="g2" since this is usually the end of the recursive call in Step 4
            x1 = x
            for N_c_iter in range(N_c):
                x1, output1, loss1 = self.forward_backward(x1, criterion, optimizer, input_var, target_var, level, grad_str="g2_bar", x_str="x2_bar", opt_step=True)

            return x1, optimizer, output1, loss1
        
        #If not coarsest level
        else:
            #Step 1 Train network to obtain fine network x1
            # start = time.time()
            x1, output1, loss1 = self.forward_backward(x, criterion, optimizer, input_var, target_var, level, grad_str="g0", x_str="x0", opt_step=True)
            # print("x1 for back ", time.time() - start) 
            
            #Step 2 Create coarsened network x1_bar
            #Note that PyTorch Dropout zeroes out node outputs, not weights. So we don't explictly coarsen
            #x1 by zeroing weights. Instead dropout will coarsen implicitly when forward is called.
            # start = time.time()
            x1_bar = self.make_x1bar(x1, self.drop_rates[str(level-1)], self.drop_rates_conv[str(level-1)])
            x1_bar.to(self.device)
            # print("make x1 bar ", time.time() - start)            
            
            #This optimizer is never used but I'm making it for consistency and for ease of passing function parameters around
            optimizer_dummy = create_optimizer(x1_bar, self.opt_method, base_lr, momentum, weight_decay)

            #Compute g1
            x1, output1, loss1 = self.forward_backward(x1, criterion, optimizer_dummy, input_var, target_var, level, grad_str="g1", x_str="x1", opt_step=False)
    
            # start = time.time()
            x1_bar, output1_bar, loss1_bar = self.forward_backward(x1_bar, criterion, optimizer_dummy, input_var, target_var, level, grad_str="g1_bar", x_str="x1_bar", opt_step=False)
            # print("x1 bar ", time.time() - start)
            
            # start = time.time()
            self.compute_vbar(x1_bar, level)
            # print("compute vbar ", time.time() - start)
            
            # start = time.time()
            criterion_bar = create_criterion(self.loss_method, x1_bar.total_num_layers, vx_term=self.vx_term, dataset=self.dataset)
            # print("criterion bar ", time.time() - start)
            
            #Step 4 - call mgopt/coarse optimization problem.
            #make x1_bar copy. this copy is needed so we can retain the original x1_bar
            #Freeze dropout so same dropout matrix is used on coarse grid solve as was used for x1_bar
            #FIXME We may need to make freeze a param in mgopt call for future experiments
            x2_bar = self.make_x2bar(x1_bar, freeze=True)
            x2_bar.to(self.device)
            optimizer2_bar = create_optimizer(x2_bar, self.opt_method, base_lr, momentum, weight_decay)
            
            # start = time.time()
            x2_bar, optimizer2_bar, output2_bar, loss2_bar = self.mgopt(x2_bar, input_var, target_var, criterion_bar, optimizer2_bar, level-1, base_lr, momentum, weight_decay, N_c)     
            # print("x2 bar ", time.time() - start)
        
            # start = time.time()
            self.compute_error(x1_bar, x2_bar, level)
            # print("compute error ", time.time() - start)
            
            # start = time.time()
            x2 = self.basic_line_search(x1, input_var, target_var, criterion,level, alpha_left=-1*self.alpha, alpha_right=self.alpha, num_test_points=self.bls_num_test_points)
            # print("line search ", time.time() - start)
            
            #Final forward/backward pass
            x3 = copy.deepcopy(x2)
            x3.to(self.device)
            optimizer3 = create_optimizer(x3, self.opt_method, base_lr, momentum, weight_decay)
            # start = time.time()
            x3, output3, loss3 = self.forward_backward(x3, criterion, optimizer3, input_var, target_var, level, grad_str="g3", x_str="x3", opt_step=True)
            # print("x3 ", time.time() - start)
            
            if self.clear_grads:
                del self.wgrads, self.bgrads
                self.wgrads, self.bgrads = self.make_grad_dict(self.levels, self.total_num_layers)
                
            # print("--------------------------------------------------")
            # print("--------------------------------------------------")
            return x3, optimizer3, output3, loss3


    def smgopt(self, x, input_var, target_var, criterion, optimizer, level, base_lr, momentum, weight_decay, data_indices, data_idx, N_c):
        """
        drop_rates: dictionary with drop rate (or coarsening factor) per level e.g., {"0": 0.5, "1": 0}
        wgrads, bgrads: holds gradients for weights and biases for each level and layer
        level: highest is fine level, lowest is coarse
        """
        
        #If coarsest level, train network with dropout, no <v,x> correction
        if level == 0:
            #Note we use grad_str="g2" since this is usually the end of the recursive call in Step 4
            data_start = data_indices[data_idx][0]
            data_end = data_indices[data_idx][1]
            _input_var = input_var[data_start:data_end]
            _target_var = target_var[data_start:data_end]
            data_idx += 1
            x1 = x
            for N_c_iter in range(N_c):
                x1, output1, loss1 = self.forward_backward(x1, criterion, optimizer, _input_var, _target_var, level, grad_str="g2_bar", x_str="x2_bar", opt_step=True)
            return x1, optimizer, output1, loss1, data_idx
        
        #If not coarsest level
        else:
            #Step 1 Train network to obtain fine network x1
            # start = time.time()
            data_start = data_indices[data_idx][0]
            data_end = data_indices[data_idx][1]
            _input_var = input_var[data_start:data_end]
            _target_var = target_var[data_start:data_end]
            data_idx += 1
            x1, output1, loss1 = self.forward_backward(x, criterion, optimizer, _input_var, _target_var, level, grad_str="g0", x_str="x0", opt_step=True)
            # print("x1 for back ", time.time() - start) 
            
            #Step 2 Create coarsened network x1_bar
            #Note that PyTorch Dropout zeroes out node outputs, not weights. So we don't explictly coarsen
            #x1 by zeroing weights. Instead dropout will coarsen implicitly when forward is called.
            # start = time.time()
            x1_bar = self.make_x1bar(x1, self.drop_rates[str(level-1)], self.drop_rates_conv[str(level-1)])
            x1_bar.to(self.device)
            # print("make x1 bar ", time.time() - start)            
            
            #This optimizer is never used but I'm making it for consistency and for ease of passing function parameters around
            optimizer_dummy = create_optimizer(x1_bar, self.opt_method, base_lr, momentum, weight_decay)

            #Compute g1
            data_start = data_indices[data_idx][0]
            data_end = data_indices[data_idx][1]
            _input_var = input_var[data_start:data_end]
            _target_var = target_var[data_start:data_end]
            x1, output1, loss1 = self.forward_backward(x1, criterion, optimizer_dummy, _input_var, _target_var, level, grad_str="g1", x_str="x1", opt_step=False)
    
            # start = time.time()
            #Compute g1_bar
            x1_bar, output1_bar, loss1_bar = self.forward_backward(x1_bar, criterion, optimizer_dummy, _input_var, _target_var, level, grad_str="g1_bar", x_str="x1_bar", opt_step=False)
            # print("x1 bar ", time.time() - start)
            
            # start = time.time()
            self.compute_vbar(x1_bar, level)
            # print("compute vbar ", time.time() - start)
            
            # start = time.time()
            criterion_bar = create_criterion(self.loss_method, x1_bar.total_num_layers, vx_term=self.vx_term, dataset=self.dataset)
            # print("criterion bar ", time.time() - start)
            
            #Step 4 - call mgopt/coarse optimization problem.
            #make x1_bar copy. this copy is needed so we can retain the original x1_bar
            #Freeze dropout so same dropout matrix is used on coarse grid solve as was used for x1_bar
            #FIXME We may need to make freeze a param in mgopt call for future experiments
            x2_bar = self.make_x2bar(x1_bar, freeze=True)
            x2_bar.to(self.device)
            optimizer2_bar = create_optimizer(x2_bar, self.opt_method, base_lr, momentum, weight_decay)
            
            # start = time.time()
            x2_bar, optimizer2_bar, output2_bar, loss2_bar, data_idx = self.smgopt(x2_bar, input_var, target_var, criterion_bar, optimizer2_bar, level-1, base_lr, momentum, weight_decay, data_indices, data_idx, N_c)     
            # print("x2 bar ", time.time() - start)
        
            # start = time.time()
            self.compute_error(x1_bar, x2_bar, level)
            # print("compute error ", time.time() - start)
            
            # start = time.time()
            x2 = self.basic_line_search(x1, _input_var, _target_var, criterion,level, alpha_left=-1*self.alpha, alpha_right=self.alpha, num_test_points=self.bls_num_test_points)
            # print("line search ", time.time() - start)
            
            #Final forward/backward pass
            x3 = copy.deepcopy(x2)
            x3.to(self.device)
            optimizer3 = create_optimizer(x3, self.opt_method, base_lr, momentum, weight_decay)
            # start = time.time()
            data_start = data_indices[data_idx][0]
            data_end = data_indices[data_idx][1]
            _input_var = input_var[data_start:data_end]
            _target_var = target_var[data_start:data_end]
            data_idx += 1
            x3, output3, loss3 = self.forward_backward(x3, criterion, optimizer3, _input_var, _target_var, level, grad_str="g3", x_str="x3", opt_step=True)
            # print("x3 ", time.time() - start)
            
            if self.clear_grads:
                del self.wgrads, self.bgrads
                self.wgrads, self.bgrads = self.make_grad_dict(self.levels, self.total_num_layers)
                
            # print("--------------------------------------------------")
            # print("--------------------------------------------------")
            return x3, optimizer3, output3, loss3, data_idx
