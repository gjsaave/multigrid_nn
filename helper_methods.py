#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 06:20:57 2022


"""
import torch
import torch.nn as nn
import numpy as np
import argparse


def full_hessian(params_list, loss_fn, outputs, targets, model, wgrads, bgrads, level):
    """
    Parameters
    ----------
    params_list : the named parameters of the model
    loss_fn : the loss function being used by the model
    outputs : the output of the model from the current input
    targets : the true labels of the model

    This method computes a full loss Hessian of a model 
    Each iteration of the loop outputs a part of the hessian. If we think of the model parameters as one long vector then we want to take the grad of each element of the vector with respect to each possible parameter. The vector is ordered starting at the first layer and moving through subsequent layers.  So the vector looks like [ (weight layer1) (bias layer1) (weight layer2) …. ]
    The outer p_i1 loop can be thought of as iterating through each element of this long parameter vector (although it does so by groupings of parameters via either all the weights in a layer or all the biases in a layer).
    For each element of the p_i1 loop we take the grad of this element with respect to every model parameter in the p_i2 loop.The p_i2 loop creates a set of rows of the hessian. 
    The torch.autograd.grad in the innermost loop returns a single partial row. This row can be either the left most indices of the Hessian or it may be appended to a current row. Each iteration contains a set of partial rows and the subsequent iterations contain the next parts of these rows e.g. iteration i=0 will contain the first 4 columns of the first 4 rows and iteration i=1 will contain the next 4 columns of the same 4 rows. After completing all iterations of the i,j loops we have all the hessian elements for the set of rows that the p_i2 loop is currently handling.
    Note that when we take a grad of weight (or bias) with respect to a weight we get a 2-dim matrix whereas if we take grad with respect to a bias we get a 1-dim vector.
    Note that we can extract the layerwise hessian from the full hessian. We just need the indices of the parameters of that layer. Then we can use those indices to extract the necessary block from the full hessian. 
    
    returns:
        layerwise_hessian_indices: contains index pairs of start and stop points for the elements of the hessian for each layer. These index pairs can be used on the full hessian to retrieve a layerwise hessian. Format is [[start_hi, end_hi, start_hj, end_hj]..] where each sub list corresponds to a layer of the model.  The number of sublists is equal to the number of layers in the model. The i iterates over dim 0 and j iterates over dim 1. This process essentially corresponds to getting the bloack matrices along the diagonal of the full hessian.
        hessian: 2d array with format [[dL/dw1dw1 dL/dw1dw2 .... dL/dw1dwn], [dL/dw2dw1 dL/dw2dw2 .... dL/dw2dwn]] where w1 corresponds to the first element in the long parameter vector, w2 corresponds to second element and so on.

    """
    
    num_params = sum(p[1].numel() for p in params_list)
    layerwise_hessian_indices = []    
    hessian = np.zeros(shape=(num_params, num_params))
    loss = loss_fn(outputs, targets, model, wgrads, bgrads, level)
    
    #Assumes each layer has bias. 
    start_hi = 0
    for p_i1 in range(0, len(params_list), 2):
        #get hessian of weights. take grad of param1 with respect to param2
        param1_name = params_list[p_i1][0]
        param1 = params_list[p_i1][1]
        grads_param1 = torch.autograd.grad(loss, param1, retain_graph=True, create_graph=True)
        
        #These 2 p_i2 for loops make up the weights and bias of a single layer
        #Loop looks at every other index to get the weights and find hessian
        start_hj = 0 
        for p_i2 in range(len(params_list)):
            param2_name = params_list[p_i2][0]
            param2 = params_list[p_i2][1]
             
            partial_rows = []
            #each i,j iteration is a partial row
            for i in range(grads_param1[0].size(0)):
                for j in range(grads_param1[0].size(1)):
                    
                    partial_rows.append([torch.autograd.grad(grads_param1[0][i][j], param2, retain_graph=True)[0].numpy().flatten()])
            
            #This block fills in the hessian with the elements we have so far
            partial_rows = np.concatenate(partial_rows, axis=0)
            end_hi = start_hi + partial_rows.shape[0]
            end_hj = start_hj + partial_rows.shape[1]
            hessian[start_hi:end_hi, start_hj:end_hj] = partial_rows
            start_hj = end_hj
        
        layer_start_hi = start_hi
        layer_start_hj = start_hi
        start_hi = end_hi
        start_hj = 0 
        
        #get hessian of biases. take grad of param1 with respect to param2
        param1_name = params_list[p_i1+1][0]
        param1 = params_list[p_i1+1][1]
        grads_param1 = torch.autograd.grad(loss, param1, retain_graph=True, create_graph=True)
        for p_i2 in range(len(params_list)):
            param2_name = params_list[p_i2][0]
            param2 = params_list[p_i2][1]
            
            partial_rows = []
            for i in range(grads_param1[0].size(0)):

                partial_rows.append([torch.autograd.grad(grads_param1[0][i], param2, retain_graph=True)[0].numpy().flatten()])
                    
            #This block fills in the hessian with the elements we have so far
            partial_rows = np.concatenate(partial_rows, axis=0)
            end_hi = start_hi + partial_rows.shape[0]
            end_hj = start_hj + partial_rows.shape[1]
            hessian[start_hi:end_hi, start_hj:end_hj] = partial_rows
            
            #if we are taking grad of params with respect to the same params then this corresponds to
            #the layerwise hessian. At p_i2-1 we have taken grads with respect to weights and biases
            if p_i1 == p_i2-1:
                layerwise_hessian_indices.append([layer_start_hi, end_hi, layer_start_hj, end_hj])   
                
            start_hj = end_hj
        
        start_hi = end_hi
             
    
    return hessian, layerwise_hessian_indices


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

                                                                                                         