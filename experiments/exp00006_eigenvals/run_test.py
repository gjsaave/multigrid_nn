#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 09:10:52 2022

@author: gjsaave
"""

import numpy as np
from numpy import linalg as LA
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.append("../../")
from logging_utils import save_correlation_list


def compute_inner_prods(weight_update, eigvecs):
    """
    Take inner product of each eigenvector with the weight update and save off each of these inner products into a long vector inner_prods.  The elements of inner_prods have the same indices as the eigenvals

    """
    weight_update_normalized = weight_update / np.linalg.norm(weight_update)
    
    inner_prods = []
    for e_i in range(eigvecs.shape[1]):
        eigvec = eigvecs[:, e_i]
        eigvec_normalized = eigvec / np.linalg.norm(eigvec)
        inner_prods.append(np.inner(weight_update_normalized, eigvec_normalized))
        
    return inner_prods
        

seeds = [1,2]
parent_path = "/Users/gjsaave/school/math599/code/experiments/exp00004_linear_mgdrop_hessian_test/"
exp_dir = "batchsize1_momentum0_weightdecay0_numlayers2_lr0.001_iters200/"
# exp_dir = "batchsize1_momentum0_weightdecay0_numlayers2_lr0.001_iters20/output"
# complex_func = np.abs #deals with the complex valued inner prods and eigenvals. can be np.abs or np.real 

output_savepath = os.path.join(parent_path, "eigen_" + exp_dir, "output")
if not os.path.exists(output_savepath):
    os.makedirs(output_savepath)

plots_savepath = os.path.join(parent_path, "eigen_" + exp_dir, "plots")
if not os.path.exists(plots_savepath):
    os.makedirs(plots_savepath)

corr_x1_with_ex1_all_seeds = []
corr_x2_bar_with_ex1_all_seeds = []
corr_x2_bar_with_ex2_bar_all_seeds = []
corr_x3_with_ex1_all_seeds = []
corr_x3_with_ex3_all_seeds = []

for seed in seeds:

    #FIXME maybe make each of these parameters a variable and plug it in via str
    exp_path = os.path.join(parent_path, "seed" + str(seed) + "_" + exp_dir, "output")
    weight_path = os.path.join(exp_path, "weights.json")
    bias_path = os.path.join(exp_path, "bias.json")
    hessian_path = os.path.join(exp_path, "hessians.json")
    layerwise_hessian_indices_path = os.path.join(exp_path, "layerwise_hessian_indices.json")
    
    with open(weight_path) as f:
        weight = json.load(f)
        
    with open(bias_path) as f:
        bias = json.load(f)
        
    with open(hessian_path) as f:
        hessian_dict = json.load(f)
        
    with open(layerwise_hessian_indices_path) as f:
        layerwise_hessian_indices = json.load(f)
    
    
    num_iters = len(bias["level0"]["x2_bar"]["fc0"]["before"])
    num_layers = len(bias["level0"]["x1"].keys())
    num_levels = len(bias.keys())
    
    #contains the correlation coefficient for each iteration
    corr_x1_with_ex1 = []
    corr_x2_bar_with_ex1 = []
    corr_x2_bar_with_ex2_bar = []
    corr_x3_with_ex1 = []
    corr_x3_with_ex3 = []
    
    for i in range(num_iters):
        
        #Create one long vector for this iteration with all model parameters ordered as [weight layer0, bias layer0, weight layer1....]
        #contains the weight update indexed by level i.e.[[update at level 0], [update at level 1], ...]
        weight_update_x1 = []
        weight_update_x2_bar = []
        weight_update_x3 = []
        #hessian for this iteration ordered as [hessian level0, hessian level1, ...]
        hessian_x1 = []
        hessian_x2_bar = []
        hessian_x3 = []
        eigvecs_x1 = []
        eigvecs_x2_bar = []
        eigvecs_x3 = []
        eigvals_x1 = []
        eigvals_x2_bar = []
        eigvals_x3 = []
        
        for level in range(num_levels):
            weight_before_x1 = []
            weight_after_x1 = []
            weight_before_x2_bar = []
            weight_after_x2_bar = []
            weight_before_x3 = []
            weight_after_x3 = []
            for layer in range(num_layers):
                
                #x1 and x3 does not happen at level 0
                if level != 0:
                    weight_before_x1.append(np.asarray(weight["level" + str(level)]["x1"]["fc" + str(layer)]["before"][i]).flatten())
                    weight_before_x1.append(np.asarray(bias["level" + str(level)]["x1"]["fc" + str(layer)]["before"][i]))
                    weight_after_x1.append(np.asarray(weight["level" + str(level)]["x1"]["fc" + str(layer)]["after"][i]).flatten())
                    weight_after_x1.append(np.asarray(bias["level" + str(level)]["x1"]["fc" + str(layer)]["after"][i]))
                    
                    weight_before_x3.append(np.asarray(weight["level" + str(level)]["x3"]["fc" + str(layer)]["before"][i]).flatten())
                    weight_before_x3.append(np.asarray(bias["level" + str(level)]["x3"]["fc" + str(layer)]["before"][i]))
                    weight_after_x3.append(np.asarray(weight["level" + str(level)]["x3"]["fc" + str(layer)]["after"][i]).flatten())
                    weight_after_x3.append(np.asarray(bias["level" + str(level)]["x3"]["fc" + str(layer)]["after"][i]))
                
                #x2_bar occurs only at level 0
                if level == 0:
                    weight_before_x2_bar.append(np.asarray(weight["level" + str(level)]["x2_bar"]["fc" + str(layer)]["before"][i]).flatten())
                    weight_before_x2_bar.append(np.asarray(bias["level" + str(level)]["x2_bar"]["fc" + str(layer)]["before"][i]))
                    weight_after_x2_bar.append(np.asarray(weight["level" + str(level)]["x2_bar"]["fc" + str(layer)]["after"][i]).flatten())
                    weight_after_x2_bar.append(np.asarray(bias["level" + str(level)]["x2_bar"]["fc" + str(layer)]["after"][i]))        
                        
            #x1 and x3 does not happen at level 0
            if level != 0:
                #Create weight update vector
                weight_before_x1 = np.concatenate(weight_before_x1)
                weight_after_x1 = np.concatenate(weight_after_x1)
                weight_before_x3 = np.concatenate(weight_before_x3)
                weight_after_x3 = np.concatenate(weight_after_x3)
                weight_update_x1.append(weight_after_x1 - weight_before_x1)
                weight_update_x3.append(weight_after_x3 - weight_before_x3)
                weight_update_x2_bar.append([])
                
                #create eigenvals and eigenvecs
                h = np.asarray(hessian_dict["level" + str(level)]["x1"][i])
                i_lower = np.tril_indices(h.shape[0], -1)
                h[i_lower] = h.T[i_lower] #copies upper triangle to lower triangle of matrix to deal with numerical error
                hessian_x1.append(h)
                w, v = LA.eig(h)
                eigvals_x1.append(w)
                eigvecs_x1.append(v)
    
                h = np.asarray(hessian_dict["level" + str(level)]["x3"][i])
                i_lower = np.tril_indices(h.shape[0], -1)
                h[i_lower] = h.T[i_lower]
                hessian_x3.append(h)
                w, v = LA.eig(h)
                eigvals_x3.append(w)
                eigvecs_x3.append(v)
                
                eigvals_x2_bar.append([])
                eigvecs_x2_bar.append([])
            
            #x2_bar occurs only at level 0
            if level == 0:
                #Create weight update vector
                weight_before_x2_bar = np.concatenate(weight_before_x2_bar)
                weight_after_x2_bar = np.concatenate(weight_after_x2_bar)
                weight_update_x1.append([])
                weight_update_x3.append([])
                weight_update_x2_bar.append(weight_after_x2_bar - weight_before_x2_bar)
                
                #create eigenvals and eigenvecs
                h = np.asarray(hessian_dict["level" + str(level)]["x2_bar"][i])
                i_lower = np.tril_indices(h.shape[0], -1)
                h[i_lower] = h.T[i_lower]
                hessian_x2_bar.append(h)
                w, v = LA.eig(h)
                eigvals_x2_bar.append(w)
                eigvecs_x2_bar.append(v)
                
                eigvals_x1.append([])
                eigvals_x3.append([])
                eigvecs_x1.append([])
                eigvecs_x3.append([])
            
        # print("seed ", seed) 
        # print("iter ", i)
        # print("weight_update_x1 ", weight_update_x1[1][0:5])
        # # print("weight_update_x2_bar ", weight_update_x2_bar)
        # print("weight_update_x3 ", weight_update_x3[1][0:5])
        # np.set_printoptions(threshold=sys.maxsize)
        # print("hessian x1 ", hessian_x1[0])
        # print(eigvecs_x1[1])
        # print(eigvals_x1[1])
        # print(eigvals_x2_bar[1])
        # print(eigvals_x3[1])
        # sys.exit()
        # # print("hessian x2 bar ", len(hessian_x2_bar))
        # # print("hessian x3 ", len(hessian_x3))
        # print("------------------------------------------------------")
        
        #Take inner product of each eigenvector with the weight update and save off each of these inner products into a long vector.  Take correlation of this vector with the list containing the eigenvalues.
        #FIXME this assumes we are only using 2 levels for now. The above code should work with more than 2 levels though so if you move to more levels you only need to fix below.
        #Notation: x1_with_ex1 mean inner product of x1 with eigenvecs of x1
        inner_prods_x1_with_ex1 = compute_inner_prods(weight_update_x1[1], eigvecs_x1[1])
        inner_prods_x2_bar_with_ex1 = compute_inner_prods(weight_update_x2_bar[0], eigvecs_x1[1])
        inner_prods_x2_bar_with_ex2_bar = compute_inner_prods(weight_update_x2_bar[0], eigvecs_x2_bar[0])
        inner_prods_x3_with_ex1 = compute_inner_prods(weight_update_x3[1], eigvecs_x1[1])
        inner_prods_x3_with_ex3 = compute_inner_prods(weight_update_x3[1], eigvecs_x3[1])
        
        # print("inner prod x1 ex1 ", inner_prods_x1_with_ex1)
        
        corr_x1_with_ex1.append(np.real(np.corrcoef(inner_prods_x1_with_ex1, eigvals_x1[1])[0][1]))
        corr_x2_bar_with_ex1.append(np.real(np.corrcoef(inner_prods_x2_bar_with_ex1, eigvals_x1[1])[0][1]))
        corr_x2_bar_with_ex2_bar.append(np.real(np.corrcoef(inner_prods_x2_bar_with_ex2_bar, eigvals_x2_bar[0])[0][1]))
        corr_x3_with_ex1.append(np.real(np.corrcoef(inner_prods_x3_with_ex1, eigvals_x1[1])[0][1]))
        corr_x3_with_ex3.append(np.real(np.corrcoef(inner_prods_x3_with_ex3, eigvals_x3[1])[0][1]))
        
        # print("corr x1 ex1 ", corr_x1_with_ex1)


    corr_x1_with_ex1_all_seeds.append(corr_x1_with_ex1)
    corr_x2_bar_with_ex1_all_seeds.append(corr_x2_bar_with_ex1)
    corr_x2_bar_with_ex2_bar_all_seeds.append(corr_x2_bar_with_ex2_bar)
    corr_x3_with_ex1_all_seeds.append(corr_x3_with_ex1)
    corr_x3_with_ex3_all_seeds.append(corr_x3_with_ex3)


save_correlation_list(output_savepath,corr_x1_with_ex1_all_seeds, corr_x2_bar_with_ex1_all_seeds, corr_x2_bar_with_ex2_bar_all_seeds, corr_x3_with_ex1_all_seeds, corr_x3_with_ex3_all_seeds)
    

corr_x1_with_ex1_mean = np.mean(corr_x1_with_ex1_all_seeds, axis=0)
corr_x1_with_ex1_std = np.std(corr_x1_with_ex1_all_seeds, axis=0)
corr_x2_bar_with_ex1_mean = np.mean(corr_x2_bar_with_ex1_all_seeds, axis=0)
corr_x2_bar_with_ex1_std = np.std(corr_x2_bar_with_ex1_all_seeds, axis=0)
corr_x2_bar_with_ex2_bar_mean = np.mean(corr_x2_bar_with_ex2_bar_all_seeds, axis=0)
corr_x2_bar_with_ex2_bar_std = np.std(corr_x2_bar_with_ex2_bar_all_seeds, axis=0)
corr_x3_with_ex1_mean = np.mean(corr_x3_with_ex1_all_seeds, axis=0)
corr_x3_with_ex1_std = np.std(corr_x3_with_ex1_all_seeds, axis=0)
corr_x3_with_ex3_mean = np.mean(corr_x3_with_ex3_all_seeds, axis=0)
corr_x3_with_ex3_std = np.std(corr_x3_with_ex3_all_seeds, axis=0)

plt.plot(list(range(num_iters)), corr_x1_with_ex1_mean, label="fine hessian (x1)/fine weights (x1)")
plt.fill_between(list(range(num_iters)), corr_x1_with_ex1_mean - corr_x1_with_ex1_std, corr_x1_with_ex1_mean + corr_x1_with_ex1_std, alpha=0.5)

plt.plot(list(range(num_iters)), corr_x2_bar_with_ex1_mean, label="fine hessian (x1)/coarse weights (x2_bar)")
plt.fill_between(list(range(num_iters)), corr_x2_bar_with_ex1_mean - corr_x2_bar_with_ex1_std, corr_x2_bar_with_ex1_mean + corr_x2_bar_with_ex1_std,  alpha=0.5)

plt.plot(list(range(num_iters)), corr_x2_bar_with_ex2_bar_mean, label="coarse hessian (x2_bar)/coarse weights (x2_bar)")
plt.fill_between(list(range(num_iters)), corr_x2_bar_with_ex2_bar_mean - corr_x2_bar_with_ex2_bar_std,  corr_x2_bar_with_ex2_bar_mean + corr_x2_bar_with_ex2_bar_std, alpha=0.5)

#These correlations tend to overlap strongly with the x1 correlations
# plt.plot(list(range(num_iters)), corr_x3_with_ex1_mean, label="fine hessian (x1)/fine weights (x3)")
# plt.fill_between(list(range(num_iters)), corr_x3_with_ex1_mean - corr_x3_with_ex1_std, corr_x3_with_ex1_mean + corr_x3_with_ex1_std, alpha=0.5)

# plt.plot(list(range(num_iters)), corr_x3_with_ex3_mean, label="fine hessian (x3)/fine weights (x3)")
# plt.fill_between(list(range(num_iters)), corr_x3_with_ex3_mean - corr_x3_with_ex3_std, corr_x3_with_ex3_mean + corr_x3_with_ex3_std, alpha=0.5)

plt.xlabel("forward/backward passes")
plt.ylabel("correlation")
plt.ylim([-1,1])
plt.title("Pearson correlation of <weight updates, eigenvectors> and eigenval size")
plt.legend()
plt.savefig(os.path.join(plots_savepath, "corr"), dpi=300, bbox_inches="tight")
plt.clf()

# print("corr_x1_with_ex1_mean", corr_x1_with_ex1_mean)
# print("corr_x3_with_ex1_mean", corr_x3_with_ex1_mean)