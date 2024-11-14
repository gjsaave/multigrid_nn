#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 04:42:52 2022

@author: gjsaave
"""
import os
import json
import numpy as np
import torch
import sys

def save_results(savepath, epochs, train_accs, val_accs, train_losses, val_losses, train_accs_all_iters, val_accs_all_iters, val_iters_list):

    results_dict = {"epochs": epochs,
                    "train_accs": train_accs,
                    "val_accs": val_accs,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accs_all_iters": train_accs_all_iters,
                    "val_accs_all_iters": val_accs_all_iters,
                    "val_iters_list": val_iters_list
                    }

    with open(os.path.join(savepath, "results.json"), "w+") as f:
        json.dump(results_dict, f, indent=2)

    # model_savepath = os.path.join(savepath, "model.pt")
    # torch.save(model.state_dict(), model_savepath)
    

def save_args(savepath, args):
    with open(os.path.join(savepath, "args.json"), "w+") as f:
        json.dump(vars(args), f, indent=2)


def save_mgdrop_stuff(savepath, mgdrop):
    
    d = mgdrop.weight_dict 
    with open(os.path.join(savepath, "weights.json"),"w+") as f:
        json.dump(d,f, cls=NumpyEncoder)
    
    d = mgdrop.bias_dict 
    with open(os.path.join(savepath, "bias.json"),"w+") as f:
        json.dump(d,f, cls=NumpyEncoder)
        
    d = mgdrop.wgrads 
    with open(os.path.join(savepath, "wgrads.json"),"w+") as f:
        json.dump(d,f, cls=NumpyEncoder)
        
    d = mgdrop.bgrads
    with open(os.path.join(savepath, "bgrads.json"),"w+") as f:
        json.dump(d,f, cls=NumpyEncoder)
        
    d = mgdrop.hessian_dict 
    with open(os.path.join(savepath, "hessians.json"),"w+") as f:
        json.dump(d,f, cls=NumpyEncoder)
        
    d = mgdrop.layerwise_hessian_indices_dict
    with open(os.path.join(savepath, "layerwise_hessian_indices.json"),"w+") as f:
        json.dump(d,f, cls=NumpyEncoder)
        
    d = mgdrop.drop_rates 
    with open(os.path.join(savepath, "drop_rates.json"),"w+") as f:
        json.dump(d,f, cls=NumpyEncoder)
        

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def save_correlation_list(savepath,corr_x1_with_ex1_all_seeds, corr_x2_bar_with_ex1_all_seeds, corr_x2_bar_with_ex2_bar_all_seeds, corr_x3_with_ex1_all_seeds, corr_x3_with_ex3_all_seeds):
    
    d = {"corr_x1_with_ex1_all_seeds": corr_x1_with_ex1_all_seeds,
         "corr_x2_bar_with_ex1_all_seeds,":corr_x2_bar_with_ex1_all_seeds,
         "corr_x2_bar_with_ex2_bar_all_seeds":corr_x2_bar_with_ex2_bar_all_seeds,
         "corr_x3_with_ex1_all_seeds":corr_x3_with_ex1_all_seeds, 
         "corr_x3_with_ex3_all_seeds":corr_x3_with_ex3_all_seeds}
    
    with open(os.path.join(savepath, "correlations.json"), "w+") as f:
        json.dump(d, f, indent=2)
    
    
