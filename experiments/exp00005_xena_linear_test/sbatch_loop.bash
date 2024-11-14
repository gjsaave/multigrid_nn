#!/bin/bash

#Choose script for running either standard baseline or mgdrop
#srun_script=srun_baseline.bash
srun_script=srun_mgdrop.bash

iters=100000000 #Set this to large number if you don't want the epochs getting cutoff
epochs=2 
batch_size=1 
num_features=2
num_classes=5
num_hidden_nodes=64 
momentum=0 
weight_decay=0 
lr=0.01 
num_layers=2 
num_evals=1 #Currently only used with baseline run, not MGDrop 
drop_rate=0.5 #Currently only used with baseline run, not MGDrop 
seed=30 
model_type=linear

#MGDrop params
levels=1
alpha=2
compute_hessian=False
clear_grads=True
store_weights=False

ntasks_per_node=1 #singleGPU has max 16 cores per node
nodes=1
queue=bigmem-1TB #options are bigmem-1TB, bigmem-3TH, singleGPU, dualGPU
num_processes=NA #not using this right now
mem=32G

CURFOLDER=$(basename "$PWD")
datapath="/users/gjsaave/data/peaks_data" 
savepath="/users/gjsaave/exp_output/${CURFOLDER}/no_hessian"

sbatch --verbose --job-name="run_${CURFOLDER}" --time=16:00:00 --partition=$queue --nodes=$nodes --ntasks-per-node=$ntasks_per_node --mem=$mem ${srun_script} $ntasks_per_node $num_processes --batch_size=$batch_size --epochs=$epochs --lr=$lr --datapath=$datapath --savepath=$savepath --num_hidden_nodes=$num_hidden_nodes --momentum=$momentum --weight_decay=$weight_decay --num_layers=$num_layers --num_evals=$num_evals --drop_rate=$drop_rate --seed=$seed --iters=$iters --num_features=$num_features --num_classes=$num_classes --model_type=$model_type --levels=$levels --alpha=$alpha --compute_hessian=$compute_hessian --clear_grads=$clear_grads --store_weights=$store_weights
 
