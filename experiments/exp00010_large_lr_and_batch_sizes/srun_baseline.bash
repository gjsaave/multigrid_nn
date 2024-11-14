#!/bin/bash

myArray=("$@")
for arg in "${myArray[@]}"; do
   echo "$arg"
done

nodes=$SLURM_JOB_NUM_NODES           # Number of nodes - the number of nodes you have requested (for a list of SLURM environment variables see "man sbatch")
cores=${myArray[0]}                             # Number MPI processes to run on each node (a.k.a. PPN)
                                     # cts1 has 36 cores per node
num_processes=${myArray[1]}

srun python run_test_baseline.py ${myArray[@]:2}
