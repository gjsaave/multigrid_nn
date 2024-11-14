# Multilevel Optimization with Dropout for Neural Networks

This repository contains the code and resources used in the research for the thesis *Multilevel Optimization with Dropout for Neural Networks*, which explores innovative methods for optimizing neural networks. This work was conducted as part of a Master’s thesis at the University of New Mexico.

## Overview

### Purpose
The project addresses the challenges of training large neural networks, which are computationally intensive and can suffer from overfitting when they become too large relative to the dataset. The primary goal is to introduce and evaluate two novel optimization algorithms, **MGDrop** and **SMGDrop**, that leverage multigrid optimization principles with dropout as a coarsening mechanism. These algorithms aim to reduce computational load and enhance generalization performance by integrating information from smaller sub-networks.

### Key Concepts
- **Multigrid Optimization (MG/OPT)**: A multilevel approach that transfers information between finer and coarser representations, generally applied in fields requiring heavy computations like partial differential equations.
- **Dropout Coarsening**: A technique used to create a reduced or “coarser” version of the neural network by stochastically removing nodes. This coarser model retains essential structure but is computationally lighter to optimize.

### Proposed Algorithms
1. **MGDrop**: Uses multigrid optimization with dropout to define smaller sub-problems, focusing on coarse representations of the network.
2. **SMGDrop**: A stochastic variant of MGDrop, which increases regularization by introducing a new data batch for each unique optimization step within the multigrid cycle.

## Key Experiments

### Performance Evaluation
The repository includes code for several experiments to evaluate the effectiveness of MGDrop and SMGDrop in various settings:
- **Effect of Network Depth and Width**: Examines how the algorithms perform on networks with varying layers and nodes per layer, demonstrating strong performance particularly in deep or wide networks.
- **Comparison of Dropout Rates**: Tests various dropout rates to determine their impact on both MGDrop and SMGDrop, comparing with standard dropout as well as a no-dropout baseline.
- **Data Variability and Generalization**: Studies performance on smaller training sets, where overfitting is more likely, to demonstrate SMGDrop’s robustness and potential to outperform traditional dropout in limited-data scenarios.

### Dataset
The experiments were conducted on the **Peaks** and **MNIST** datasets, popular benchmarks in machine learning research. The code in this repository is designed to be adaptable to other classification datasets as well.

## Repository Structure
- `models`: Contains the nonlinear and CNN models.
- `mgopt_methods`: Contains the MGDrop and SMGDrop optimization algorithms. 
- `experiments/`: Scripts for running the experiments and generating the results found in the thesis. Each subfolder is a unique experiment.
- `datasets`: Contains methods for loading and preprocessing the datasets used in the experiments.


## Requirements
The code is implemented in Python using [PyTorch](https://pytorch.org/) (version 1.10.2). 

## Usage
1. **Setup**: Clone the repository and install dependencies:
   ```bash
   git clone git@github.com:gjsaave/multigrid_nn.git
   cd multigrid_nn
   
