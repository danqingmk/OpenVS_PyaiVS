# PyaiVS unifies AI workflows to accelerate ligand discovery

## Overview

**PyaiVS** is a comprehensive and modular machine learning framework tailored for molecular virtual screening (VS) and classification/regression model development. With a single line of code, users can:

- Rapidly build classification or regression models for various datasets  
- Automatically identify and recommend optimal algorithms  
- Access a suite of integrated machine learning and deep learning models  
- Utilize commonly used molecular descriptors and fingerprints  
- Choose from multiple dataset splitting strategies  
- Efficiently screen large-scale compound libraries  

## Installation

### Installation via Conda

```bash
# Create a Python 3.8 environment
conda create -n pyaivs python=3.8

# Activate the environment
conda activate pyaivs

# Install dependencies
conda install rdkit
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch  # Ensure CUDA >= 10.2
conda install -c dglteam dgl==0.4.3post2
conda install xgboost hyperopt mxnet requests pandas==1.2.0
pip install PyaiVS
``` 

## Basic Workflow

The workflow of PyaiVS consists of two main steps：

1. Train models and optimize parameters
2. Screen compounds using the best-performing model

Below is a code example:

```bash
from script import model_bulid, virtual_screen

# Step 1: Train models and find optimal parameters
model_bulid.running('your_dataset.csv',      # Input dataset
                    out_dir='./dataset',     # Output directory
                    run_type='param',        # Parameter optimization mode
                    cpus=4)                  # Number of CPUs to use

# Step 2: Generate results and get model recommendations
model_bulid.running('your_dataset.csv', 
                    out_dir='./dataset',
                    run_type='result',       # Result computation mode
                    cpus=4)

# Step 3: Screen compounds using the best model
virtual_screen.model_screen(model='SVM',      # Best-performing algorithm (e.g., SVM, selected based on evaluation metrics)
                            split='random',   # Best data splitting method (e.g., 'random', selected based on metrics)
                            FP='MACCS',       # Best fingerprint type (e.g., MACCS, selected based on metrics)
                            model_dir='./dataset/model_save',  # Path to the saved model
                            screen_file='./database/compound_library.csv',  # Compound library to be screened
                            sep=';',          # File delimiter
                            smiles_col='smiles')  # Column name containing SMILES strings
