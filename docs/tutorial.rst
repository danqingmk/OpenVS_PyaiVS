Tutorial
========

**Welcome to the PyaiVS tutorial**. In this tutorial, we will demonstrate a complete virtual screening workflow for finding inhibitors of the ABCG2 transporter using PyaiVS. The steps include **preparing a dataset of known actives/inactives, optimizing the model parameters, training and evaluating a predictive model, and then using that model to screen a library of new compounds**. Additionally, we will also show how to apply PyaiVS for **regression tasks**, using the **BACE dataset** to predict pIC50 values of compounds.

The Virtual Screening of ABCG2 Inhibitors.
------------------------------------------

ABCG2 (also known as the breast cancer resistance protein) is an ATP-binding cassette transporter implicated in multidrug resistance. Identifying effective inhibitors of ABCG2 can help overcome chemotherapy resistance, which makes it an interesting case for virtual screening.

Before diving into the code, let’s understand how to use this tool for virtual screening.  
PyaiVS follows a two-step workflow:

1. **Model Building**: Train and optimize classification models on your dataset  
2. **Virtual Screening**: Use the best model to screen a compound library

Prerequisites
^^^^^^^^^^^^^

Before you begin, make sure you have:

- Installed OpenVS_PyaiVS and all dependencies  
- A training dataset with known active/inactive molecules (CSV format with SMILES)  
- A compound library to screen (CSV format with SMILES)

Step 1: Prepare Your Data
^^^^^^^^^^^^^^^^^^^^^^^^^

For model building, you need a dataset where molecules are labeled as active or inactive.  
The dataset should be in CSV format and include at least:

- **SMILES column** (molecular structure)  
- **Target column** (1 = active, 0 = inactive)

Example training dataset:

+-----------------------------+--------+
| SMILES                      | lable  |
+=============================+========+
| CC1=CC=CC=C1C(=O)NC2=CC=... | 1      |
+-----------------------------+--------+
| CC1=CC=C(C=C1)NC(=O)C2=...  | 0      |
+-----------------------------+--------+

For the screening compound library, you need a CSV file with at least a SMILES column:

.. code-block:: text

    smiles
    CSC1=C(c2ccc(C)s2)/C(=N/C(C)(C)C)C1
    CSC1=C(c2ccccc2)/C(=N/C(C)(C)C)C1
    C/N=C1\CC(SC)=C1c1ccc(OC)cc1
    ...

Step 2: Build and Optimize Your Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start by building a model using your training dataset.  
OpenVS_PyaiVS allows training multiple models with different configurations and selects the best one automatically.

.. code-block:: python

    from script import model_bulid

    # Train models with parameter optimization
    model_bulid.running('./dataset/abcg2.csv',
                        out_dir='./dataset/abcg2/this_work',
                        run_type='param',
                        cpus=128)

    # Evaluate models and identify the best one
    model_bulid.running('./dataset/abcg2.csv',
                        out_dir='./dataset/abcg2/this_work',
                        run_type='result',
                        cpus=128)

What this does:

- First call optimizes parameters for all specified models  
- Second call evaluates the models and selects the best one

**Key arguments of `running()` function**:

- ``file_name``: Path to training dataset  
- ``out_dir``: Output directory  
- ``run_type``: `'param'` for parameter optimization, `'result'` for model evaluation  
- ``cpus``: Number of CPU cores to use  
- ``split``: Data splitting method (`random`, `scaffold`, `cluster`, or `all`)  
- ``model``: List of models to train (`SVM`, `KNN`, `DNN`, `RF`, `XGB`, `gcn`, `gat`, `attentivefp`, `mpnn`, or `all`)  
- ``FP``: Molecular fingerprint type (`MACCS`, `ECFP4`, `pubchem`, or `all`)

Step 3: Run Virtual Screening
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After building and optimizing the model, use the best model to perform virtual screening:

.. code-block:: python

    from script import virtual_screen

    virtual_screen.model_screen(model='GCN',
                                split='cluster',
                                FP='None',
                                model_dir='./dataset/abcg2/this_work/model_save',
                                screen_file='./database/compounds_to_screen.csv',
                                sep=';',
                                smiles_col='smiles')

**Key arguments of `model_screen()` function**:

- ``model``: Model type for screening (e.g., `'SVM'`, `'KNN'`, `'DNN'`)  
- ``split``: Data splitting method used in training (`random`, `scaffold`, etc.)  
- ``FP``: Fingerprint type (e.g., `'MACCS'`, `'ECFP4'`)  
- ``model_dir``: Directory containing the trained model  
- ``screen_file``: Path to the compound library CSV  
- ``prop``: Probability threshold for activity (default: 0.5)  
- ``sep``: CSV delimiter character  
- ``smiles_col``: Name of the SMILES column in the library

The function will:

- Identify the best model based on your specifications  
- Convert molecules into proper features  
- Predict activity for each compound  
- Apply Lipinski’s Rule of Five filtering  
- Save all compounds that pass into a new CSV file

Step 4: Check the Results
^^^^^^^^^^^^^^^^^^^^^^^^^

After screening, results can be found in a folder named ``screen`` (created at the same level as ``model_save``).  
The output file will be named after your input file and suffixed with the probability threshold:

.. code-block:: text

    dataset/abcg2/this_work/screen/gcn_cluster_gcn_screen_0.8.csv

This file includes SMILES strings for compounds that:

- Are predicted to be active by the model (above threshold)  
- Pass Lipinski’s Rule of Five (i.e., drug-like properties)

Complete End-to-End Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a complete script that performs both model building and virtual screening:

.. code-block:: python

    from script import model_bulid, virtual_screen

    # Step 1: Build and optimize models
    model_bulid.running('./dataset/abcg2.csv',
                        out_dir='./dataset/abcg2/this_work',
                        run_type='param',
                        cpus=128)

    # Step 2: Evaluate models and find the best one
    model_bulid.running('./dataset/abcg2.csv',
                        out_dir='./dataset/abcg2/this_work',
                        run_type='result',
                        cpus=128)

    # Step 3: Use the best model for virtual screening
    virtual_screen.model_screen(model='GCN',
                                split='cluster',
                                FP='None',
                                model_dir='./dataset/abcg2/this_work/model_save',
                                screen_file='./database/compounds_to_screen.csv',
                                sep=';',
                                smiles_col='smiles')

This workflow will generate the following output:

- **Optimized model parameters**: ``./dataset/abcg2/this_work/param_save/``  
- **Model performance results**: ``./dataset/abcg2/this_work/result_save/``  
- **Saved trained models**: ``./dataset/abcg2/this_work/model_save/``  
- **Virtual screening results**: ``./dataset/abcg2/this_work/screen/``

Congratulations! You should now have successfully completed your first virtual screening task using PyaiVS.

Using PyaiVS for Regression Tasks
---------------------------------

The **PyaiVS** framework supports not only classification tasks but also regression tasks. In this documentation, we will demonstrate how to use **PyaiVS** for a regression task to predict the **pIC50** value on the **BACE dataset**. The pIC50 value is a measurement of the binding affinity of a compound to a target, commonly used in drug discovery to predict biological activity. The BACE dataset is a benchmark dataset provided by MoleculeNet, which focuses on predicting the activity of BACE-1 inhibitors. The target value for each compound in this dataset is its pIC50 value. The goal of this task is to predict the pIC50 value given the molecular structure (SMILES representation).

Preparing the Data
^^^^^^^^^^^^^^^^^^

The BACE dataset contains the SMILES representation of compounds and their corresponding pIC50 values. The data format is as follows:

.. code-block:: text

mol,pIC50
CC(C)(C)C(=O)N1CCN(CC1)C2CCCCC2,7.25
CC(C)(C)C(=O)N1CCN(CC1)C2CCCCC2,6.93
    ...

Here, the `mol` column represents the SMILES of the compound, and `pIC50` is the target variable, representing the compound’s binding affinity to BACE-1.

Training the Regression Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyaiVS** provides a simple interface for training regression models. Below is an example of how to train a model to predict pIC50 values:

```python
from script import model_bulid

# Step 1: Train the model and optimize hyperparameters
best_settings = model_bulid.running(
    './dataset/example_reg.csv',    # Path to the input data
    out_dir='./out_reg',            # Output directory to save trained models
    split=["random"],               # Data splitting method (here we use random)
    model=["SVM"],                  # Models to use (supports SVM, XGBoost, Random Forest, etc.)
    FP=["MACCS"],                   # Molecular fingerprints (here we use MACCS)
    key="rmse",                     # Evaluation metric (here we use RMSE)
    task_type="reg",                # Specify as a regression task
    cpus=4                          # Number of CPUs to use
)
```

In this code, we specify that we want to train an **SVM** model using **MACCS** molecular fingerprints and evaluate using **RMSE** (Root Mean Squared Error). The data is split randomly using the `random` method, but other options like `scaffold` or `cluster` can also be used.

Activity Prediction
^^^^^^^^^^^^^^^^^^^

Once the model is trained and evaluated, you can use the trained model to screen a set of compounds and predict their pIC50 values. Below is an example of how to screen compounds:

```python
from script import virtual_screen

# Step 2: Screen compounds using the best model
virtual_screen.model_screen(
    best_settings=best_settings,                     # Pass the trained model configuration
    screen_file="./database/compounds_to_screen.csv",   # Path to the compound library to be screened
    smiles_col="smiles",                             # SMILES column name
    output_file="./result/result_svm_maccs_random_rmse_reg.csv",  # Output file name
    task_type="reg"                                  # Specify as a regression task
)
```

Output Results
^^^^^^^^^^^^^^

After activity prediction is completed, **PyaiVS** will generate a CSV file containing the predicted pIC50 values for each compound. The output file will contain the SMILES string of each compound along with its corresponding predicted pIC50 value. For example:

```
smiles,predict_value
CC(C)(C)C(=O)N1CCN(CC1)C2CCCCC2,7.22
CC(C)(C)C(=O)N1CCN(CC1)C2CCCCC2,6.95
...
```
