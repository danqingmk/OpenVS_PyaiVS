Developer Guide
===============

This guide is intended for developers and advanced users who want to understand the internals of PyaiVS or extend its functionality. It provides an overview of the code structure and instructions on how to add new algorithms, descriptors, or split strategies to the framework. 
PyaiVS was designed to unify various components of AI-driven virtual screening into a single package. As such, it integrates multiple machine learning models, feature representations, and data handling techniques under the hood. Understanding how these pieces fit together will help you modify or extend the software.

**Overview of Code Structure**

The PyaiVS codebase is organized into modules that correspond to different parts of the virtual screening workflow. Below is an overview of key components:

    **model_bulid module** – This module handles model training and evaluation workflows. It contains the logic for reading in the dataset, generating molecular features (descriptors or graphs), splitting the data into training/validation/test sets, training models, performing hyperparameter or model selection (the run_type='param' mode), and outputting results. In essence, model_bulid orchestrates the end-to-end process of going from input data to a trained model and performance metrics.

    **virtual_screen module** – This module is responsible for applying a trained model to new data (external compound libraries). It provides functions such as model_screen which take a trained model (or a file containing the model) and a set of candidate molecules, then computes predictions for those candidates. The module likely includes routines to load the model, featurize the new molecules in the same way as the training data, and output the predicted scores or classifications to the user.

    **Machine Learning algorithms** – PyaiVS integrates a total of nine machine learning algorithms, spanning both conventional methods and deep learning models.Traditional algorithms (e.g., Random Forest, Support Vector Machine, logistic regression) are probably implemented via scikit-learn or similar libraries. Deep learning models such as GCN, GAT, and Attentive FP are implemented using PyTorch (with DGL for graph operations).These algorithms are used through a unified interface in model_bulid. For example, the code may have a mapping from algorithm names to the actual model implementation. During training, PyaiVS will invoke the appropriate library or custom code for the chosen algorithm.

    **Molecular representations** – The package supports five types of molecular representations for input features. This includes fingerprint-based descriptors like ECFP4 (circular fingerprints) and MACCS keys, which are computed using RDKit for each molecule. It also includes graph-based representations where molecules are converted into graph objects (nodes for atoms, edges for bonds) for use with graph neural networks (e.g., GCN, GAT, Attentive FP). There may also be other representations such as physicochemical descriptor vectors or other fingerprint types. The code responsible for featurization will choose the appropriate method based on the representation parameter (for instance, calling an RDKit function to get a fingerprint bit vector, or a DGL utility to create a graph from an RDKit Mol).

    **Data splitting strategies** – PyaiVS includes multiple data splitting strategies for model validation. Likely options are: 

       **random split** (shuffling the dataset into train/test)

       **scaffold split** (separating molecules by their core scaffolds so that the test set contains scaffolds not seen in training)

       **cluster-based split** (clustering molecules by similarity and then splitting clusters between train/test to ensure diversity). 

    **Utilities and helpers** – In addition to the main modules above, there are various utility functions and possibly sub-modules. For example, there might be:

    · A module or section for metric calculations (computing AUC, confusion matrix, etc.).

    · Functions to save and load models (e.g., using Pickle or torch serialization).

    · Helper functions to parse input files (reading CSV or SMILES files, cleaning data).

    · Definitions of hyperparameter grids or default training parameters for each algorithm.
    These utilities support the core functionality and make the code more modular and maintainable.

With this overview, you can begin to locate the areas of the code relevant to the changes you want to make. Below, we discuss specific extension points in PyaiVS.

**Extending PyaiVS**

PyaiVS can be extended with new machine learning models, molecular descriptors, and data splitting strategies. Adding any of these requires implementing the component and integrating it so the framework recognizes it. The process is similar for each type:

   · New Machine Learning Model (e.g., LightGBM): Implement the model in the codebase (for example, add a class or function in the models module). Follow the pattern of existing models (ensure it provides required training and prediction methods). Then register the model so PyaiVS can use it (for instance, add it to the model registry or configuration that lists available models). For LightGBM, wrap an LGBMClassifier from the LightGBM library (requiring that the library is installed) in a new model class. Finally, verify the model accepts the chosen descriptors and produces predictions as expected.

   · New Molecular Descriptor Method: Create the descriptor calculation in the descriptors module (e.g., a function that takes a molecule and returns a feature vector). Integrate this new method by adding it to the descriptor registry or pipeline configuration so PyaiVS can recognize it by name. Ensure the descriptor’s output format (such as a NumPy array of features) matches what the models expect, and handle any required parameters or initialization.

   · New Data Splitting Strategy: Implement the strategy as a new class or function, mirroring the interface of existing splitting strategies (which output train/test splits as indices or subsets). Register this strategy in the splitting configuration so it can be selected (for example, by adding a case in the splitting selection logic or a new entry in a strategy lookup table). Make sure the new splitter yields the training and test sets (and validation if applicable) in the correct format to be consumed by PyaiVS.

.. note:: To be updated
