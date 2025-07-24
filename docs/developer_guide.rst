Developer Guide
===============

This guide is intended for developers and advanced users who want to understand the internals of PyaiVS or extend its functionality. It provides an overview of the code structure and instructions on how to add new algorithms, descriptors, or split strategies to the framework. It also outlines guidelines for contributing to the project.
PyaiVS was designed to unify various components of AI-driven virtual screening into a single package. As such, it integrates multiple machine learning models, feature representations, and data handling techniques under the hood. Understanding how these pieces fit together will help you modify or extend the software.

Overview of Code Structure

The PyaiVS codebase is organized into modules that correspond to different parts of the virtual screening workflow. Below is an overview of key components:

    model_bulid module – This module handles model training and evaluation workflows. It contains the logic for reading in the dataset, generating molecular features (descriptors or graphs), splitting the data into training/validation/test sets, training models, performing hyperparameter or model selection (the run_type='param' mode), and outputting results. In essence, model_bulid orchestrates the end-to-end process of going from input data to a trained model and performance metrics.
    virtual_screen module – This module is responsible for applying a trained model to new data (external compound libraries). It provides functions such as model_screen which take a trained model (or a file containing the model) and a set of candidate molecules, then computes predictions for those candidates. The module likely includes routines to load the model, featurize the new molecules in the same way as the training data, and output the predicted scores or classifications to the user.
    Machine Learning algorithms – PyaiVS integrates a total of nine machine learning algorithms, spanning both conventional methods and deep learning models.Traditional algorithms (e.g., Random Forest, Support Vector Machine, logistic regression) are probably implemented via scikit-learn or similar libraries. Deep learning models such as GCN, GAT, and Attentive FP are implemented using PyTorch (with DGL for graph operations).These algorithms are used through a unified interface in model_bulid. For example, the code may have a mapping from algorithm names to the actual model implementation. During training, PyaiVS will invoke the appropriate library or custom code for the chosen algorithm.
    Molecular representations – The package supports five types of molecular representations for input features. This includes fingerprint-based descriptors like ECFP4 (circular fingerprints) and MACCS keys, which are computed using RDKit for each molecule. It also includes graph-based representations where molecules are converted into graph objects (nodes for atoms, edges for bonds) for use with graph neural networks (e.g., GCN, GAT, Attentive FP). There may also be other representations such as physicochemical descriptor vectors or other fingerprint types. The code responsible for featurization will choose the appropriate method based on the representation parameter (for instance, calling an RDKit function to get a fingerprint bit vector, or a DGL utility to create a graph from an RDKit Mol).
    Data splitting strategies – PyaiVS includes multiple data splitting strategies for model validation. Likely options are: random split (shuffling the dataset into train/test), scaffold split (separating molecules by their core scaffolds so that the test set contains scaffolds not seen in training), and cluster-based split (clustering molecules by similarity and then splitting clusters between train/test to ensure diversity). The splitting logic might reside in the model_bulid module or a dedicated utility. It's used in both the parameter optimization phase (to evaluate models under different splits) and in the final training (to report an unbiased performance metric).
    Utilities and helpers – In addition to the main modules above, there are various utility functions and possibly sub-modules. For example, there might be:

    A module or section for metric calculations (computing AUC, confusion matrix, etc.).

    Functions to save and load models (e.g., using Pickle or torch serialization).

    Helper functions to parse input files (reading CSV or SMILES files, cleaning data).

    Definitions of hyperparameter grids or default training parameters for each algorithm.
    These utilities support the core functionality and make the code more modular and maintainable.

Understanding the layout: for instance, if you open the project folder, you might see a structure like:
   pyaiVS/
      __init__.py
      model_bulid.py
      virtual_screen.py
      algorithms/        # (possibly, implementations or wrappers for each ML algorithm)
      features/          # (possibly, code for computing descriptors)
      data_utils.py      # (functions for splitting data, reading files)
      ...
(This is a hypothetical layout to illustrate where things might reside.)

With this overview, you can begin to locate the areas of the code relevant to the changes you want to make. Below, we discuss specific extension points in PyaiVS.

Adding New Machine Learning Models

One way to extend PyaiVS is to add support for a new machine learning algorithm beyond the ones already integrated. For example, you might want to add XGBoost as a new algorithm, or incorporate a new deep learning architecture.

Here are guidelines for adding a new model:

    Implement or import the model: If the new algorithm is available via an external library (e.g., XGBoost or LightGBM), you can use that library. Ensure it’s installed in your environment and import the necessary classes in the PyaiVS code. If it’s a custom model, implement the model’s class or training functions. For deep learning models, this could mean writing a new PyTorch model class.

    Register the model in PyaiVS: Locate where PyaiVS selects and instantiates models based on a name or identifier. In the model_bulid module, this might be a series of if/elif statements or a dictionary mapping algorithm names to model constructors. Add an entry for your model. For example, if adding XGBoost, you might add logic such that when algorithm="XGBoost", PyaiVS will initialize an XGBClassifier with certain default parameters.

    Integrate training and prediction: Ensure that the new model can be trained and used for prediction within the PyaiVS pipeline. If the model follows a scikit-learn interface (with .fit() and .predict() methods), integration can be straightforward. If it’s a PyTorch model, you might need to write a training loop unless one is already provided. You may reuse patterns from existing deep models in the code (e.g., how the GCN model is trained).

    Handle model-specific features: If the model requires a specific input format or has unique requirements, address those. For instance, if your model only works with fingerprint vectors, you should indicate that only certain representations are compatible. Conversely, if it’s a graph-based model, ensure it receives a graph object. PyaiVS might have a section where it checks the representation and algorithm combination; you may need to update this logic to include your new model.

    Hyperparameters and optimization: Optionally, decide how hyperparameters for the new model will be handled. You can provide default values (e.g., number of trees for a random forest, or network architecture details for a neural net). If you want the run_type='param' mode to consider this model, add a range of hyperparameters or configurations for it in the parameter search routine. This could be as simple as adding your model to the list of algorithms that run_type='param' iterates over, or as involved as adding a grid of hyperparameter values in the code.

    Testing the new model: After adding the model, test it on a small dataset. Run model_bulid.running(..., run_type='param') including your new model to see that it trains and produces results. Then try run_type='result' to ensure it can train fully and the model can be saved and loaded. Verify that virtual_screen.model_screen works with the model (i.e., it can take the trained model and make predictions on new data). This process will confirm that your integration is successful.

By following these steps, you can incrementally build support for new algorithms into PyaiVS. The modular design of the package (with a unified training pipeline) should facilitate adding new models as long as you hook into the existing interfaces properly.

Adding New Molecular Descriptors

Another extension point is introducing new molecular descriptors or representation methods. PyaiVS comes with a set of built-in representations (fingerprints, graphs, etc.), but you may want to use a different descriptor (for example, a custom fingerprint, a descriptor set like RDKit’s topological features, or embeddings from a pretrained model).

To add a new molecular representation:

    Implement the descriptor calculation: Write a function that takes a molecule and produces the descriptor. For instance, if adding a new fingerprint type, use RDKit (or another library) to calculate it. Ensure this function can be applied to all molecules in your dataset efficiently (perhaps vectorizing over a list of molecules if possible). If the descriptor is complex (e.g., requires an external model or a web service), ensure you handle those dependencies.

    Integrate with the feature pipeline: Find where PyaiVS generates features from molecules. This could be in model_bulid.running or a helper function that converts SMILES to features. Add your descriptor as a new option. For example, there might be a conditional like if representation == "ECFP4": compute Morgan fingerprint. You would add elif representation == "MyDesc": compute your descriptor. Make sure to also handle any normalization or data formatting your descriptor might need (e.g., scaling continuous descriptors, handling array shapes, etc.).

    Specify compatibility with models: Consider which algorithms can work with your new descriptor. Most descriptors that yield a fixed-length numerical vector can be used with any traditional ML or fully-connected network. If your descriptor is an image or a sequence, you’d need a model that can handle that (which is beyond typical usage). In general, as long as your descriptor results in a numeric feature vector per molecule, you can plug it into the existing models (scikit-learn models can handle it as part of their X input, and PyTorch models can handle it if they have been designed for vector inputs or you adapt the network).

    Update representation lists (if any): PyaiVS might maintain a list or enumeration of valid representation strings. Add your new representation name so that the program recognizes it and perhaps so that it’s included in any documentation or error messages. If run_type='param' should consider this representation, include it in the search. For example, if previously the code tried representations ["ECFP4", "MACCS", "Graph"], you might expand it to ["ECFP4", "MACCS", "Graph", "MyDesc"].

    Test the new descriptor: Run a quick experiment to ensure that when you specify your new representation, the pipeline executes without errors. Check that the values being generated make sense (maybe print out a snippet of the feature vector for one molecule to verify it’s in the expected range or format). Then verify that models train on these features and yield results. This will confirm that your descriptor is correctly integrated.

By adding new descriptors, you expand the capability of PyaiVS to explore different feature spaces for virtual screening. This can be especially powerful if your new descriptor encodes information not captured by existing ones (for example, a pharmacophore-based bit vector, or a learned molecular embedding from another AI model).

Implementing Custom Split Strategies

Robust model evaluation often requires trying different ways of splitting data into training and testing sets. PyaiVS supports several out-of-the-box strategies,  but you might conceive of a new strategy (for example, time-based splits, or splitting by compound origin, etc.).

To add a custom data splitting strategy:

    Write the splitting function: Define a function (perhaps in the data utilities section of the code) that takes your dataset (and any relevant parameters) and returns indices or subsets for train/validation/test. For example, a time-based split might sort compounds by the date of discovery and take the earliest 80% as training and the latest 20% as test, simulating prospective validation. Ensure your function outputs in a format consistent with other split functions (commonly a tuple like (train_indices, test_indices) or (train_set, valid_set, test_set) depending on whether you use a separate validation set).

    Integrate with the pipeline: Identify where the splitting strategy is chosen in model_bulid.running. It might use a variable or parameter (e.g., split="random" or "cluster"). Add your new strategy here. For instance, if the user specifies split="time", call your time-based splitting function. If strategies are stored in a dictionary, add an entry mapping "time" to your function.

    Maintain reproducibility and options: If your split method involves randomness (like random shuffling or random cluster assignment), ensure you incorporate the random seed from PyaiVS (if provided) or otherwise allow reproducibility. You may also allow the user to pass specific arguments (though typically, split strategies are chosen by name only; any specific parameters could be hardcoded or inferred).

    Adapt any cross-validation or parameter search logic: In run_type='param' mode, if the code evaluates models under different splits, adding a new strategy means it could be included in that rotation. Decide if your new split should be part of that automatic exploration. If yes, insert it accordingly (with caution, as it will increase the search space).

    Test the new splitting method: Try running the pipeline with your split. For example, call model_bulid.running(data_path, run_type='result', split='time', algorithm='RandomForest', representation='ECFP4') to see that:

        The data is split as you expect (you might print the sizes of train/test to verify).

        Model training and evaluation proceed without errors using that split.

        The results make sense (e.g., if using time-based split, likely the model might have slightly lower performance if the distribution shifted over time – just an example of what to expect).

Adding custom split strategies allows you to tailor model validation to scenarios that the default strategies don’t cover. This can be important in drug discovery, where splits by scaffold or other criteria simulate how models perform on truly novel chemistry.

Extensions and Future Work

.. note:: To be updated
