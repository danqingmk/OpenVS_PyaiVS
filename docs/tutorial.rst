Tutorial
========

**Welcome to the PyaiVS tutorial**. In this tutorial, we will demonstrate a complete virtual screening workflow for finding inhibitors of the ABCG2 transporter using PyaiVS. The steps include **preparing a dataset of known actives/inactives, optimizing the model parameters, training and evaluating a predictive model, and then using that model to screen a library of new compounds**.

ABCG2 (also known as the breast cancer resistance protein) is an ATP-binding cassette transporter implicated in multidrug resistance. Identifying effective inhibitors of ABCG2 can help overcome chemotherapy resistance, which makes it an interesting case for virtual screening. PyaiVS provides an integrated platform to build machine learning models for such tasks, combining multiple algorithms, molecular representations, and data splitting strategies.

In a recent study, PyaiVS was used to screen over 4 million compounds for potential ABCG2 inhibitors, leading to the experimental discovery of several novel inhibitor molecules.This tutorial will walk through the same process on a smaller scale, using example data.

**Dataset Preparation**

The first step is to prepare your dataset of compounds with known activity against ABCG2. PyaiVS expects the data in a tabular format (commonly a CSV file) containing at least a column for molecular structures and a column for the target variable. Typically, the molecular structure is provided as a SMILES string, and the target variable can be a binary label (e.g., 1 for inhibitor, 0 for non-inhibitor).
For example, you might have a file ABCG2_inhibitors.csv with the following columns:

    SMILES – the chemical structure in SMILES notation (e.g., CCOc1ccc2nc(S(=O)(=O)N)c(Br)nc2c1).

    Active – the activity label, where 1 indicates the compound is an ABCG2 inhibitor and 0 indicates it is not active (or a much weaker activity).

Ensure that the data is clean (e.g., valid SMILES strings, no missing labels). If your data is in an SD file or another format, you may need to convert it to CSV or load it with RDKit/Pandas and then pass the appropriate objects to PyaiVS. PyaiVS can likely accept a pandas DataFrame or a file path to the dataset. In this tutorial, we'll assume the dataset file path is used directly.

Once your dataset is prepared and accessible, you can proceed to use PyaiVS for model building.

**Parameter Optimization**

Before committing to a specific model, PyaiVS can help you explore which machine learning algorithm and which molecular representation work best for your dataset. This is done via the *model_bulid.running(..., run_type='param')* mode, which performs a parameter optimization (or model selection) routine.

When you run PyaiVS in this mode, it will internally try different combinations of algorithms and molecular representations (and possibly hyperparameters) to identify the best-performing model for your data.This may include testing traditional machine learning models (like Random Forest or SVM) using fingerprint descriptors, as well as deep learning models (like Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), or Attentive FP) using graph-based representations.PyaiVS also evaluates different data splitting strategies (e.g., random split vs. scaffold-based split vs. clustering-based split) to ensure the model performance is robust and not an artifact of a particular train/test split.

Let's run the parameter optimization step on our dataset:

.. code-block:: python

   from pyaiVS import model_bulid

   # Specify the path to the dataset (CSV file)
   data_path = "abcg2.csv"

   # Run parameter optimization to find the best algorithm and representation
   best_settings = model_bulid.running(data_path, run_type='param')

   
When you execute the above, PyaiVS will begin testing various model configurations. This process may take some time depending on the size of your dataset and the number of combinations (since it could train multiple models under the hood). As it runs, you should see output indicating which algorithm and representation are being evaluated and the performance (e.g., cross-validated AUC-ROC, accuracy, etc.) for each.

After completion, the *best_settings* object (its type could be a dictionary or a custom object, depending on implementation) will contain the details of the best-found model. For example, it might return something like:

   best_settings['algorithm'] – the name of the best algorithm (e.g., "GCN" or "RandomForest").

   best_settings['representation'] – the best molecular representation (e.g., "Graph" for a graph-based model, or "ECFP4" for an ECFP4 fingerprint).

   best_settings['split'] – the data splitting strategy that was used/best (e.g., "cluster").

   It may also include optimal hyperparameters or model objects, depending on the design of PyaiVS.

You can inspect this output or log to see what model PyaiVS recommends. In our hypothetical scenario, let's say the parameter search concludes that a Graph Convolutional Network using a graph representation of molecules (with a clustering-based split for training/testing) yielded the highest validation performance.

**Model Evaluation (Training the Final Model)**

Once the optimal model type and features are identified, the next step is to train the final model on the dataset and evaluate its performance. This is accomplished with *model_bulid.running(..., run_type='result')*. In this mode, PyaiVS will typically train the chosen model on the training portion of your data and then evaluate it on a test set (or via cross-validation), providing performance metrics as output.

Using the example best settings from above (GCN algorithm with graph representation), we would run:

.. code-block:: python

   from pyaiVS import model_bulid
   # Using the best model settings obtained from the previous step:
   final_model = model_bulid.running(data_path, run_type='result',
                                  algorithm=best_settings['algorithm'],
                                  representation=best_settings['representation'],
                                  split=best_settings.get('split', 'cluster'))


In this code, we pass the dataset again along with the specific algorithm, representation, and split strategy that were determined to be optimal. PyaiVS will then train the model (e.g., train a GCN on the entire training set) and evaluate it. The evaluation may be done on a hold-out test set if a split strategy was used (for example, if split='cluster', PyaiVS might have internally split the data into a training set and a test set based on cluster groups; it will now report performance on that held-out test set).

You should see output such as final accuracy, ROC-AUC, precision/recall, or other relevant metrics for the model. These metrics give you an idea of how well the model is able to distinguish ABCG2 inhibitors from non-inhibitors. For instance, you might get a message like: "Best model: GCN (Graph representation) achieved AUC-ROC = 0.85 on the test set." (The actual performance will depend on your data.)

At this point, final_model may be an object representing the trained model (for example, a scikit-learn model or a PyTorch model wrapped in a PyaiVS interface). The PyaiVS pipeline might also save the trained model to disk (e.g., as a file in a results directory, or a Pickle file) so that you can reload it later for screening. Check the documentation or console output for any indication of where the model is saved. Commonly, a file like best_model.pkl or a timestamped output directory might be created to store the model and results.

Now we have a trained model that appears to perform well in distinguishing likely ABCG2 inhibitors. The next step is to use this model for virtual screening.

**Virtual Screening with the Trained Model**

Virtual screening involves taking a large collection of candidate compounds (for example, a chemical library or database) and using our model to predict which of those compounds are likely to be active (in this case, ABCG2 inhibitors). PyaiVS provides a function virtual_screen.model_screen(...) for this purpose.

Before running the screening, prepare your library of candidate compounds in a format that PyaiVS can process. This might be a SMILES file (each line is a SMILES and perhaps an identifier) or a CSV with a SMILES column, or another format that the virtual_screen module supports. For our example, let's assume we have a file base.csv that contains thousands of SMILES of compounds to screen.

Using the trained model (from the previous step) and the library file, we can execute the virtual screening as follows:

.. code-block:: python

   from pyaiVS import virtual_screen

   # Use the trained model to screen a library of compounds
   screening_results = virtual_screen.model_screen(final_model, 
                                               "base.csv", 
                                               output_file="screening_results.csv", 
                                               top_k=50)

In this code:

    ・The first argument final_model is the model we trained (we are passing the in-memory model object). PyaiVS will also accept a path to a saved model file here if you have the model saved instead of in memory (for example, you could provide something like "best_model.pkl" if such a file was produced).

    ・"base.csv" is the path to the file containing the virtual library of compounds to be screened. PyaiVS will read this file and compute the necessary molecular features for each compound (e.g., fingerprints or graphs, matching the representation the model expects).

    ・output_file="screening_results.csv" tells PyaiVS to write the screening outcomes to a CSV file. Typically, this CSV might contain each compound (by an ID or SMILES) along with the predicted score or probability of being an active inhibitor.

    ・top_k=50 is an optional parameter (in this example) specifying that we are interested in the top 50 predicted hits. If supported, PyaiVS will identify the 50 compounds with the highest predicted probability of being ABCG2 inhibitors and could, for instance, write them to a separate file or highlight them in the output. (If top_k is not specified, PyaiVS will simply output scores for all compounds; you can then sort the results to find the top candidates manually.)

After running model_screen, the variable screening_results may contain the raw predictions (for example, a list of predicted values or a data structure). More importantly, the file screening_results.csv will be created. You can open this file to examine the results of the virtual screening. It might look like:
   SMILES,Predicted_Score
   CCOc1ccc2nc(S(=O)(=O)N)c... , 0.95
   O=c1cc(-c2ccccn2)onc1OCC... , 0.90
   ... (other compounds and scores)

Where "Predicted_Score" could be a probability (between 0 and 1) of being an inhibitor, or some score where higher means more likely active. The compounds would be sorted by score if top_k was used and the output was filtered.

You can then take the top candidates (in our example, 50 compounds) for further analysis, such as more detailed in silico modeling (docking, pharmacophore analysis) or even experimental testing.

Full Workflow Example

Below is a full example script (as might be found in example.py) that combines all the steps above into one coherent workflow. This example assumes that you have prepared ABCG2_inhibitors.csv as described and have a file virtual_library.smi with compounds to screen.

.. code-block:: python

   from pyaiVS import model_bulid, virtual_screen

   # Step 1: define the dataset path
   data_file = "ABCG2_inhibitors.csv"

   # Step 2: run parameter optimization to select best model and representation
   best_config = model_bulid.running(data_file, run_type='param')
   print("Best configuration found:", best_config)

   # Step 3: train the final model using the best configuration
   final_model = model_bulid.running(data_file, run_type='result',
                                  algorithm=best_config['algorithm'],
                                  representation=best_config['representation'],
                                  split=best_config.get('split', 'cluster'))
   # (The final_model now holds the trained model. Performance metrics are shown in the console output.)

   # Step 4: perform virtual screening on a new library of compounds
   library_file = "virtual_library.smi"  # input file with SMILES of compounds to screen
   virtual_screen.model_screen(final_model, library_file, 
                            output_file="predicted_hits.csv", top_k=100)
   # The results of the screening are saved to "predicted_hits.csv". The top 100 predicted compounds are written (along with their scores).

In this script, we go from data to predictions in four steps. After running it, you would inspect predicted_hits.csv to review the compounds that the model predicted as likely ABCG2 inhibitors. Those compounds could be candidates for follow-up in a lab experiment or further computational analysis.
