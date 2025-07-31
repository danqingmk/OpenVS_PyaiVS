Installation
============

This document outlines the essential requirements for successfully installing and running PyaiVS. PyaiVS integrates multiple machine learning models, molecular descriptors, and data splitting methods tailored for drug discovery applications.

System Requirements
-------------------

To efficiently run PyaiVS, your system should meet the following specifications:

- **CPU**: Multi-core processor recommended for parallel processing tasks  
- **RAM**: Minimum 8GB, 16GB or more recommended for handling large datasets  
- **GPU**: CUDA-compatible GPU (optional, but recommended for deep learning models)  
- **CUDA**: Version 10.2 or higher (required for GPU acceleration)  
- **Disk Space**: At least 5GB for software and its dependencies

Python Environment Setup
------------------------

PyaiVS requires **Python 3.8**. It is strongly recommended to use **Conda** to manage your environment::

    # Create a new conda environment
    conda create -n pyaivs_env python=3.8

    # Activate the environment
    conda activate pyaivs_env

Core Dependency Installation
----------------------------

The following core dependencies must be installed in the specified order:

1. RDKit
^^^^^^^^

RDKit is essential for molecular structure processing and generating molecular descriptors::

    conda install rdkit -c conda-forge

2. PyTorch
^^^^^^^^^^

PyaiVS uses **PyTorch 1.9.0** for deep learning models::

    conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

Ensure your CUDA version is compatible (10.2 or higher). For CPU-only installations::

    conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch

3. Deep Graph Library (DGL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

DGL is required for graph-based models::

    conda install -c dglteam dgl==0.4.3post2

4. Additional Required Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install these extra packages required by different PyaiVS components::

    conda install xgboost hyperopt pandas scikit-learn numpy
    pip install mxnet requests

These dependencies support various machine learning algorithms used in the package:

+------------------------+-----------------------------------------+
| Model Type             | Required Packages                       |
+========================+=========================================+
| Machine Learning       | scikit-learn, xgboost                   |
+------------------------+-----------------------------------------+
| Deep Learning          | pytorch, dgl                            |
+------------------------+-----------------------------------------+
| Hyperparameter Opt.    | hyperopt                                |
+------------------------+-----------------------------------------+
| Data Processing        | pandas, numpy                           |
+------------------------+-----------------------------------------+

Installing the PyaiVS Package
-----------------------------

Once all dependencies are set up, install the PyaiVS package::

    pip install PyaiVS

Installation Verification
-------------------------

To verify that PyaiVS is installed correctly, run the following simple test::

    # Import main modules
    from script import model_bulid, virtual_screen

    # This should execute without errors
    print("PyaiVS is installed correctly!")

Troubleshooting
---------------

CUDA Compatibility Issues
^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter CUDA-related errors:

- Use ``nvidia-smi`` to verify your CUDA version  
- Ensure the correct CUDA version of PyTorch is installed  
- Set the appropriate environment variables::

    import os
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

Memory Issues
^^^^^^^^^^^^^

If you run into memory errors when handling large datasets or complex models:

- Reduce batch size in model configuration  
- Use CPU mode if GPU memory is limited  
- Process datasets in chunks whenever possible

Example of specifying CPU device::

    model_bulid.running('./dataset/abcg2.csv', run_type='result', cpus=4)

Package Dependency Conflicts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you face dependency conflicts:

- Create a new Conda environment  
- Install dependencies in the exact order listed above  
- Avoid mixing conda and pip installs for the same package

Next Steps
----------

After installation, refer to the **Getting Started Guide** for your first virtual screening task using PyaiVS.
