Installation
============
**Installation via pip**

PyaiVS is available on the Python Package Index (PyPI) and can be installed using pip. We recommend using Python 3.8+ and installing PyaiVS in a clean virtual environment (such as one created with venv or Conda) to avoid dependency conflicts. To install PyaiVS and its core dependencies with pip, run:

.. code-block:: bash

    pip install PyaiVS rdkit torch dgl


This single command will install the PyaiVS library along with RDKit (for cheminformatics), PyTorch (for deep learning models), and the Deep Graph Library (DGL) for graph neural networks, if those packages are not already present.

**Installation via conda**

Installing PyaiVS with Anaconda/Miniconda is recommended for ease of managing complex dependencies like RDKit and GPU libraries. The following steps demonstrate a conda-based installation:

\#. Create a conda environment (optional): Create a new environment for PyaiVS to keep its packages isolated.

.. code-block:: bash

    conda create -n pyaivs_env python=3.8 -y
    conda activate pyaivs_env

The above creates and activates an environment named "pyaivs_env" with Python 3.9. (You may choose a different Python version as needed.)

\#. Install RDKit and PyTorch: Use conda to install RDKit and PyTorch. RDKit is available via the conda-forge channel, and PyTorch can be installed from PyTorch’s channel (or conda-forge for CPU-only version).

.. code-block:: bash

   conda install -c conda-forge rdkit
   conda install -c pytorch pytorch torchvision torchaudio cpuonly

The first line installs RDKit (and its dependencies) from conda-forge. The second line installs PyTorch and related packages from the official PyTorch channel (the cpuonly tag ensures the CPU version is installed; omit it or specify a cudatoolkit version if you want GPU support). You can adjust the PyTorch command to a specific CUDA toolkit version (e.g., adding cudatoolkit=11.8) for GPU acceleration.

Optional: If you plan to use graph-based models, you should also install DGL. You can install the CPU version of DGL via conda-forge (if available) or via pip. For example:

.. code-block:: bash

   pip install dgl

(For GPU support in DGL, use the appropriate dgl-cuda package via pip or conda, matching your CUDA version.)
#. Install PyaiVS: With the environment prepared and core dependencies installed, you can now install PyaiVS itself.

.. code-block:: bash
   pip install PyaiVS

This will download and install the PyaiVS package from PyPI into your conda environment. Since the heavy dependencies (RDKit, PyTorch, etc.) are already installed, this step should be quick. If in the future PyaiVS is made available on conda-forge, you could alternatively use conda install -c conda-forge pyaiVS to install it entirely via conda.
