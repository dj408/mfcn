# README

This code reproduces the experiments from [Manifold Filter Combine Networks (Johnson, Chew, Viswanath, et al.)](https://arxiv.org/abs/2307.04056).

The `convergence` folder contains standalone code to reproduce the numerical convergence experiments, first presented in "Convergence of Manifold Filter-Combine Networks" (Johnson et al.) at the NeurIPS 2024 Symmetry and Geometry in Neural Representations workshop. Consult the README in that folder for further details regarding the convergence experiments.

## Dependencies

1. This project requires Python>=3.11.
2. It also requires PyTorch. To install the correct version of PyTorch for your system, see [PyTorch's "Getting Started" guide](https://pytorch.org/get-started/locally/).
3. To use the Jupyter notebooks in this repo, first [install Jupyter](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
4. Finally, this project also makes use of several other common python packages, which can be installed with pip:
```
pip3 install \
numpy \
pandas \
scipy \
scikit-learn \
matplotlib \
matplotlib-inline \
accelerate \
torch_geometric \
torchmetrics
```

## Running experiments

Jupyter notebooks for the ellipsoids node regression and melanoma patient manifold classification experiments are in the `notebooks` folder of this repo. These notebooks provide step-by-step interactive workflows to reproduce these experiments, from processing the datasets, to running model cross-validations, to tabulating the results. 
> Note that you must have Python>=3.11 installed and available as an ipykernel to Jupyter to use these notebooks.

## Data availability

Datasets for the convergence and ellipsoids node regression experiments are easily generated with the provided scripts (and experiment notebooks).

The dataset for the melanoma patient manifold classification task is comprised of three CSV files, available in a [Mendeley Data repository](https://data.mendeley.com/datasets/79y7bht7tf/).

## Notes

- We recommend a project folder structure of:
```
mfcn
    |_code [clone this repo's files into here]
    |_data
          |_melanoma
          |_ellipsoids_node
          |_convergence
    |_models
    |_results
```

- If your system does not support relative paths with "../", it may be helpful to add the path to the project's `code` folder to the `PYTHONPATH` (for importing files as modules), e.g.:
```
export PYTHONPATH="<path/to/mfcn/code>":$PYTHONPATH
```

- Set general experiment arguments in `args_template.py`. Note that the `__post_init__` method in this file is how the project directories on your machine(s) are set. Modify this method with the keys and paths of your choice. (This provides a convenient way to pass this key as a command line argument in the data processing and experiment execution scripts, and set directories correctly when running the scripts on different systems.) Note that many of the arguments in `args_template.py` store default values which are overridden in the task-specific args files.

- Set experiment-specific arguments (which will override those in `args_template.py`) in `ellipsoids_node_regress/args.py` or `melanoma/args.py` files.

