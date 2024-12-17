# README

The code in this repository reproduces the numerical experiments found in the "Convergence of Manifold Filter-Combine Networks" (Johnson et al.) paper from the NeurIPS 2024 Symmetry and Geometry in Neural Representations workshop. To use this code:

1. Set save paths and experiment parameters in the `config.py` file.
2. Run the main experiment script: `python3 convergence_experiment.py`
3. To reproduce the spectral filter and eigenvalue convergence plots, run the plot creation script:`python3 make_convergence_plots.py`
