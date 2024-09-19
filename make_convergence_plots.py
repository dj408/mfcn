"""
Script to generate and save convergence
experiments plots.
"""
import sys
sys.path.insert(0, '../')
import config as cf
import manifold_sampling as ms
import plotting_convergence as pc

import pickle
import numpy as np
import matplotlib.pyplot as plt

case = cf.SAMPLING_CASES[0][0] # 'uniform'
density_ratio_str = str(cf.SAMPLING_CASES[0][1]) # '1'
filename = cf.get_exp_results_filename(
    case, 
    density_ratio_str
)

"""
open results pickle
"""
print(f'opening \'{filename}\'')
with open(f"{cf.SAVE_DIR}/{filename}", "rb") as f:
    records_l = pickle.load(f)


plt.rcParams.update({
    "figure.figsize": cf.FIG_SIZE,
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12
})

"""
eigenvalues convergence plot
"""
pc.sphere_eigenvalues_convergence_plot(
    cf.GRAPH_TYPE,
    run_is=range(cf.N_RUNS),
    last_eval_i=cf.LAST_EIGENVAL_IDX,
    Ns=cf.Ns,
    records_l=records_l,
    plot_style='runs', # quantiles_error, runs, quantiles
    mult_true_evals=cf.get_nontriv_sphere_LBO_eigenvals(),
    save_path=f'{cf.PLOT_SAVE_PATH}/eigenvalues_converg_plot.png'
)


"""
spectral filter convergence plot
"""
plt.clf()
runs_norms_by_n, runs_max_norms, ns = pc.get_spect_filt_norms(
    Ns=cf.Ns,
    records_l=records_l,
    spectral_filter_fn=cf.lowpass_spectral_filter,
    get_LBO_eigenvals_fn=cf.get_nontriv_sphere_LBO_eigenvals,
    eval_wLf_fn=cf.eval_wLf,
    norm_eval_op=ms.norm_eval_op,
    norm_type=cf.CONVERG_NORM
)

slope = pc.get_filter_converg_loglog_slope(
    runs_norms_by_n,
    ns
)
print(f'log-log regression slope = {slope:.4f}')

pc.spectral_filter_convergence_plot(
    cf.GRAPH_TYPE,
    runs_norms_by_n, 
    runs_max_norms, 
    ns,
    log_y_axis=True,
    save_path=f'{cf.PLOT_SAVE_PATH}/spectral_filter_converg_plot.png',
    show_plot=True
)

