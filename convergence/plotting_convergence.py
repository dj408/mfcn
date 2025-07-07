"""
Functions to create (and save and/or
display) convergence of (1) spectral filters
and (2) eigenvalues on the sphere, from
experiment results records_l, plus needed
helper functions.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import (
    Tuple,
    List,
    Optional,
    Callable,
    Iterable
)


def get_spect_filt_norms(
    Ns: Tuple[int],
    records_l: List[List[dict]],
    spectral_filter_fn: Callable,
    get_LBO_eigenvals_fn: Callable,
    eval_wLf_fn: Callable,
    norm_eval_op: Callable,
    norm_type: int | str = 2, # 'inf'
    start_n_i: Optional[int] = None,
    end_n_i: Optional[int] = None
) -> Tuple[List[float], List[float], List[int]]:
    r"""
    Computes the discretization errors of the 
    spectral filter $w(L_n)x$ versus the analagous
    filter on the manifold, $w(\mathcal{L})f$. This
    error is measured as the norm of the difference
    between the discrete and continuum normalized-
    evaluated filtered function values.
    
    Args:
        Ns: samples sizes used in each experiment run.
        records_l: list of experiment results records
            for each run. Each run's record is a list of
            dictionaries, one for each sample size N.
        spectral_filter_fn: function that applies the
            spectral filter (e.g., $\e^{-lambda}$).
        get_LBO_eigenvals_fn: function that computes the
            (known) eigenvalues of the Laplace-Beltrami 
            operator LBO.
        eval_wLf_fn: function that applies the spectral
            filter function to the (known) eigenfunctions
            of the LBO of a manifold, using its known 
            eigenvalues.
        norm_eval_op: function that normalizes the evaluated
            points on a manifold.
        norm_type: which l-norm to use when evaluating the
            discretization error: int or 'inf'.
        start_n_i: optional left subsetting index of Ns, to 
            prevent plotting smaller sample sizes.
        end_n_i: optional right subsetting index of Ns, to 
            prevent plotting larger sample sizes.
    Returns:
        Tuple of lists: all convergence norms by run, 
        max norms by run, and (possibly subsetted) sample
        sizes.
    """
    # init empty results containers
    runs_norms_by_n = [None] * len(records_l)
    runs_max_norms = [None] * len(records_l)
    
    # set N indexes, subsetting if needed
    N_is = range(len(Ns))
    if start_n_i is not None:
        N_is = N_is[start_n_i:]
        if end_n_i is not None:
            N_is = N_is[:(end_n_i + 1 - start_n_i)]
    else:
        if end_n_i is not None:
            N_is = N_is[:(end_n_i + 1)]
    
    # 'records_l' is a list (by run) of one sampling case's records_l 
    # (a list (by N) of dicts)
    for run_i, rec in enumerate(records_l):
        # print(f'processing record for run {run_i + 1}')
        thetas_psis_by_n = [rec[n_i]['x_intrinsic'] for n_i in N_is]
        ns = [rec[n_i]['n'] for n_i in N_is]
        wLnxs = [rec[n_i]['low_pass_convolution'] for n_i in N_is]
        
        # evaluate filtered eigenfunctions
        LBO_eigenvals = get_LBO_eigenvals_fn()
        wLfs = [
            norm_eval_op(
                eval_wLf_fn(
                    LBO_eigenvals, 
                    thetas_psis, 
                    spectral_filter_fn
                ),
                n
            ) \
            for (thetas_psis, n) in zip(thetas_psis_by_n, ns)
        ]

        # compute norms/discretization errors
        ord = np.inf if norm_type == 'inf' else norm_type
        norms_by_n = [
            np.linalg.norm(wLnx - wLf, ord=ord) \
            for wLnx, wLf in zip(wLnxs, wLfs)
        ]
        runs_norms_by_n[run_i] = norms_by_n
        runs_max_norms[run_i] = np.max(norms_by_n)
        
        # print table of norm values by N
        # print('n', ' ' * 4, f'{norm_type}-norm')
        # for n, norm in zip(ns, norms_by_n):
        #     print(f'{n:>6} {norm:.12f}')
    return runs_norms_by_n, runs_max_norms, ns


def get_filter_converg_loglog_slope(
    runs_norms_by_n: List[float],
    ns: List[int]
) -> float:
    """
    Calculates the slope for a log-log
    regression of filter convergence norms
    computed in 'get_spect_filt_norms' against
    sample sizes.
    
    Args:
        runs_norms_by_n: list of convergence norms in order
            of n, for each run.
        ns: list of (possibly subsetted, in 'get_spect_filt_norms') 
            sample sizes.
    Returns:
        Slope coefficient.
    """
    # log10 and combine all data across all runs
    data = [None] * len(runs_norms_by_n)
    for i, norms_by_n in enumerate(runs_norms_by_n):
        data[i] = np.stack(
            (np.log10(ns), np.log10(norms_by_n)),
            axis=1
        )
    data = np.concatenate(data)
    
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    slope = reg.coef_[0].item()
    return slope
    


def spectral_filter_convergence_plot(
    graph_type: str,
    runs_norms_by_n: List[float],
    runs_max_norms: List[float], 
    ns: List[int],
    plot_style: str = 'quantiles', # 'raw', 'mean'
    quantiles_style_q: Tuple[int] = (25, 50, 75),
    figsize: Tuple[float, float] = (8., 3.),
    title: str = r'Convergence of spectral filter $w({\lambda}) = e^{-\lambda}$ on the sphere',
    log_y_axis: bool = False,
    ylabel: str = 'discretization error',
    save_path: str = None,
    show_plot: bool = False
) -> None:
    r"""
    Plots the discretization error of a discrete spectral
    filter approximation to a spectral manifold filter,
    over increasing sample size (of function-on-the-manifold
    values).

    Args:
        graph_type: string key of graph type (e.g., 
            'knn' or 'eps').
        runs_norms_by_n: list of convergence norms in order
            of n, for each run.
        runs_max_norms: list of max norm for each n, for
            each run.
        ns: list of (possibly subsetted, in 'get_spect_filt_norms') 
            sample sizes.
        plot_style: key for plotting mean + std of runs'
            results ('mean'), quantiles ('quantiles'),
            or all raw results' lines ('raw').
        quantiles_style_q: tuple holding quantile values (0-100)
            for np.percentile's 'q' parameter, used when 
            plot_style == 'quantiles'. Use (0, 50, 100)
            for min-median-max.
        figsize: plt's figure size parameter.
        title: title for the plot.
        log_y_axis: whether to display the y-axis at log scale.
        ylabel: label for the y-axis.
        save_path: optional filepath for saving the plot.
        show_plot: whether to call plt.show() for the plot.
    Returns:
        None; saves and/or displays plot.
    """
    plt.rcParams["figure.figsize"] = figsize
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Helvetica"]

    N_subset_is = range(len(runs_norms_by_n[0]))
    
    # plot the norms
    if plot_style == 'raw':
        for norms_by_n in runs_norms_by_n:
            plt.plot(ns, norms_by_n, c='tab:blue', alpha=0.7)
            
    elif plot_style == 'mean':
        
        central_norms_by_n = [
            np.mean([norms_by_n[n_i] \
                     for norms_by_n in runs_norms_by_n]) \
            for n_i in N_subset_is
        ]
        std_norms_by_n = [
            np.std([norms_by_n[n_i] for norms_by_n in runs_norms_by_n]) \
            for n_i in N_subset_is 
        ]
        # print(std_norms_by_n)
        upper_bounds = [
            central_norms_by_n[n_i] + std_norms_by_n[n_i] \
            for n_i in N_subset_is  
        ]
        lower_bounds = [
            central_norms_by_n[n_i] - std_norms_by_n[n_i] \
            for n_i in N_subset_is  
        ]
        lower_bounds = [b if b >= 0.0 else 0.0 for b in lower_bounds]
        
    elif plot_style == 'quantiles':
        quantile_norms_by_n = np.stack([
            np.percentile(
                [norms_by_n[n_i] for norms_by_n in runs_norms_by_n],
                q=quantiles_style_q,
                method='linear'
            ) \
            for n_i in N_subset_is
        ])

        central_norms_by_n = quantile_norms_by_n[:, 1]
        lower_bounds = quantile_norms_by_n[:, 0]
        upper_bounds = quantile_norms_by_n[:, 2]
        
    # print(lower_bounds)
    plt.plot(ns, central_norms_by_n, c='tab:blue', alpha=1.)
    plt.fill_between(
        ns, lower_bounds, upper_bounds, 
        color='gray', alpha=0.5
    )

    # x axis
    plt.xlabel(r'$n$')
    plt.xscale('log', base=2)
    plt.xticks(ns)

    # y axis
    if log_y_axis:
        plt.yscale('log', base=10)
    plt.ylabel(ylabel)
    ylim_top = max(runs_max_norms) * 1.1 \
        if plot_style == 'raw' \
        else max(upper_bounds) * 1.1
    # plt.ylim(bottom=0., top=ylim_top)
    # plt.minorticks_off()
    
    # title_norm = '\infty' if norm_type == 'inf' else '2'
    # plt.title(
    #     rf'$||w(L_n) P_n f - P_n w(\mathcal{{L}}) f||_{title_norm}$ by $n$'
    #     '\n(low pass spectral filter, uniform sampling of 2-sphere)'
    # )
    if 'eps' in graph_type:
        title += r' ($\epsilon$-graph)'
        plt.yticks([1e-3, 1e-2, 1e-1])
        
    elif 'knn' in graph_type:
        title += r' ($k$-NN graph)'
        plt.yticks([1e-3, 1e-2, 1e-1])
    plt.title(title)
    plt.grid(which='both')

    # optional save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show_plot:
        plt.show()


def sphere_eigenvalues_convergence_plot(
    graph_type: str,
    Ns: Tuple[int],
    records_l: List[List[dict]],
    plot_style: str = 'quantiles', # 'single_run'
    quantiles_style_q: Tuple[int] = (25, 50, 75),
    last_eval_i: int = 10, # multiplicity 2i + 1
    skip_first_eval: bool = True,
    n_nontriv_eigenvals: int = 2,
    mult_true_evals: List[float] = None,
    run_is: Iterable[int] = None,
    figsize: Tuple[float, float] = (8., 3.),
    include_legend: bool = False,
    title: str = r'Convergence of eigenvalues on the sphere', # for $\phi_1^n$ and $\phi_3^n$',
    multiplicitous_colors: Tuple[str] = (
        'tab:blue', 'tab:green', 'tab:orange', 'tab:red'
    ),
    save_path: str = None,
    show_plot: bool = False
) -> None:
    r"""
    Plots the convergence of a graph Laplacian's
    eigenvalues (hopefully) to their expected spherical 
    harmonic values, with increasing sample size of 
    function-on-the-manifold values.

    Note that these eigenvalues have multiplicity:
    $\lambda_i = i(i + 1)$ for $i \geq 0$, with 
    multiplicity $2i + 1$.

    Args:
        graph_type: string key of graph type (e.g., 
            'knn' or 'eps').
        Ns: samples sizes used in each experiment run.
        records_l: list of experiment results records
            for each run. Each run's record is a list of
            dictionaries, one for each sample size N.
        plot_style: style of plot to produce: whether to 
            plot one run's eigenvalues ('single_run'), or
            to display the quantiles of aggregated runs
            ('quantiles') or quantiles of errors 
            ('quantiles_error').
        quantiles_style_q: tuple holding quantile values (0-100)
            for np.percentile's 'q' parameter, used when 
            plot_style == 'quantiles'. Use (0, 50, 100)
            for min-median-max.
        last_eval_i: int = 10, # multiplicity 2i + 1
        skip_first_eval: if true, the first eigenvalue
            (e.g. if trivial/0) is excluded.
        n_nontriv_eigenvals: number of nontrivial eigenvalues
            to plot, in their spherical multiplicity 2i + 1.
        mult_true_evals: list of true/expected nontrivial 
            eigenvalues, ignoring multiplicity, e.g. 
            np.array([2, 6]) * 2. * np.pi for k-NN graphs 
            on sphere; np.array([2, 6]) / (8. * np.pi) for 
            epsilon graphs on sphere.
        run_is: indexes of the runs in 'records_l' to use as 
            data for the plot, if plot_syle == 'runs'.
        figsize: plt's figure size parameter.
        include_legend: whether to include a legend labeling
            the eigenvalues.
        title: title for the plot.
        multiplicitous_colors: which colors to apply to 
            eigenvalues within the same multiplicity.
        save_path: optional filepath for saving the plot.
        show_plot: whether to call plt.show() for the plot.
    Returns:
        None; saves and/or displays plot.
    """
    plt.rcParams["figure.figsize"] = figsize
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Helvetica"]

    start_eval_i = 1 if skip_first_eval else 0
    multiplicities = [
        (2 * i + 1) for i in range(start_eval_i, n_nontriv_eigenvals + 1)
    ]
    last_eval_i = np.sum(multiplicities) + 1
    # print(multiplicities) # [3, 5]

    if plot_style == 'runs':
        if len(run_is) == 1:
            alpha = 1.0
        else:
            alpha = 1.0 / len(run_is) * 3.

        for run_i in run_is:
            one_run_evals_by_n = np.array([
                rec_n['eigenvalues'][start_eval_i:(last_eval_i)] \
                for rec_n in records_l[run_i]
            ]).T # shape (n_evals, n_Ns)
            # print(one_run_evals_by_n.shape)
            # print(one_run_evals_by_n)
    
            # color eigenvalues by multiplicity
            eigenvals_colors = []
            for j, mult in enumerate(multiplicities):
                eigenval_color = multiplicitous_colors[j]
                eigenvals_colors.extend([eigenval_color] * mult)
            # print(eigenvals_colors)
        
            for i, one_run_evals \
            in enumerate(one_run_evals_by_n):
                plt.plot(
                    Ns, 
                    one_run_evals,
                    c=eigenvals_colors[i],
                    alpha=alpha,
                    label=fr'$\lambda_{i + 1}$'
                )
    elif 'quantiles' in plot_style:
        # shape (n_runs/records, n_Ns, n_evals)
        runs_evals_by_n = np.stack([
            np.array([
                rec_n['eigenvalues'][start_eval_i:last_eval_i] \
                for rec_n in record
            ]) \
            for record in records_l
        ])
        # shape (n_runs/records, n_Ns, n_evals) = (10, 9, 8)
        
        mults = [0] + multiplicities
        mult_arrays = [
            runs_evals_by_n[:, :, mults[i-1]:(mults[i-1] + mults[i])] \
            for i in range(1, len(mults))
        ]
        
        evals_quantiles_by_n_l = [None] * len(mult_arrays)
        for i, mult_arr in enumerate(mult_arrays):
            mult_arr = np.swapaxes(mult_arr, 0, 1) \
                .reshape((len(Ns), -1)) # shape (n_Ns, n_runs * n_evals)

            if 'error' in plot_style:
                mult_true_eval = mult_true_evals[i]
                mult_arr = np.abs(mult_arr - mult_true_eval)
                
            evals_quantiles_by_n_l[i] = np.percentile(
                mult_arr,
                q=quantiles_style_q,
                axis=1,
                method='linear'
            ).T # shape (n_Ns, len(q))

        for i, quantiles_by_n in enumerate(evals_quantiles_by_n_l):
            medians_by_n = quantiles_by_n[:, 1]
            lower_bounds = quantiles_by_n[:, 0]
            upper_bounds = quantiles_by_n[:, 2]
            
            plt.plot(
                Ns, 
                medians_by_n, 
                c=multiplicitous_colors[i], 
                alpha=1.
            )
            plt.fill_between(
                Ns, lower_bounds, upper_bounds, 
                color='gray', alpha=0.5
            )
        
    # y axis
    if 'error' not in plot_style:
        # plt.yticks([0, 2, 6])
        if 'eps' in graph_type:
            exp_evals_nonmult = np.array([
                (i * (i + 1.)) / (8. * np.pi) \
                for i in range(0, 3)
            ])
            ylabels = [
                fr'$\frac{{{i}({i} + 1)}}{{8\pi}}$' \
                for i in range(0, 3)
            ]
            
        elif graph_type == 'knn':
            # plt.yticks([0, 2, 6])
            exp_evals_nonmult = np.array([
                (i * (i + 1.)) * (2. * np.pi) \
                for i in range(0, 3) # (2. * np.pi)
            ])
            ylabels = [
                fr'${i}({i} + 1) \cdot 2\pi$' \
                for i in range(0, 3)
            ]
            
        ylabels[0] = '0'
        # plt.ylabel(, rotation=0., labelpad=12.)
        plt.yticks(
            exp_evals_nonmult,
            # np.concatenate((np.array([0.]), exp_evals_nonmult)), 
            labels=ylabels # ['0'] + 
        )
    elif 'error' in plot_style:
        if 'eps' in graph_type:
            plt.ylabel(r'absolute error from $i(i+1) / 8\pi$')
        elif graph_type == 'knn':
            plt.ylabel(r'absolute error from $i(i+1) \cdot 2\pi$')
    
    # x axis
    plt.xlabel(r'$n$')
    plt.xscale('log', base=2)
    plt.xticks(Ns)

    # title, legend, grid
    if 'eps' in graph_type:
        title += r' ($\epsilon$-graph)'
    elif graph_type == 'knn':
        title += r' ($k$-NN graph)'
        
    plt.title(title)
    if include_legend:
        plt.legend(reverse=True, loc='lower left')
    plt.grid()
    
    # optional save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show_plot:
        plt.show()
    
        