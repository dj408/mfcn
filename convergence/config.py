"""
Configuration file for spectral convergence
experiments.

IMPORTANT: set save paths here.
"""
import numpy as np
from typing import (
    Tuple,
    List,
    Optional,
    Callable
)


"""
general experiment configuration
"""
N_RUNS = 10
MACHINE = 'local'
if MACHINE == 'local':
    SAVE_DIR = "./"
    PLOT_SAVE_PATH = f'{SAVE_DIR}/figures'
elif MACHINE in ('cluster', 'CLUSTER'):
    SAVE_DIR = "/scratch/user/${USER}/"
    PLOT_SAVE_PATH = f'{SAVE_DIR}/figures'
PICKLE_FILE_SUFFIX = 'pkl'
VERBOSITY = 1


"""
manifold params
"""
MANIFOLD_TYPE = 'sphere'
DENSITY_ON_MANIFOLD = 1. / (4. * np.pi)

if MANIFOLD_TYPE == 'circle':
    D_MANIFOLD = 1
    RADIUS = 1. # or, could set so vol of 2d unit ball is 1...
elif MANIFOLD_TYPE == 'sphere':
    D_MANIFOLD = 2 
    SPH_HARM_SEED = 286534
    # SPH_HARM_FN_MAX_DEG = 5
    # N_SPH_HARM_FNS = 5
    RADIUS = 1. # or 3. / (4. * np.pi) # so volume = 1.


"""
sampling params
"""
SAMPLING_SEEDS = (
    483271,
    786545,
    685234,
    905482,
    734509,
    111345,
    572345,
    325489,
    218947,
    578329
)
# note 2^12 = 4096, but need 13 to include in np.arange
Ns = 2 ** np.arange(6, 15) 
INITIAL_SAMPLE_N = Ns[-1] * 2
SAMPLING_CASES = (
    ('uniform', 1),
)
# params controlling bias and density ratio for nonuniform sampling
NONUNIF_MODE = 'oversample'
NONUNIF_SQUARED = False

"""
graph construction params
"""
GRAPH_TYPE = 'knn' # 'knn'
ETA_TYPE = 'indicator' # 'indicator', 'identity'
if GRAPH_TYPE == 'epsilon':
    # mode/value for epsilon-graph's epsilon param
    EPSILON = 'auto' # 'quantile'
    # note: higher eps_constant seems to make convergence of
    # higher-frequency eigenvalues slower
    EPS_CONSTANT = 1. # 4.5 # 1., 0.2
elif GRAPH_TYPE == 'knn':
    KNN_K = 'auto'
    K_CONSTANT = 1.


"""
filter params
"""
# simple spectral lowpass filter (e^-t)
def lowpass_spectral_filter(eigenvals):
    return np.exp(-eigenvals)

# wavelet filters
J = 4 # max wavelet order
Q = 3 # max scattering moment norm


"""
sphere spectral params
"""
sph_harm_params = [
    {'weight': 1.,
     'degree': 1,
     'order': 0},
    {'weight': 1.,
     'degree': 2,
     'order': 0}
]
# or, generate random linear combination of sph. harmonic functions
# to use as the measurable, continuous function on the manifold
# (no repeated degrees, to avoid eigenvalue multiplicity)
# sph_harm_rs = np.random.RandomState(seed=cf.SPH_HARM_SEED)
# sph_harm_params = ms.gen_rand_lin_combo_sph_harm_params(
#     n_terms=cf.N_SPH_HARM_FNS,
#     max_deg=cf.SPH_HARM_FN_MAX_DEG,
#     random_state=sph_harm_rs
# )

def get_nontriv_sphere_LBO_eigenvals(
    max_eigenval_i: int = 2
) -> np.ndarray:
    """
    Computes the Laplace-Beltrami operator eigenvalues for epsilon-graph 
    of sphere, ignoring the trivial 0 first eigenvalue.
    These are needed for computing continuum filtrations ('wLfs').

    Args:
        max_eigenval_i: index of the maximum eigenvalue
            to return, excluding the first (index 0) and
            multiplicity.
    Returns:
        Array of LBO eigenvalues.
    """
    if 'eps' in GRAPH_TYPE:
        denom = 8. * np.pi
    elif GRAPH_TYPE == 'knn':
        denom = 1. / (2. * np.pi)
        
    LBO_evals = np.array([
        i * (i + 1) / denom \
        for i in range(1, max_eigenval_i + 1)
    ])
    # LBO_evals = [2., 6., 12.]
    return LBO_evals


def eval_wLf(
    LBO_eigenvals: float,
    thetas_psis: np.ndarray,
    spectral_filter_fn: Callable = lowpass_spectral_filter
) -> float:
    """
    Spectral filter evaluation of a function 
    on the continuum of a 2-sphere (hence uses
    a Laplace-Beltrami Operator (LBO) eigenvalues).

    The function spelled out here must match the
    spherical harmonic functions spelled out in
    'sph_harm_params' above.

    Notes: (1) wLf needs to be normalized-evaluated
    in order to be compared to a discretized/sampled
    evaluation filtration; (2) Our theta/psi parameter-
    ization is swapped vs. Wikipedia's articles on 
    sph. coords/harmonics.
    
    Args:
        LBO_eigenval:
        thetas_psis: shape (n, 2).
        spectral_filter_fn:
    Returns:
        Float value of the filtration on the
        manifold.
    """
    wLf = spectral_filter_fn(LBO_eigenvals[0]) * \
        0.5 * np.sqrt(3. / np.pi) * np.cos(thetas_psis[:, 1]) \
        + spectral_filter_fn(LBO_eigenvals[1]) * \
        0.25 * np.sqrt(5. / np.pi) * (3. * np.cos(thetas_psis[:, 1]) ** 2 - 1)
    return wLf
    

"""
eigendecomp. of Laplacian params
"""
# kappa = max number of eigenpairs (k < n for eigendecomp. of sparse matrices!)
# higher kappa seems to help convergence of discrete spectral filter,
# but potentially at computational cost of spectral decomp...
KAPPAS = [64] * len(Ns) # [int(N / 8) for N in Ns]
# how many eigenvalues/vectors to save in records?
# sph. harmonics eigenvals = l(l + 1) [l is degree] with multiplicity 2l + 1
N_EIGENPAIRS_SAVE = np.sum([2 * i + 1 for i in range(3)])


"""
unit test params
"""
RUN_UNIT_TESTS = False
ABS_EQUIV_THRESH = 1.0e-12
REL_EQUIV_THRESH = 0.95


"""
plotting params
"""
FIG_SIZE = (8, 3)
CONVERG_NORM = 2
LAST_EIGENVAL_IDX = 9


"""
experiment-wide getter functions
"""
def get_exp_results_filename(
    case: str = 'uniform',
    density_ratio_str: str = '1'
) -> str:
    """
    Produces a filename string based
    on the experiment parameters set
    in this file. Useful for saving
    then retrieving a file of experiment
    results.

    Args:
        case: sampling case ('uniform' or
            'nonuniform').
        density_ratio_str: string value of
            the density ratio.
    Returns:
        String to use a filename.
    """
    filename = f'{MANIFOLD_TYPE}_{GRAPH_TYPE}_' \
    + f'{case}_{density_ratio_str}_' \
    + f'{N_RUNS}runs_N{min(Ns)}-{max(Ns)}' \
    + f'.{PICKLE_FILE_SUFFIX}'
    return filename

