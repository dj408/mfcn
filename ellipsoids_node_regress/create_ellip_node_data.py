"""
Script to generate synethic data for a node-
level ellipsoid regression task.

Notes: 
- make sure experiment args are set in
'ellipsoids_node_regress.args.py'.
- '--type combination' creates datasets of linear combinations of 
eigenvectors 1...evec_i (indexed from 0; 0th is trivial/constant)
- '--type single' creates datasets of single eigenvector values, 
from first nonconstant eigenvector up through --evec_i

Example call:
python3.11 create_ellip_node_data.py \
-m desktop \
-n 1024 -a 3 -b 2 -c 1 \
--type combination \
--ambient_dim 8 \
--add_noise \
--n_datasets 10 \
--min_evec_i 1 \
--max_evec_i 20 \
--save_plots
"""

import sys
sys.path.insert(0, '../')
import os
import argparse
import pickle
from importlib import reload
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import special_ortho_group

import dataset_creation as dc
import manifold_sampling as ms
import graph_construction as gc
import wavelets as w
import dataset_creation as dc
import utilities as u
import plotting
import ellipsoids_node_regress.args as a

# time script execution
t_0 = time.time()


"""
clargs
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine', default='desktop', type=str, 
                    help='key for the machine in use (default: desktop): see args_template.py')
parser.add_argument('-n', default='1024', type=int, 
                    help='number of points to sample on ellipsoid (default=1024)')
parser.add_argument('-a', default='3', type=int, 
                    help='scale parameter for ellipsoid x-axis (default=3)')
parser.add_argument('-b', default='2', type=int, 
                    help='scale parameter for ellipsoid y-axis (default=2)')
parser.add_argument('-c', default='1', type=int, 
                    help='scale parameter for ellipsoid z-axis (default=1)')
parser.add_argument('-t', '--type', default='combination', type=str, 
                    help="dataset type to create: 'combination' of eigenvectors (up to the ith eigenvector passed to --evec_i) or 'single' for the (ith/--evec_i) eigenvector alone; default='combination'")
parser.add_argument('-e', '--ambient_dim', default='3', type=int, 
                    help='ambient dimension of 2-ellipsoids (default=3)')
parser.add_argument('-o', '--add_noise', default=False, action='store_true')
parser.add_argument('-i', '--min_evec_i', default='1', type=int, 
                    help='minimum eigenvector index to use, indexing from 0 (default=1)')
parser.add_argument('-x', '--max_evec_i', default='20', type=int, 
                    help='maximum eigenvector index to use, indexing from 0 (default=20)')
parser.add_argument('-s', '--evec_stride', default='1', type=int, 
                    help='index stride value for eigenvectors to include/skip (default=1)')
parser.add_argument('-d', '--n_datasets', default='1', type=int, 
                    help='number of ellipsoids/datasets to generate (default=1)')
parser.add_argument('-r', '--coef_rand_dist', default='unif', type=str, 
                    help="random distribution type for sampling linear combination coefficients when '--type combination' (default='unif')")
parser.add_argument('--save_plots', default=False, action='store_true')
clargs = parser.parse_args()

args = a.Args(
    MACHINE=clargs.machine,
    N_PTS_ON_MANIFOLD=clargs.n,
    AMBIENT_DIM=clargs.ambient_dim,
    ADD_NOISE_TO_MANIFOLD_SAMPLE=clargs.add_noise
    
)
abc = np.array([clargs.a, clargs.b, clargs.c])
random_state = np.random.RandomState(args.POINT_SAMPLING_SEED)

# '--type combination' creates datasets of linear combinations of 
# eigenvectors 1...evec_i (indexed from 0; 0th is trivial/constant)
evec_range = range(clargs.min_evec_i, clargs.max_evec_i + 1, clargs.evec_stride)
save_root = f"{args.DATA_DIR}/{args.RAW_DATA_FILENAME}"

# set 'combination' vs. 'single' eigenvector dataset vars
if 'comb' in clargs.type.lower():
    n_datasets = clargs.n_datasets
    datasets_save_dir = f"{save_root}_evecs{clargs.min_evec_i}-{clargs.max_evec_i}_combinations"
    if clargs.ambient_dim > 3:
        datasets_save_dir += f"_ambient{clargs.ambient_dim}"
    if clargs.add_noise:
        datasets_save_dir += f"_noisy"
    save_filenames = [f"comb_{i}.pkl" for i in range(clargs.n_datasets)]
    figures_save_dir = f"figures/evec_combinations"
    
# '--type single' creates datasets of single eigenvector values, 
# from first nonconstant eigenvector up through --evec_i
elif 'single' in clargs.type.lower():
    datasets_save_dir = f"{save_root}_single_evecs{clargs.min_evec_i}-{clargs.max_evec_i}"
    if clargs.evec_stride is not None and clargs.evec_stride > 1:
        datasets_save_dir += f"_stride{clargs.evec_stride}"
    if clargs.ambient_dim > 3:
        datasets_save_dir += f"_ambient{clargs.ambient_dim}"
    if clargs.add_noise:
        datasets_save_dir += f"_noisy"
    save_filenames = [f"evec_{i}.pkl" for i in evec_range]
    n_datasets = len(save_filenames)
    figures_save_dir = f"figures/single_evecs"

else:
    raise NotImplementedError(
        f"Ellipsoids dataset type={clargs.type} not implemented."
    )

# make datasets datasets_save_dir if it doesn't already exist
os.makedirs(datasets_save_dir, exist_ok=True)
os.makedirs(figures_save_dir, exist_ok=True)


"""
Loop through the number of ellipsoids/datasets needed
"""
for i in range(n_datasets):
    print(f'\nCreating dataset {i+1}.')
    """
    Generate ellipsoid(s) uniform point samples
    """
    manifold_xyz = np.stack(
        ms.unif_ellipsoid_xyz(
            n=args.N_PTS_ON_MANIFOLD, 
            abc=abc,
            random_state=random_state
        ), 
        axis=-1
    )

    """
    AMBIENT DIMENSION > 3 [optional]
    Embed and rotate 2-ellipsoids in ambient dim. > 3
    build using scipy.sparse 'lil_matrix', then convert to
    'csc_matrix' where columns are non-sparse dimensions
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix
    """
    intrinsic_dim = manifold_xyz.shape[1]
    if args.AMBIENT_DIM > intrinsic_dim:
        print(f'\tEmbedding ellipsoid into {args.AMBIENT_DIM}-d space.')

        # put existing dims in leftmost cols of new (n, new_dim) matrix
        # where rest of cols are filled with 0s

        # dense construction
        # manifold_pts = np.concatenate(
        #     (manifold_pts, 
        #      np.zeros(
        #         (manifold_pts.shape[0], 
        #          new_dim - manifold_pts.shape[1])
        #     )), 
        # axis=-1)

        # sparse construction
        col_indices = range(intrinsic_dim)
        manifold_pts = lil_matrix((manifold_xyz.shape[0], args.AMBIENT_DIM))
        for j in col_indices:
            manifold_pts[:, j] = manifold_xyz[:, j]
        manifold_pts = manifold_pts.tocsc()

        # apply uniformly random rotation operation in the new ambient dim
        # rrm_random_state = np.random.RandomState(args.RAND_ROTATION_SEEDS[i])
        rrm_gen_frozen = special_ortho_group(
            dim=args.AMBIENT_DIM, 
            seed=args.RAND_ROTATION_SEEDS[i]
        )
        rrm = rrm_gen_frozen.rvs(1)
        manifold_pts = manifold_pts @ rrm
    else:
        manifold_pts = manifold_xyz
    # manifold_pts.shape = (n_samples, ambient_dim)

    """
    NOISE [optional]
    """
    if args.ADD_NOISE_TO_MANIFOLD_SAMPLE:
        print(f'\tAdding Gaussian noise to ellipsoid point sample.')
        # before adding noise, compute 'true' eigenvectors
        # from graph Laplacian of a k-NN graph constructed from
        # no-noise uniformly sampled points lying on the true manifold
        graph_for_eigenvecs_true = gc.KNNGraph(
            x=manifold_pts,
            n=args.N_PTS_ON_MANIFOLD,
            k=args.K_OR_EPS, # 'auto',
            d_manifold=args.D_MANIFOLD,
            eta_type='indicator'
        )
        graph_for_eigenvecs_true.calc_Laplacian()
        k_decomp_for_eigenvecs_true = max(args.KAPPA + 1, clargs.max_evec_i + 1)
        eigenvals_true, eigenvecs_true = eigsh(
            graph_for_eigenvecs_true.L, 
            k=k_decomp_for_eigenvecs_true, 
            which='SM'
        )
        del graph_for_eigenvecs_true

        # generate and add Gaussian noise to manifold_pts
        noise = ms.gaussian_noise(
            d=manifold_pts.shape[1], 
            n=manifold_pts.shape[0], 
            seed=args.MANIFOLD_SAMPLE_NOISE_SEEDS[i],
            var_constant=args.SAMPLE_NOISE_VAR_CONSTANT
        )
        manifold_pts += noise
        del noise

    
    r"""
    GRAPH CONSTRUCTION
    Construct graph based on observed (i.e. possibly noisy manifold
    sample data), compute its graph Laplacian, and eigendecompose to 
    obtain the first $\kappa$ eigenvectors for MFCN-spectral.
    """
    if args.GRAPH_TYPE == 'knn':
        graph = gc.KNNGraph(
            x=manifold_pts,
            n=args.N_PTS_ON_MANIFOLD,
            k=args.K_OR_EPS, # 'auto',
            d_manifold=args.D_MANIFOLD,
            eta_type='indicator'
        )
    elif args.GRAPH_TYPE == 'epsilon':
        graph = gc.EpsilonGraph(
            x=manifold_pts,
            n=args.N_PTS_ON_MANIFOLD,  
            eps=args.K_OR_EPS, # 'auto'
            d_manifold=args.D_MANIFOLD,
            eta_type='indicator'
        )
    
    graph.calc_Laplacian()
    k_decomp = max(args.KAPPA + 1, clargs.max_evec_i + 1)
    eigenvals_obs, eigenvecs_obs = eigsh(graph.L, k=k_decomp, which='SM')
    # enforce strictly zero first eigenvalue (eigsh may return near-zero 
    # first eigenvalues numerically)
    eigenvals_obs[0] = 0.
    # note eigenvecs_obs has shape (N, k)

    
    """
    TARGETS
    Set node signal regression target to a nontrivial 'true' eigenvector 
    value, or a linear combination of higher-order 'true' eigenvectors, 
    and rescale to a numerically nice interval.
    """
    # if no noise was added to sampled points, use the observed
    # eigenvectors as the 'true' eigenvectors for generating targets
    if not args.ADD_NOISE_TO_MANIFOLD_SAMPLE:
        eigenvecs_true = eigenvecs_obs
        
    if 'comb' in clargs.type.lower():
        target_values = ms.rand_bandlimit_evec_signal(
            eigenvecs_true,
            clargs.max_evec_i,
            random_state,
            clargs.coef_rand_dist, # 'unif'
            datasets_save_dir # to save record of random coef values
        )
    elif 'single' in clargs.type.lower():
        # signal values = 'single' (i-th) eigenvector from one ellipsoid 
        # (typically skipping the first trivial/constant eigenvector)
        target_values = eigenvecs_true[:, evec_range[i]]
        # target_values = eigenvecs_obs[:, (i + 1)]
            
    # map/rescale target signal values onto [-1, 1] -> first map onto [0, 1], 
    # then take 2x - 1. This generally helps a regressor head
    max_val, min_val = np.max(target_values), np.min(target_values)
    target_values = (target_values - min_val) / (max_val - min_val)
    target_values = 2. * target_values - 1
    
    
    """
    PLOTS OF SIGNALS [optional]
    Save plots of signal results (if --save_plots)
    """
    if clargs.save_plots:
        # add 1 to evec index; math usually indexes from 1 not 0
        evec_label = evec_range[i] + 1
        # plot distribution of node target values
        plt.hist(target_values)
        plt.title(f'Distribution of node target values (N={args.N_PTS_ON_MANIFOLD})')
        plt.xlabel('Target value')
        plt.ylabel('Count')
        plt.savefig(f'{figures_save_dir}/targets_hist_{evec_label}.png')
        plt.clf()
        
        # plot 3-d ellipsoid points, colored by target values
        plt.rcParams["figure.figsize"] = (8., 8.)
        title = fr"Ellipsoid point target values in 3-d space (N={args.N_PTS_ON_MANIFOLD})"
        if 'single' in clargs.type.lower():
            title += fr": $\lambda_{{{evec_label}}}$ (indexing from 0)"
        plotting.plot_3d(
            title=title,
            x=manifold_xyz[:, 0],
            y=manifold_xyz[:, 1],
            z=manifold_xyz[:, 2],
            c=target_values
        )
        plt.savefig(f'{figures_save_dir}/ellipsoid_node_targets_{evec_label}.png')
        plt.clf()
        
        plt.close('all')

    
    """
    Compute normalized-evaluated observed manifold sample
    values ('Pnfs') and the spectral filters.
    """
    Pnfs = ms.get_manifold_coords_as_fn_vals(
        manifold_pts,
        args.COORD_FN_AXES,
        norm_evaluate=True
    )
    
    # Wjs_spectral = w.spectral_wavelets(
    #     eigenvals=eigenvals_obs[1:], 
    #     J=args.J,
    #     include_low_pass=args.INCLUDE_LOWPASS_WAVELET
    # ) # shape (J+2, k) with lowpass

    # # non-wavelet low-pass spectral filter (for MCN-spectral models)
    # lowpass_filters = w.spectral_lowpass_filter(
    #     eigenvals=eigenvals_obs[1:],
    #     c=None
    # )
    
    """
    The 'raw' data is a list of length 1 (to work with
    existing methods) containing a dictionary
    """
    data_dictl = [None]
    data_dictl[0] = {
        'Pnfs': Pnfs,
        'target': target_values,
        'W': graph.W, # needed for 'edge_index' in PyG Data objects
        # 'Wjs_spectral': Wjs_spectral,
        'L_eigenvals': eigenvals_obs[1:(args.KAPPA + 2)],
        'L_eigenvecs': eigenvecs_obs[:, 1:(args.KAPPA + 2)],
        # 'spectral_lowpass_filters': lowpass_filters
    }
    
    """
    Save dataset.
    """
    # save 'raw' data
    filename = save_filenames[i]
    save_path = f"{datasets_save_dir}/{filename}"

    # [optional] save raw data
    # save_path = f'{args.DATA_DIR}/{filename}.pkl'
    # with open(save_path, "wb") as f:
    #     pickle.dump(data_dictl, f, protocol=pickle.HIGHEST_PROTOCOL)  
    # print(f'Data saved as \'{filename}\'.')

    # save as pytorch geometric dataset
    dc.pickle_as_pytorch_geo_Data(
        args,
        data_dictl,
        save_path
    )

"""
Print script execution time
"""
t_overall = time.time() - t_0
t_min, t_sec = u.get_time_min_sec(t_overall)
print(
    f'\n{n_datasets} {clargs.type} ellipsoid datasets created in:'
    f' {t_min:.0f}min, {t_sec:.4f}sec.'
)

