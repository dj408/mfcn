"""
Script: run manifold sampling experiment.

Set experiment configuration in 'config.py'.
"""

import sys
sys.path.insert(0, './code')
sys.path.insert(0, '../')
import time
import pickle

import config as cf
import manifold_sampling as ms
import graph_construction as gc
import wavelets as w
import utilities as u

import numpy as np
from math import log10, ceil, floor
import scipy.sparse
from scipy.sparse import coo_array
from scipy.sparse.linalg import eigsh

"""
loop through SAMPLING_CASES x Ns
"""
for case_i, (case, density_ratio) in enumerate(cf.SAMPLING_CASES):
    print_banner = '-' * 50
    print(f'{print_banner}\nSampling case: \'{case}\''
          f'(density ratio = {density_ratio})\n{print_banner}\n')

    """
    loop through multiple runs
    """

    # new list container for each case's runs
    case_runs_l = [None] * cf.N_RUNS

    for run_i in range(cf.N_RUNS):
        print_divider = '-' * 17
        print(f'{print_divider}Starting run {run_i + 1}{print_divider}')
        prev_t_elapsed = None
        rel_t_elapsed = None
        seed = cf.SAMPLING_SEEDS[run_i]
        print(f'sampling seed = {seed}')
        sampling_rs = np.random.RandomState(seed=seed)

        # new list container for this run (of one case)
        run_records = [None] * len(cf.Ns)

        # for each run, loop through increasing Ns
        for n_i, n in enumerate(cf.Ns):
            print(f'n = {n}')
            time_n_i_start = time.time()

            """
            generate sample points on a manifold; define function on the manifold
            - both uniform and nonuniform samples start from large 
              uniform sample 'x_pool'
            """
            if cf.MANIFOLD_TYPE == 'circle':
                
                thetas = ms.unif_circle_thetas(
                    n=cf.INITIAL_SAMPLE_N, 
                    random_state=sampling_rs
                )
                manifold_pool = thetas
                
                # convert to xy euclidean coords
                x_pool = ms.convert_complex_to_R2(
                    np.exp(1j * thetas) * cf.RADIUS
                )
                
                # set the axis of bias in nonuniform sampling: y/imaginary axis
                nonunif_v = x_pool[:, 1]
                
            elif cf.MANIFOLD_TYPE == 'sphere':
                
                sph_theta, sph_psi = ms.unif_sphere_thetas_psis(
                    n=cf.INITIAL_SAMPLE_N,
                    random_state=sampling_rs
                )
                manifold_pool = np.stack((sph_theta, sph_psi), axis=-1)
                
                # convert to xyz euclidean coords and stack
                sph_x, sph_y, sph_z = ms.convert_sph_coords_to_eucl(
                    cf.RADIUS,
                    sph_theta,
                    sph_psi
                )
                x_pool = np.stack((sph_x, sph_y, sph_z), axis=1)
                
                # set the axis of bias in nonuniform sampling
                nonunif_v = sph_z
            
            elif cf.MANIFOLD_TYPE == 'random_values':
                # or, choose random values for x
                x_pool = sampling_rs.rand(cf.INITIAL_SAMPLE_N)
        
            """
            uniform vs. nonuniform sampling densities
            """
            if case == 'uniform':
                # take first n random points on manifold for x
                # sample_idx = np.arange(n)
                # or: be 'doubly random' and take any n random points
                sample_idx = sampling_rs.choice(
                    np.arange(x_pool.shape[0]),
                    n
                )
                emp_density_ratio = 1.
        
            elif case == 'nonuniform':
                nonunif_alpha = ms.get_linear_alpha_from_density_ratio(density_ratio)
                # sample manifold points to be denser towards the z poles
                p = ms.get_polar_nonunif_p(
                    v=nonunif_v, 
                    mode=cf.NONUNIF_MODE, 
                    alpha=nonunif_alpha,
                    squared=cf.NONUNIF_SQUARED
                )
                emp_density_ratio = np.max(p) / np.min(p)
                sample_idx = ms.get_nonunif_sample_idx(
                    n=n,
                    p=p,
                    random_state=sampling_rs
                )
                
            else:
                print('Error: sampling case not recognized! Exiting.')
                break
        
            # finally, sample points on manifold from x_pool
            x_intrinsic = manifold_pool[sample_idx]
            x = x_pool[sample_idx]
        
            print(f'empirical density ratio: {emp_density_ratio:.6f}')
        
            """
            graph construction and Laplacian
            """
            t_graph_start = time.time()
            
            if cf.GRAPH_TYPE == 'epsilon':
                graph = gc.EpsilonGraph(
                    x=x,
                    n=n,
                    eps=cf.EPSILON, 
                    eps_constant=cf.EPS_CONSTANT,
                    d_manifold=cf.D_MANIFOLD,
                    eta_type=cf.ETA_TYPE,
                    verbosity=cf.VERBOSITY
                )
                graph_params = {
                    'type': 'epsilon',
                    'epsilon': graph.eps
                }
                
            elif cf.GRAPH_TYPE == 'knn':
                graph = gc.KNNGraph(
                    x=x,
                    n=n,
                    d_manifold=cf.D_MANIFOLD,
                    k=cf.KNN_K,
                    k_constant=cf.K_CONSTANT,
                    knn_algorithm='ball_tree',
                    knn_logic='or',
                    eta_type=cf.ETA_TYPE,
                    verbosity=cf.VERBOSITY
                )
                graph_params = {
                    'type': 'knn',
                    'k': graph.k
                }
        
            t_graph_creation = time.time() - t_graph_start
            t_min, t_sec = u.get_time_min_sec(t_graph_creation)
            print(f'Time-graph creation: {t_min:.0f}min, {t_sec:.4f}sec.')
        
            t_Lapl_start = time.time()
            graph.calc_Laplacian()
        
            t_Lapl_calc = time.time() - t_Lapl_start
            t_min, t_sec = u.get_time_min_sec(t_Lapl_calc)
            print(f'Time-Laplacian calc: {t_min:.0f}min, {t_sec:.4f}sec.')
        
            """
            eigendecomposition of graph Laplacian
            - L_n is a real symmetric square matrix -> use scipy.sparse.eigsh
            - if L_n was square but NOT symmetric -> scipy.sparse.eigs
            """
            t_eigendecomp_start = time.time()
            
            # sparse-eigendecompose Ln, up to k smallest eigenpairs, k < n
            # eigenvectors array has shape (n, k) [cols are e-vecs]
            kappa = cf.KAPPAS[n_i]
            if kappa >= n:
                kappa = n - 1
            eigenvalues, eigenvectors = eigsh(graph.L, k=kappa, which='SM')
            if eigenvalues[0] > cf.ABS_EQUIV_THRESH:
                print(
                    f'Warning: first eigenvalue of graph Laplacian != 0.'
                    f' ({eigenvalues[0]:.4E})'
                )
            eigenvalues[0] = 0.
        
            t_eigendecomp = time.time() - t_eigendecomp_start
            t_min, t_sec = u.get_time_min_sec(t_eigendecomp)
            print(f'Time-eigendecomposition: {t_min:.0f}min, {t_sec:.4f}sec.')
            
            """
            sample a function on the manifold
            """
            if cf.MANIFOLD_TYPE == 'circle':
                f_vals = ms.apply_circ_harm(thetas[sample_idx])
                
            elif cf.MANIFOLD_TYPE == 'sphere':
                f_vals = ms.apply_sph_harm_lc(
                    cf.sph_harm_params,
                    sph_theta[sample_idx],
                    sph_psi[sample_idx]
                )
                
            elif cf.MANIFOLD_TYPE == 'random_values':
                f_vals = x_pool[sample_idx]
        
        
            """
            discrete approximation of manifold convolutions
            - a filter (as function of the eigenvalues) on the 
            Fourier domain
            """
            # 0. wavelet filters and scattering q-moments
            Pnf = ms.norm_eval_op(f_vals, n)
            wavelet_filters = w.spectral_wavelets(eigenvalues, cf.J)
            Wjf_0, Wjf_1, Wjf_2 = w.wavelet_spectral_convolutions(
                Pnf=Pnf,
                eigenvals=eigenvalues,
                wavelet_filters=wavelet_filters,
                J=cf.J,
                eigenvectors=eigenvectors,
                verbosity=cf.VERBOSITY
            )
        
            # store scattering q-moments in dict
            # shapes: 0th (Q, ); 1st (Q, J+1); 2nd (Q, J(J+1)/2)
            scat_moments = {
                i: w.scattering_moments(Wjf, cf.Q) \
                for i, Wjf in enumerate((Wjf_0, Wjf_1, Wjf_2))
            }
        
            # 1. lowpass (e.g., e^-t) spectral convolutional operator
            fourier_coeffs_Pnf = np.einsum(
                'n,nk->k', 
                Pnf, 
                eigenvectors
            )
            low_pass_convolution = np.einsum(
                'k,k,nk->n',
                # np.exp(-eigenvalues), 
                cf.lowpass_spectral_filter(eigenvalues),
                fourier_coeffs_Pnf, 
                eigenvectors
            )
    
            """
            unit tests
            """
            if cf.RUN_UNIT_TESTS:
                print('Running unit tests...')
                t_unit_tests_start = time.time()
                
                # check shapes
                # print(f'eigenvectors shape: {eigenvectors.shape}')
                # print(f'fourier_coeffs shape: {fourier_coeffs.shape}')
                
                # unit test: check that eigenvalues are as expected for sph.
                # harmonics
                
                # unit test: check that eigenvectors have (squared) l2-norm 1
                for i, evec in enumerate(eigenvectors.T):
                    sq_norm = np.dot(evec, evec)
                    if np.abs(sq_norm - 1.) > cf.ABS_EQUIV_THRESH:
                        print(f'\tWarning: eigenvector {i+1} did not have norm 1!')
                        break
        
                # unit test: check that eigenvectors are orthogonal
                from itertools import combinations
                all_evect_idx_pairs = list(combinations(
                    range(eigenvectors.shape[1]), 2
                ))
                for (i, j) in  all_evect_idx_pairs:
                    dp = np.dot(eigenvectors[:, i], eigenvectors[:, j])
                    if np.abs(dp) > cf.ABS_EQUIV_THRESH:
                        print(
                            f'\tWarning: eigenvectors {i+1}, {j+1} are not orthogonal!'
                            f' (dot prod = {dp:.4E})'
                        )
                        break
                        
                print('\tdone.')
        
                '''
                
                # unit test: the sum of Fourier coeffs times their corresponding
                # eigenvectors should recreate x (to 0.95 tolerance)
                fc_times_evects = np.einsum(
                    'k,nk->n', 
                    fourier_coeffs_Pnf, 
                    eigenvectors
                )
                if not np.allclose(fc_times_evects, Pnf, rtol=cf.REL_EQUIV_THRESH):
                    print(f'\tWarning: Fourier coeffs times eigenvectors did not'
                          f' recreate Pnf(x)!')
                
                # unit test: sum of squared Fourier coeffs should approximate
                # the squared norm of f(x)
                sum_sq_fourier_coeffs = np.sum(np.square(fourier_coeffs_Pnf))
                sq_norm_Pnfx = np.dot(Pnfx, Pnfx)
                if (sum_sq_fourier_coeffs < (cf.REL_EQUIV_THRESH * sq_norm_Pnfx)) \
                    or (sum_sq_fourier_coeffs > sq_norm_Pnfx):
                    mult = sq_norm_Pnfx / sum_sq_fourier_coeffs
                    print(
                        f'\tWarning: Sum of squared Fourier coeffs '
                        f'({sum_sq_fourier_coeffs:.4E}) != ||f(x)||^2 ({sq_norm_Pnfx:.4E}).'
                        f'\n\tMultiple of difference: {mult:.4E}'
                        f'\n\t(However, this is expected if too few eigenvectors are used!)'
                    )
                '''
                t_unit_tests = time.time() - t_unit_tests_start
                t_min, t_sec = u.get_time_min_sec(t_unit_tests)
                print(f'Time-unit tests: {t_min:.0f}min, {t_sec:.4f}sec.')
    
            
            """
            timing stats
            """
            t = time.time() - time_n_i_start
            if prev_t_elapsed is not None:
                rel_t_elapsed = t / prev_t_elapsed
            prev_t_elapsed = t
            t_min, t_sec = u.get_time_min_sec(t)
            time_str = f'Time-total: {t_min:.0f}min, {t_sec:.1f}sec.'
            if rel_t_elapsed is not None:
                time_str += f'\n\t({rel_t_elapsed:.2f}x slower than last n)'
                print(time_str)
            print()
    
            
            """
            save results
            """
            n_results_dict = {}
            n_results_dict['run_i'] = run_i
            n_results_dict['manifold_type'] = cf.MANIFOLD_TYPE
            n_results_dict['sampling'] = case
            # n_results_dict['case'] = case
            n_results_dict['n'] = n
            # n_results_dict['x'] = x
            n_results_dict['x_intrinsic'] = x_intrinsic
            n_results_dict['density_ratio'] = density_ratio
            n_results_dict['empirical_density_ratio'] = emp_density_ratio
            n_results_dict['graph_params'] = graph_params
            n_results_dict['nonzero_prop'] = graph.nonzero_prop
            
            n_results_dict['eigenvalues'] = eigenvalues[:cf.N_EIGENPAIRS_SAVE]
            n_results_dict['eigenvectors'] = eigenvectors[:, :cf.N_EIGENPAIRS_SAVE] 
            # n_results_dict['f_vals'] = f_vals
            n_results_dict['fourier_coeffs_Pnf'] = fourier_coeffs_Pnf
            n_results_dict['low_pass_convolution'] = low_pass_convolution
            n_results_dict['wavelet_convolutions'] = Wjf_0, Wjf_1, Wjf_2
            n_results_dict['scattering_moments'] = scat_moments
            
            n_results_dict['time_graph_creation'] = t_graph_creation
            n_results_dict['time_Laplacian_calc'] = t_Lapl_calc
            n_results_dict['time_eigendecomp'] = t_eigendecomp
            n_results_dict['time_total'] = t

            # insert case's run's N's results dict into case's records list
            run_records[n_i] = n_results_dict

        # insert run's records list into case's list
        case_runs_l[run_i] = run_records
        
    
        """
        pickle records
        - one file per sampling case, storing multiple Ns
        NOTE: un-indent this one tab and save 'records_l' to save all cases'
        runs in one master list
        """
        obj_to_save = case_runs_l
        filename = cf.get_exp_results_filename()
        save_path = f'{cf.SAVE_DIR}/{filename}'
        with open(save_path, "wb") as f:
            pickle.dump(obj_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Case records saved (\'{case}\' case).\n')

print('Done.')

