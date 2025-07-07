"""
Script to run cross-validation for (P and/or
spectral wavelet) MFCN or GNN baseline models, 
on the melanoma or ellipsoids tasks. Pickles 
final metrics records and train timing dictionary.

Important: set args in 'melanoma.args' or
'ellipsoids_node_regress.args' and run with, 
e.g.:

python3.11 experiments_cv.py \
--machine borah \
--mfcn_p \
--smart_p_wavelets \
--dataset melanoma \
--n_folds 10 \
--n_epochs 10000 \
--burn_in 100 \
--patience 50 \
--learn_rate 0.005 \
--batch_size 8 \
--verbosity 0

python3.11 experiments_cv.py \
--machine borah \
--mfcn_p \
--smart_p_wavelets \
--gcn \
--gat \
--sage \
--gin \
--dataset ellipsoids-combination \
--multi_dataset_dir "../data/ellipsoids_node/ellipsoids_node_1_1024_knn-auto_evecs1-20_combinations_ambient8" \
--use_args_excl_dataset_indices \
--n_folds 5 \
--n_epochs 10000 \
--burn_in 100 \
--patience 50 \
--learn_rate 0.01 \
--batch_size 1024 \
--verbosity 0
"""
import sys
sys.path.insert(0, '../')
import os
import time
import pickle
import argparse

import torch
import numpy as np
import pandas as pd

import MFCN.mfcn as mfcn
import baselines.gnn as gnn
import utilities as u
import data_utilities as du
import cv
# import train_fn
# import optim


"""
clargs

! Note: ensure 'spectral' or 'p' is the last substring when 
model bool clargs are split by '_'. If an alternative to
'mcn_spectral' is introduced, a more sophisticated
tagging scheme will need to be coded where MFCN model kwargs
are assigned and parsed by downstream functions / spectral filters 
are precomputed in `dataset_creation.py`.
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine', default='desktop', type=str, 
                    help='key for the machine in use (default: desktop): see args_template.py')
# MFCNs with wavelet filters
parser.add_argument('--mfcn_p', default=False, action='store_true')
parser.add_argument('--smart_p_wavelets', default=False, action='store_true')
parser.add_argument('--mfcn_spectral', default=False, action='store_true')

# MCNs with simple, non-wavelet filter
parser.add_argument('--mcn_p', default=False, action='store_true')
parser.add_argument('--mcn_spectral', default=False, action='store_true')

# Baseline GNNs
parser.add_argument('--gcn', default=False, action='store_true')
parser.add_argument('--sage', default=False, action='store_true')
parser.add_argument('--gin', default=False, action='store_true')
parser.add_argument('--gat', default=False, action='store_true')

# MLP baseline (using MFCN_Module with no filtering/combination)
parser.add_argument('--mlp', default=False, action='store_true')

# MCN/MFCN channel output parameters
parser.add_argument('--mcn_within_filter_chan_out', type=str, default=None,
                    help='Comma-separated list of integers for MCN within-filter channel outputs (e.g., "32,16")')
parser.add_argument('--mfcn_within_filter_chan_out', type=str, default=None,
                    help='Comma-separated list of integers for MFCN within-filter channel outputs (e.g., "8,4")')
parser.add_argument('--mfcn_cross_filter_combos_out', type=str, default=None,
                    help='Comma-separated list of integers for MFCN cross-filter combinations (e.g., "8,4")')

parser.add_argument('-d', '--dataset', default='melanoma', type=str, 
                    help='dataset key: melanoma or ellipsoid')
parser.add_argument('-u', '--multi_dataset_dir', default=None, type=str, 
                    help='dataset directory (default: None)')
parser.add_argument('--use_args_excl_dataset_indices', default=False, action='store_true')

parser.add_argument('-i', '--max_n_datasets', default=None, type=int, 
                    help='for testing: max number of multi-datasets (in --multi_dataset_dir) to actually use (default: None, to use all)')

parser.add_argument('-f', '--n_folds', default='10', type=int, 
                    help='number of cross-validation folds to run (default: 10)')
parser.add_argument('-e', '--n_epochs', default='10000', type=int, 
                    help='max num. of epochs to run in each fold (default: 10000)')
parser.add_argument('-b', '--burn_in', default='4', type=int, 
                    help='min num. of epochs to run before enforcing early stopping (default: 4)')
parser.add_argument('-p', '--patience', default='32', type=int, 
                    help='if args.STOP_RULE is no_improvement, max num. of epochs'
                    'without improvement in validation loss (default: 32)')
parser.add_argument('-l', '--learn_rate', default='0.003', type=float, 
                    help='learning rate hyperparameter (default: 0.003)')
parser.add_argument('-t', '--batch_size', default='32', type=int, 
                    help='train set minibatch size hyperparameter (default: 32)')
parser.add_argument('-v', '--verbosity', default='0', type=int, 
                    help='integer controlling volume of print output during execution')
parser.add_argument('--spectral_c', default=None, type=float, 
                    help='Spectral coefficient for the model (default: None)')
clargs = parser.parse_args()


"""
args
"""
# interpret clargs
dataset_key = clargs.dataset.lower() # 'ellipsoids-combination', 'melanoma'
pyg_data_list_filepaths = [None] # placeholder
model_bools_keys = (
    (clargs.mfcn_p, 'mfcn_p'),
    (clargs.mcn_p, 'mcn_p'),
    (clargs.mfcn_spectral, 'mfcn_spectral'),
    (clargs.mcn_spectral, 'mcn_spectral'),
    (clargs.mlp, 'mlp'),
    (clargs.gcn, 'gcn'),
    (clargs.sage, 'sage'),
    (clargs.gin, 'gin'),
    (clargs.gat, 'gat'),
)
model_keys = []
for model_bool, model_key in model_bools_keys:
    if model_bool:
        model_keys.append(model_key)

# Parse MCN/MFCN channel output parameters into tuples
def parse_channel_outputs(param_str):
    if (param_str is None) or (param_str == 'None'):
        return None
    try:
        return tuple(int(x) for x in param_str.split(','))
    except ValueError:
        print(f"Error: Invalid channel output parameter format: {param_str}")
        return None

mcn_within_filter_chan_out = parse_channel_outputs(clargs.mcn_within_filter_chan_out)
mfcn_within_filter_chan_out = parse_channel_outputs(clargs.mfcn_within_filter_chan_out)
mfcn_cross_filter_combos_out = parse_channel_outputs(clargs.mfcn_cross_filter_combos_out)

# if using P-wavelets in MFCN, choose wavelet scales method
if clargs.mfcn_p and clargs.smart_p_wavelets:
    p_wavelet_scales = 'handcrafted'
else:
    p_wavelet_scales = 'dyadic'


# import correct task args file
if 'ellip' in dataset_key:
    import ellipsoids_node_regress.args as a
    # save models within 'combination' or 'single' dataset type collections
    new_subdir = dataset_key.split("-")[1]
    # model_save_subdir = f"ellipsoids_node/{new_subdir}"
elif 'melan' in dataset_key:
    import melanoma.args as a
    new_subdir = None

# init args (with command-line overrides)
args = a.Args(
    MACHINE=clargs.machine,
    MODEL_NAME='model', # placeholder; renamed in models loop
    # MODEL_SAVE_SUBDIR=model_save_subdir,
    # WAVELET_TYPE=model_keys[0],
    P_WAVELET_SCALES=p_wavelet_scales,
    N_FOLDS=clargs.n_folds,
    N_EPOCHS=clargs.n_epochs,
    BURNIN_N_EPOCHS=clargs.burn_in,
    NO_VALID_LOSS_IMPROVE_PATIENCE=clargs.patience,
    LEARN_RATE=clargs.learn_rate,
    # >0 prints epoch-by-epoch stats in train_fn
    VERBOSITY=clargs.verbosity,
    # MCN/MFCN channel output parameters
    MCN_WITHIN_FILTER_CHAN_OUT=mcn_within_filter_chan_out,
    MFCN_WITHIN_FILTER_CHAN_OUT=mfcn_within_filter_chan_out,
    MFCN_CROSS_FILTER_COMBOS_OUT=mfcn_cross_filter_combos_out,
    SPECTRAL_C=clargs.spectral_c,
)
args.set_model_save_dirs(new_subdir=new_subdir)
args.set_batch_sizes(train_size=clargs.batch_size)

if 'ellip' in dataset_key:
    # import MFCN.mfcn_ellipsoid_args as a
    num_nodes_one_graph = args.N_PTS_ON_MANIFOLD
    orig_subdir_name = args.MODEL_SAVE_SUBDIR

    # note 'coefs.pkl' is the default value of 'coef_save_picklename'
    # in ms.rand_bandlimit_evec_signal (which creates ellipsoid-combination
    # signals)
    if clargs.multi_dataset_dir is not None:
        pyg_data_list_filepaths = [
            f"{clargs.multi_dataset_dir}/{f}" \
            for f in os.listdir(clargs.multi_dataset_dir) \
            if 'coefs.pkl' not in f
        ]

    # if excluding some datasets in a multi-datasets dir, remove from
    # dataset filepaths list to loop through here
    if clargs.use_args_excl_dataset_indices \
    and (args.EXCLUDE_DATASET_INDICES is not None):
        pyg_data_list_filepaths = [
            # if statement logic: rm dirs from filename, remove '.pkl', grab integer after '_'
            path for path in pyg_data_list_filepaths \
            if int(path.split("/")[-1].split(".")[0].split("_")[1]) \
            not in args.EXCLUDE_DATASET_INDICES
        ]
                
        # pyg_data_list_filepaths = [
        #     pyg_data_list_filepaths[i] \
        #     for i, path in enumerate(pyg_data_list_filepaths) \
        #     if i not in args.EXCLUDE_DATASET_INDICES
        # ]
        
           
elif 'melan' in dataset_key:
    # import MFCN.mfcn_melanoma_args as a
    num_nodes_one_graph = None
    model_save_subdir = args.MODEL_SAVE_SUBDIR
    pyg_data_list_filepaths = [None]

"""
[optional]: only run on a subset of a multi-dataset 
"""
if clargs.max_n_datasets is not None:
    pyg_data_list_filepaths = pyg_data_list_filepaths[:(clargs.max_n_datasets)]
    

"""
cv splits
- for the melanoma dataset, cv.get_cv_idxs_for_dataset stratify-samples
the folds using the raw data_dictl pickled during dataset creation
"""
cv_idxs_unexpanded = cv.get_cv_idxs_for_dataset(args, dataset_key)


"""
training objects' kwargs
"""
base_module_kwargs = {
    'task': args.TASK,
    'target_name': args.TARGET_NAME,
    'device': args.DEVICE
}

if 'ellip' in dataset_key:
    # node-level regression task: no readout/head (final node-pooling layer only)
    fc_model_kwargs = None
    gnn_out_channels = 1
    
elif 'melan' in dataset_key:
    gnn_out_channels = None 
    # melanoma graph-level classification task needs readout and classifier head
    fc_model_kwargs = {
        'base_module_kwargs': base_module_kwargs,
        'output_dim': args.OUTPUT_DIM,
        'hidden_dims_list': args.NN_HIDDEN_DIMS,
        'use_batch_normalization': args.USE_BATCH_NORMALIZATION,
        'use_dropout': args.MLP_USE_DROPOUT,
        'dropout_p': args.MLP_DROPOUT_P
    }


"""
run k-fold cross validation, for models x datasets
"""
# since results are saved-overwritten after each model, 
# fix the save file prefix
results_file_prefix = "-".join([k for k in model_keys])
if new_subdir is not None:
    save_dir = f'{args.RESULTS_SAVE_DIR}/{new_subdir}/{results_file_prefix}_{args.ts}'
else:
    save_dir = f'{args.RESULTS_SAVE_DIR}/{results_file_prefix}_{args.ts}'
print(save_dir)
os.makedirs(save_dir, exist_ok=True)


# loop models
for i, model_key in enumerate(model_keys):

    # init model's results container dicts
    # for each model, we save metrics and timing results containers
    model_metrics_records = []
    model_timing_dict = {}
    args.MODEL_NAME = model_key
    model_timing_dict[model_key] = []

    # set model's save filenames for results containers
    save_filenames = {}
    for result_key in ('metrics', 'times'):
        if len(model_keys) > 1:
            save_filenames[result_key] = f'cv_{result_key}_{model_key}.pkl'
        else:
            save_filenames[result_key] = f'cv_{result_key}.pkl'

    # print 'new model starting CVs' message
    print('-' * 50)
    print(
        f"Starting {args.N_FOLDS}-fold CVs of {model_key} on"
        f" {len(pyg_data_list_filepaths)} \'{dataset_key}\'\ndatasets")
    print(f"\tstarted {time.ctime()}")
    print('-' * 50)

    # reset 'model_kwargs' for each new model
    model_kwargs = {
        'base_module_kwargs': base_module_kwargs,
        'fc_kwargs': fc_model_kwargs,
        'verbosity': args.VERBOSITY
    }

    if model_key == 'mlp':
        model_class = mfcn.MFCN_Module
        model_kwargs.update({
            'wavelet_type': None,
            'non_wavelet_filter_type': None,
            'n_channels': max(args.MANIFOLD_N_AXES, args.AMBIENT_DIM),
            'num_nodes_one_graph': num_nodes_one_graph,
            'channel_pool_key': args.MFCN_FINAL_CHANNEL_POOLING,
            'node_pooling_key': args.MFCN_FINAL_NODE_POOLING,
            'node_pool_linear_out_channels': gnn_out_channels,
            'within_Wj_ns_chan_out_per_filter': args.MFCN_WITHIN_FILTER_CHAN_OUT,
            'cross_Wj_ns_combos_out_per_chan': None
        })
    elif ('mfcn' in model_key) or ('mcn' in model_key):
        model_class = mfcn.MFCN_Module
        filter_type = model_key.split("_")[-1] # 'p', 'spectral', or 'none'
        # MCN models have no wavelets
        args.WAVELET_TYPE = filter_type if ('mfcn' in model_key) else None
        args.NON_WAVELET_FILTER_TYPE = filter_type if ('mcn' in model_key) else None
        # MCN models have no cross-filter combinations
        cross_Wj_ns_combos_out_per_chan = None \
            if ('mcn' in model_key) else args.MFCN_CROSS_FILTER_COMBOS_OUT
        # set cross-channel combinations depending on 'mcn' vs. 'mfcn' model
        within_Wj_ns_chan_out_per_filter = args.MCN_WITHIN_FILTER_CHAN_OUT \
            if ('mcn' in model_key) else args.MFCN_WITHIN_FILTER_CHAN_OUT
        
        model_kwargs['wavelet_type'] = args.WAVELET_TYPE
        model_kwargs['non_wavelet_filter_type'] = args.NON_WAVELET_FILTER_TYPE
        model_kwargs['filter_c'] = args.SPECTRAL_C
        model_kwargs['n_channels'] = max(args.MANIFOLD_N_AXES, args.AMBIENT_DIM)
        model_kwargs['num_nodes_one_graph'] = num_nodes_one_graph
        model_kwargs['J'] = args.J
        model_kwargs['include_lowpass_wavelet'] = args.INCLUDE_LOWPASS_WAVELET
        model_kwargs['within_Wj_ns_chan_out_per_filter'] = within_Wj_ns_chan_out_per_filter
        model_kwargs['cross_Wj_ns_combos_out_per_chan'] = cross_Wj_ns_combos_out_per_chan
        model_kwargs['max_kappa'] = args.MFCN_MAX_KAPPA
        model_kwargs['channel_pool_key'] = args.MFCN_FINAL_CHANNEL_POOLING
        model_kwargs['node_pooling_key'] = args.MFCN_FINAL_NODE_POOLING
        model_kwargs['node_pool_linear_out_channels'] = gnn_out_channels
        
    else: # GCN, GAT, GIN, GraphSAGE
        model_class = gnn.GNN_FC
        model_kwargs['gnn_type'] = model_key
        model_kwargs['in_channels'] = -1 # -1 is 'auto'; inferred from first batch
        # if 'out_channels' is not None, pytorch geometric GNNs will apply
        # a final linear layer to convert hidden embeddings to size 'out_channels'
        model_kwargs['out_channels'] = gnn_out_channels
        model_kwargs['channel_pool_key'] = args.GNN_FINAL_CHANNEL_POOLING
        model_kwargs['dropout_p'] = args.GNN_DROPOUT_P if args.GNN_USE_DROPOUT else 0.
        
    args.set_model_name_timestamp(new_timestamp=False)
    

    # loop datasets
    for j, pyg_data_list_filepath in enumerate(pyg_data_list_filepaths):

        print('-' * 50)
        print(
            f'Running {args.N_FOLDS}-fold CV of {model_key} on dataset'
            f' {j + 1} of {len(pyg_data_list_filepaths)}')
        print(f'\tstarted {time.ctime()}')
        print('-' * 50)

        if pyg_data_list_filepath is not None:
            # create dataset identifier, e.g., "comb_0",
            # under 'dataset' key; make new subsubdir for
            # cv runs of each dataset
            dataset_id = pyg_data_list_filepath \
                .split("/")[-1] \
                .split(".")[0] # remove '.pkl'
        else: # 'None' for melanoma
            dataset_id = dataset_key

        # update train_history filename prefix for model-dataset combo
        args.set_model_save_dirs(train_hist_prefix=dataset_id)
        
        # run k-fold cross validation of one MFCN model
        metrics_records, epoch_times_l = cv.run_cv(
            args,
            dataset_id,
            n_folds=args.N_FOLDS,
            model_name=args.MODEL_NAME,
            model_class=model_class,
            model_kwargs=model_kwargs,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                'lr': args.LEARN_RATE,
                'betas': args.ADAM_BETAS,
                'weight_decay': args.ADAM_WEIGHT_DECAY
            },
            cv_idxs=cv_idxs_unexpanded,
            n_oversamples=args.N_OVERSAMPLES,
            using_pytorch_geo=True,
            pyg_data_list_filepath=pyg_data_list_filepath,
            verbosity=args.VERBOSITY,
        )
        
        # save model's metrics and runtimes in all-model containers
        model_metrics_records += metrics_records
        model_timing_dict[args.MODEL_NAME] += epoch_times_l

        """
        pickle all-model metrics and times containers
        - overwrite saved files after each dataset completes all its folds
        (we hence overwrite model's results files 'n_datasets' times)
        """
        for result_key, obj \
        in (('metrics', model_metrics_records), ('times', model_timing_dict)):
            result_filename = save_filenames[result_key]
            full_save_path = f'{save_dir}/{result_filename}'
            u.pickle_obj(full_save_path, obj, overwrite=True)
            print(f'{model_key} {result_key} CV results saved (overwrite {j}) in\n\'{full_save_path}\'.')


