"""
Function to create a dataset for
MFCN / wavelet filtration of manifolds.
"""
import sys
sys.path.insert(0, '../')

import manifold_sampling as ms
import graph_construction as gc
import wavelets as w
import utilities as u
import data_utilities as du

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.base import spmatrix
from scipy.sparse.linalg import eigsh
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Any
)
import torch
from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset
)
from torch_geometric.loader import DataLoader


def create_manifold_wavelet_dataset(
    args,
    manifolds_dictl: List[dict],
    pickle_dictl_out: bool = True
) -> None | List[Dict[str, Any]]: # pickles or returns
    """
    
    Args:
        args: an ArgsTemplate object.
        manifolds_dictl: list of dictionaries holding
            keyed objects for each manifold in a 
            manifold dataset (e.g., features tensor,
            target value, etc.).
        pickle_dictl_out: bool whether to pickle the
             output list of dictionary, or else return
             it.
    Returns:
        Either None (and pickles the output),
        or returns a list of dictionaries, each
        dictionary being one data sample's record,
        containing its id, target, channel signals,
        etc.
    """

    if args.COORD_FN_AXES == 'all':
        coord_fn_axes = tuple(range(args.MANIFOLD_N_AXES))
    
    # if args.SAVE_SPECTRAL_OBJS:
    #     args.RAW_DATA_FILENAME += '_spectral'
    # if args.SAVE_P_OBJS:
    #     args.RAW_DATA_FILENAME += '_P'
    # if args.SAVE_SCAT_MOMENTS_OBJS:
    #     args.RAW_DATA_FILENAME += '_SMs'
    save_filename = args.RAW_DATA_FILENAME + '.pkl'

    """
    time overall dataset generation
    """
    print('Generating manifold dataset...')
    t_0 = time.time()
    
    
    """
    init empty list of dicts to hold manifolds'
    parameters, targets, and scattering moments
    
    final structure:
    'key': value [e.g. training targets for the manifold]
    'scattering_moments':
        |-> wavelet type: 'spectral' | 'P'
            |-> scattering moment order: 0 | 1 | 2
                |-> arrays of scattering moment values:
                    rows = Wjs (args.J/jprime...args.J); 
                    cols = qth-order norms (1...args.Q)
    """
    # init empty lists to populate
    out_dictl = [None] * args.N_MANIFOLDS
    Pnfs = [None] * args.N_MANIFOLDS
    graph_Laplacians = [None] * args.N_MANIFOLDS
    # graph_Ps = [None] * args.N_MANIFOLDS
    
    """
    function-on-manifold values
    """
    for i, manifold_dict in enumerate(manifolds_dictl):
        manifold = manifold_dict[args.MANIFOLD_KEY]
        out_dictl[i] = {}
        
        # use normalize-evaluated 'coordinate values' as 
        # function-on-manifold values, or vice versa...
        # (conflates geometry and signal...?)
        Pnfs[i] = ms.get_manifold_coords_as_fn_vals(
            manifold,
            args.COORD_FN_AXES,
            norm_evaluate=True
        )
        if i == 0:
            print('Pnfs[0].shape:', Pnfs[i].shape)
    
        # construct graph
        # 'W' is the sparse, eta-kernelized weighted adjacency matrix
        # (but where eta = indicator (default), weights are all 1 or 0)
        if args.GRAPH_TYPE == 'knn':
            graph = gc.KNNGraph(
                x=manifold,
                n=manifold.shape[0], # args.N_PTS_ON_MANIFOLD
                k=args.K_OR_EPS, # 'auto',
                d_manifold=args.D_MANIFOLD,
                eta_type='indicator'
            )
        elif args.GRAPH_TYPE == 'epsilon':
            graph = gc.EpsilonGraph(
                x=manifold,
                n=manifold.shape[0], # args.N_PTS_ON_MANIFOLD 
                eps=args.K_OR_EPS, # 'auto'
                d_manifold=args.D_MANIFOLD,
                eta_type='indicator'
            )

        # manifold coords array no longer needed -> delete
        del manifold_dict[args.MANIFOLD_KEY]

        if args.SAVE_SPECTRAL_OBJS:
            # compute Laplacians and Ps (lazy random walk matrices)
            # save graph Laplacians (for spectral wavelets) and LRWMs to list
            graph.calc_Laplacian()
            # print(type(graph.L))
            graph_Laplacians[i] = graph.L

        # if args.SAVE_P_OBJS:
        if args.WAVELET_TYPE == 'P':
            # Ws are already calculated when calculating Laplacians
            if not args.SAVE_SPECTRAL_OBJS:
                graph.calc_W()
            P = gc.calc_LRWM(graph.W, normalize=False)
            # print(P.shape)
            # graph_Ps[i] = P
            out_dictl[i]['P'] = P

        # save manifold objects in 'out_dictl'
        out_dictl[i] = out_dictl[i] | manifold_dict
        if args.SAVE_GRAPH:
            out_dictl[i]['W'] = graph.W
        if args.SAVE_FN_VALS:
            out_dictl[i]['Pnfs'] = Pnfs[i]
            
    
    """
    spectral wavelet operators and/or scattering moments
    """
    if args.SAVE_SPECTRAL_OBJS:
        print(
            f'Working on spectral decomp. of L objects...'
            f'\n\tkappa = {args.KAPPA}'
        )

        # track total time for all sparse eigendecompositions
        if args.VERBOSITY > 0:
            t_eigendecomp_start = time.time()
            
        for i, L in enumerate(graph_Laplacians):
            # track time for each sparse eigendecomposition
            if args.VERBOSITY > 1:
                print(f"eigendecomposing graph {i}")
                t_i_0 = time.time()
                
            spectral_sm_dict = {}
            # decompose kappa + 1 eigenpairs, since we discard first
            # (trivial / constant, with lambda = 0)
            eigenvals, eigenvecs = eigsh(L, k=args.KAPPA + 1, which='SM')
            # eigenvals[0] = 0.
            # print('eigenvecs.shape:', eigenvecs.shape) # shape (N, k)
            # don't include first (trivial) eigenpair
            eigenvals, eigenvecs = eigenvals[1:], eigenvecs[:, 1:]
            out_dictl[i]['L_eigenvals'] = eigenvals
            out_dictl[i]['L_eigenvecs'] = eigenvecs
            

            # if args.WAVELET_TYPE is not None:
                # Wjs_spectral = w.spectral_wavelets(
                #     eigenvals=eigenvals, 
                #     J=args.J,
                #     include_low_pass=args.INCLUDE_LOWPASS_WAVELET
                # )
                # out_dictl[i]['Wjs_spectral'] = Wjs_spectral
                
            # if args.NON_WAVELET_FILTER_TYPE == 'spectral':
            #     lowpass_filters = w.spectral_lowpass_filter(eigenvals)
            #     out_dictl[i]['spectral_lowpass_filters'] = lowpass_filters
        
            if args.SAVE_SCAT_MOMENTS_OBJS:
                # print('Calculating spectral-based scattering moments...')
                for j in coord_fn_axes:
                    spectral_axis_sm_dict = w.get_spectral_wavelets_scat_moms_dict(
                        L=L,
                        eigenvals=eigenvals,
                        eigenvecs=eigenvecs,
                        Pnf=Pnfs[i][:, j],
                        J=args.J,
                        Q=args.Q,
                        include_low_pass=args.INCLUDE_LOWPASS_WAVELET,
                        verbosity=args.VERBOSITY
                    )
                    key = f'axis_{j}'
                    spectral_sm_dict[key] = spectral_axis_sm_dict
                
                # save in manifold's existing dict
                out_dictl[i]['scattering_moments'] = {}
                out_dictl[i]['scattering_moments']['spectral'] = spectral_sm_dict
            if args.VERBOSITY > 1:
                t_i_eigendecomp = time.time() - t_i_0
                t_min, t_sec = u.get_time_min_sec(t_i_eigendecomp)
                print(f'\t{t_min:.0f}min, {t_sec:.4f}sec.')
                
        # after all graphs have been eigendecomposed
        if args.VERBOSITY > 0:
            t_eigendecomp = time.time() - t_eigendecomp_start
            t_min, t_sec = u.get_time_min_sec(t_eigendecomp)
            print(f'Data processing time ({args.N_MANIFOLDS} graphs):')
            print(f'\t{t_min:.0f}min, {t_sec:.4f}sec. total')
            print(f'\t{t_eigendecomp / args.N_MANIFOLDS:.4f} sec/graph')
    
    
    """
    lazy random walk wavelet scattering moments
    """
    if args.WAVELET_TYPE == 'P' and args.SAVE_SCAT_MOMENTS_OBJS:
        print('Calculating P-based scattering moments...')
        for i, dict in enumerate(out_dictl):
            # print(list(dict.keys()))
            P_wavelets_sm_dict = {}
            for j in coord_fn_axes:
                P_axis_sm_dict = w.get_P_wavelets_scat_moms_dict(
                    P=dict['P'],
                    Pnf=Pnfs[i][:, j],
                    J=args.J,
                    Q=args.Q,
                    include_lowpass=args.INCLUDE_LOWPASS_WAVELET,
                    verbosity=args.VERBOSITY
                )
                key = f'axis_{j}'
                P_wavelets_sm_dict[key] = P_axis_sm_dict
            
            # enter into dict
            if 'scattering_moments' not in out_dictl[i]:
                out_dictl[i]['scattering_moments'] = {}
            out_dictl[i]['scattering_moments']['P'] = P_wavelets_sm_dict

        if not args.SAVE_P_OBJS:
            del dict['P']

    if pickle_dictl_out:
        """
        pickle dataset (list of dicts)
        """
        save_path = f'{args.DATA_DIR}/{save_filename}'
        with open(save_path, "wb") as f:
            pickle.dump(out_dictl, f, protocol=pickle.HIGHEST_PROTOCOL)  
        print(f'Data saved as \'{save_filename}\'.')
        out_dictl = None
    else:
        pass # returns non-None 'out_dictl' below
    
    t_overall = time.time() - t_0
    t_min, t_sec = u.get_time_min_sec(t_overall)
    print(
        f'{args.N_MANIFOLDS} manifold'
        f' {args.GRAPH_TYPE} graphs and associated objects generated'
        f' in:\n\t{t_min:.0f}min, {t_sec:.4f}sec.'
    )

    return out_dictl


class WaveletPyGData(Data):
    """
    Subclass of PyG 'Data' class that collates
    wavelet filter attributes in new (first) dimension,
    instead of concatenating in dim 0 (default).

    Note that `mfcn.get_Batch_spectral_Wjxs` expects 
    `Wjs_spectral` (spectral filter tensors) in the shape 
    (n_graphs, n_eigenpairs, n_filters).
    
    Reference:
    https://pytorch-geometric.readthedocs.io/en/2.5.0/advanced/batching.html
    """
    def __cat_dim__(self, key, value, *args, **kwargs):
        # note: these keys have been deprecated, and
        # shape of 'Wjs_spectral' from w.spectral_wavelets transposed
        if (key =='L_eigenvals'):
        # or (key =='Wjs_spectral') \
        # or (key == 'spectral_lowpass_filters'):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def pickle_as_pytorch_geo_Data(
    args,
    data_dictl: Optional[List[Dict[str, Any]]] = None,
    extra_attribs: Optional[Dict[str, type]] = {
        'L_eigenvecs': float, 
        'L_eigenvals': float
    },
    manual_save_path: Optional[str] = None
) -> None:
    """
    Loads a manifold wavelet dataset
    into a list of PyTorch Geometric 
    Data Objects, and pickles it.

    Args:
        args: ArgsTemplate subclass, holding
            experiment arguments.
        data_dictl: if a list of dictionaries of 
            data objects already exists, use what's
            passed to this arg; else create it from
            data at args.RAW_DATA_FILENAME.
        extra_attribs: dict of extra attributes and their types
            to add to each Data object, such as 'L_eigenvecs' and
            'L_eigenvals', 'id', etc. The keys in this list must
            also be in data_dictl to be added to the Data objects.
        manual_save_path: if not None, save
            result to this filename; else use what's at
            args.PYG_DATA_LIST_FILENAME.
    Returns:
        None (pickles the list).
    """
    if data_dictl is None:
        print(f'Creating PyG dataloaders from \'{args.RAW_DATA_FILENAME}\'')
    
        # open pickled data dict list
        pyg_data_list_filepath = f'{args.DATA_DIR}/{args.RAW_DATA_FILENAME}'
        with open(pyg_data_list_filepath, "rb") as f:
            data_dictl = pickle.load(f)
    
    # create master list of torch_geometric.data.Data objects
    data_list = [None] * len(data_dictl)
    for i, dict in enumerate(data_dictl):

        # 'x': input signal data tensor
        # shape (n_nodes, n_features/channels)
        Pnfs = dict['Pnfs']
        if isinstance(Pnfs, spmatrix): # sp.coo_matrix
            # if not a scipy 'coo_matrix' but another sparse type,
            # first convert to 'coo_matrix', then to 'sparse_coo_tensor'
            if not isinstance(Pnfs, sp.coo_matrix):
                Pnfs = Pnfs.tocoo()
            indices = torch.tensor(
                np.stack((Pnfs.row, Pnfs.col)), 
                dtype=torch.long
            )
            values = torch.tensor(Pnfs.data, dtype=torch.float)
            x = torch.sparse_coo_tensor(
                indices, 
                values, 
                Pnfs.shape
            )
        else:
            x = torch.tensor(
                Pnfs, 
                dtype=args.FEAT_TENSOR_DTYPE
            )
        
        edge_index = torch.tensor( # shape (2, n_edges)
            np.stack(dict['W'].nonzero()), 
            dtype=torch.long
        )
        
        y = torch.tensor(
            dict[args.TARGET_NAME],
            dtype=args.TARGET_TENSOR_DTYPE
        )
        
        # optional: log-transform main target
        if args.LOG_TRANSFORM_TARGETS:
            y = torch.log(y)
        
        # additional attribute(s) for Data objects
        extra_attribs_kwargs = {}
        
        for attrib, data_type in extra_attribs.items():
            if attrib in dict:
                # print(f'{attrib} type: {type(dict[attrib])}')
                dtype = du.convert_to_torch_dtype(
                    data_type=data_type,
                    int_dtype=torch.long,
                    float_dtype=args.FLOAT_TENSOR_DTYPE
                )
                if data_type == str:
                    extra_attribs_kwargs[attrib] = dict[attrib]
                else:
                    extra_attribs_kwargs[attrib] = torch.tensor(
                        dict[attrib], 
                        requires_grad=False,
                        dtype=dtype
                    )
            
        # finally, create and add Data object to master list
        data = WaveletPyGData(
            x=x, 
            y=y,
            edge_index=edge_index,
            **extra_attribs_kwargs
        )
        data_list[i] = data

    save_path = manual_save_path \
        if (manual_save_path is not None) \
        else f'{args.DATA_DIR}/{args.PYG_DATA_LIST_FILENAME}'
    if save_path[-4:] != '.pkl':
        save_path += '.pkl'
    # save_path = f'{args.DATA_DIR}/{save_filename}'
    with open(save_path, "wb") as f:
        pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)  
    print(f'PyG Data list saved to \'{save_path}\'.')


def get_pyg_data_or_dataloaders(
    args,
    pyg_data_list_filepath: Optional[str] = None,
    set_idxs_dict: Optional[Dict[str, int]] = None,
    verbosity: int = 0
) -> Tuple[Dict[str, DataLoader], Optional[float]]:
    """
    Splits and loads a manifold wavelet dataset
    into a dict of train/valid/test (PyTorch Geometric) 
    DataLoaders, or a single Data object (with train/valid/
    test mask attributes, in the case of node classification 
    for a single graph).

    NOTE: if dealing with a one-graph (node-level) dataset,
    this method will attempt to move the (lone) Data object 
    onto the device specified in 'args'.
    
    This function also returns a positive class rebalancing
    weight, if doing binary classification (done
    here, since this is where the train set is split,
    and this weight should be calculated only on
    the train set).

    NOTE: if dataset is from oversampling,
    samples from same subject should be in
    the same train/valid/test set, and those
    split indexes should be generated upstream
    and fed to this function in 'set_idxs_dict'.

    Args:
        args: ArgsTemplate subclass, holding
            experiment arguments.
        pyg_data_list_filepath: manual override of filepath
            to the PyG Data list dataset pickle.
        set_idxs_dict: optional dictionary of
            index arrays keyed by set name
            ('train'/'valid'/'test'). If 'None',
            set indexes are generated here, using
            the args. 
        verbosity: controls volume of print output
            as function executes.
    Returns:
        Tuple of: (1) PyG Data object or dictionary of  
        PyGDataLoaders keyed by set name, and 
        (2) positive class rebalancing weight (float), 
        if doing binary classification.
    """
    if pyg_data_list_filepath is None:
        pyg_data_list_filepath = f'{args.DATA_DIR}/{args.PYG_DATA_LIST_FILENAME}'
    # open pickled PyG Data object list
    with open(pyg_data_list_filepath, "rb") as f:
        data_list = pickle.load(f)
    
    # [if needed] generate train/valid/test set split idxs
    if set_idxs_dict is None:
        set_idxs_dict = du.get_train_valid_test_idxs(
            seed=args.TRAIN_VALID_TEST_SPLIT_SEED,
            n=len(data_list),
            train_prop=args.TRAIN_PROP,
            valid_prop=args.VALID_PROP
        )

    # [optional] extract positive class balance weight
    # from train set, in binary classification tasks
    if 'bin' in args.TASK.lower() \
    and 'class' in args.TASK.lower():
        train_idx = set_idxs_dict['train']
        train_set = [data_list[i] for i in train_idx]
        train_targets = [data.y for data in train_set]
        
        n = len(train_targets)
        ct_1s = np.sum(train_targets)
        rebalance_pos_wt = torch.tensor((n - ct_1s) / ct_1s)
        p = ct_1s / n
        perc_1s = p * 100
        mcc_acc = 100 - perc_1s if perc_1s < 50 else perc_1s
        mcc_f1 = (2 * p) / (p + 1)

        if verbosity > 0:
            print(f'binary target % of 1s: {perc_1s:.1f}%')
            print(f'\t-> balanced positive class weight: {rebalance_pos_wt:.2f}')
            print(f'\t-> majority-class classifier accuracy: {mcc_acc:.1f}%')
            print(f'\t-> majority-class classifier F1: {mcc_f1:.2f} (if maj. class is pos. class)')
    else:
        rebalance_pos_wt = None
    

    # node-level tasks
    if 'node' in args.TASK:

        # [node-level task] if there's only one graph in the dataset
        if len(data_list) == 1:
            data = data_list[0]

            # assign 'set_idxs_dict' idxs to the 'mask' attributes of the 
            # graph's Data object
            for i, (set_key, idx) in enumerate(set_idxs_dict.items()):
                set_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                set_mask[idx] = True
                
                if 'train' in set_key.lower():
                    data.train_mask = set_mask
                elif 'val' in set_key.lower():
                    data.val_mask = set_mask
                elif 'test' in set_key.lower():
                    data.test_mask = set_mask

            # move the one-graph dataset onto device
            data.to(args.DEVICE)
        
            return (data, rebalance_pos_wt)

        # [node-level task] if there are multiple graphs in the dataset
        else:
            raise NotImplementedError(
                f"Creating Dataloaders for node-level tasks for multiple" 
                f" graphs not yet been implemented!"
            )
            
    # graph-level tasks
    elif 'graph' in args.TASK:
        # create dict of dataloaders for data_container
        data_container = {}

        # populate dict with set dataloaders
        for i, (set_key, idx) in enumerate(set_idxs_dict.items()):
            dataset = [data_list[i] for i in idx]
            # print('dc.get_pyg_data_or_dataloaders: len(dataset)', len(dataset))
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=args.BATCH_SIZES[i],
                shuffle=('train' in set_key),
                drop_last=args.DATALOADER_DROP_LAST
            )
            data_container[set_key] = dataloader
        # print('\tPyG DataLoader ready.')
    
    # other task
    else:
        raise NotImplementedError(
                f"Creating Dataloaders for this task ({args.TASK})"
                f" has not been implemented! Did you forget 'node' or"
                f" 'graph' in args.TASK?"
            )
        
    return (data_container, rebalance_pos_wt)




"""
scratch
"""
# NOTE: only need to save base P matrix (done above) for
# efficient recursive P-matrix filtration
# if args.SAVE_MFCN_P_OBJS:
#     print('Calculating P-based wavelet filter operators...')
#     for i, P in enumerate(graph_Ps):
        # Wjs_P = w.lazy_rw_wavelets(
        #     P=P,
        #     J=args.J,
        #     include_low_pass=args.INCLUDE_LOWPASS_WAVELET,
        # )
        # out_dictl[i]['Wjs_P'] = Wjs_P

