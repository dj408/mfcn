"""
Ellipsoids node regression task 
experiment-specific arguments.

Note that other args (such as save
directories) not overridden here
default to their values in the
ArgsTemplate.py file.
"""
import sys
sys.path.insert(0, '../')
from args_template import ArgsTemplate
from dataclasses import dataclass
import os
import time
import datetime
from torch import float32, long
from typing import (
    Tuple,
    Any,
    Optional,
    Iterable,
    Callable
)


@dataclass
class Args(ArgsTemplate):

    # exclusions
    EXCLUDE_DATASET_INDICES: Optional[Tuple[int]] = None

    # seeds
    CV_SPLIT_SEED: int = 142789
    ABC_SEED: int = 274851
    POINT_SAMPLING_SEED: int = 956438
    RAND_ROTATION_SEEDS: Tuple[int] = (
        217389,
        347880,
        228911,
        609583,
        784753,
        578915,
        109348,
        983745,
        674392,
        435893,
        574289,
        434258,
        834972,
        847338,
        248759,
        458246,
        674728,
        494238,
        984570,
        132487
    )
    MANIFOLD_SAMPLE_NOISE_SEEDS: Tuple[int] = (
        473751,
        785347,
        234758,
        587937,
        785943,
        784315,
        958722,
        312758,
        548258,
        308700
    )
    # FN_ON_MANIFOLD_SEED: int = 758976
    DATALOADERS_SEED: int = 549256
    TRAIN_VALID_TEST_SPLIT_SEED: int = 245879
    # TORCH_MANUAL_SEED: int = 
    TORCH_MANUAL_SEEDS: Tuple[int] = (
        365784,
        577580,
        758291,
        259737,
        270589,
        588354,
        784927,
        147833,
        620748,
        444291
    )
    
    # data creation params
    SAVE_SPECTRAL_OBJS: bool = True
    SAVE_SCAT_MOMENTS_OBJS: bool = False
    SAVE_FN_VALS: bool = True # needed for torch_geometric.data.Data objects
    SAVE_GRAPH: bool = True # needed for torch_geometric.data.Data objects
    
    # which axes should serve as function-on-manifold values?
    # 'all', or if == (2, ) -> fn on manifolds are 'z' coord. values
    COORD_FN_AXES: str | Tuple[int] = 'all' 
    STACK_CHANNEL_INPUTS: bool = False
    D_MANIFOLD: int = 2
    AMBIENT_DIM: int = 8
    MANIFOLD_N_AXES: int = 3
    N_MANIFOLDS: int = 1
    N_PTS_ON_MANIFOLD: int = 1024
    ADD_NOISE_TO_MANIFOLD_SAMPLE: bool = True
    SAMPLE_NOISE_VAR_CONSTANT: Optional[float] = 0.05
    N_OVERSAMPLES: Optional[int] = None
    GRAPH_TYPE: str = 'knn'
    K_OR_EPS: str | int = 'auto'

    # filters params
    NON_WAVELET_FILTER_TYPE: Optional[str] = 'spectral' # 'spectral', 'p'
    SPECTRAL_C: Optional[float] = 0.5
    WAVELET_TYPE: Optional[str] = 'p' # 'spectral', 'p'
    KAPPA: int = 20 # max num. nontrivial eigenpairs calc'd
    MFCN_MAX_KAPPA: Optional[int] = 20 # max num. eigenpairs used in MFCN-spectral model: leave None to use all saved
    J: int = 4
    Q: int = 4 # which scat. moments to calc
    # Q_IDX = which scat. moments to use
        # (1, ) for q=2 moments only
        # (0, 1, 2, 3) for q=1...4
    Q_IDX = (0, 1, 2, 3)
    INCLUDE_LOWPASS_WAVELET: bool = True
    P_WAVELET_SCALES: str = 'handcrafted' # 'dyadic'
    HANDCRAFT_P_CMLTV_KLD_QUANTILES: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875) # (0.2, 0.4, 0.6, 0.8)

    # training data params
    FEATURE_SCALING_TYPE: str = None
    DROP_ZERO_VAR_FEATURES: bool = True
    LOG_TRANSFORM_TARGETS: bool = False
    RAW_DATA_FILENAME: str = 'ellip_node_reg_data.pkl' 
    PYG_DATA_LIST_FILENAME: str = 'ellip_node_reg_pyg_data_list.pkl'
    # DATASETS_DICT_FILENAME: str = 'ellip_node_reg_datasets.pkl'
    DATALOADER_N_WORKERS: int = 0
    DATALOADER_DROP_LAST: bool = False

    # TRAIN_PROP: float = 0.8
    # note: for ex, 0.2 (vs 0.21) may underpopulate the valid set
    # VALID_PROP: float = 0.1
    # change batch size depending on memory avail; use powers of 2
    BATCH_SIZES: Tuple[int] = (1024, 1024, 1024)

    # model params
    MODEL_NAME: str = 'model'
    GNN_TYPES: Tuple[str] = ('gcn', 'sage', 'gin', 'gat')
    TASK: str = 'node_regression'
    TARGET_NAME: str = 'target'
    TARGET_TENSOR_DTYPE = float32
    NN_HIDDEN_DIMS: Tuple[int] = (128, 64, 32, 16) # (512, 64, 16, 4)
    MCN_WITHIN_FILTER_CHAN_OUT: Tuple[int] = (32, 16)
    MFCN_WITHIN_FILTER_CHAN_OUT: Tuple[int] = (8, 4)
    MFCN_CROSS_FILTER_COMBOS_OUT: Tuple[int] = (8, 4)
    MFCN_FINAL_CHANNEL_POOLING: Optional[str] = None # 'moments', 'max', 'mean'
    MFCN_FINAL_NODE_POOLING: Optional[str] = 'linear' # 'max', 'mean'
    GNN_FINAL_CHANNEL_POOLING: Optional[str] = None
    # GNN_FINAL_NODE_POOLING: Optional[str] = None
    OUTPUT_DIM: int = N_PTS_ON_MANIFOLD
    USE_BATCH_NORMALIZATION: bool = True
    MLP_USE_DROPOUT: bool = False
    MLP_DROPOUT_P: float = 0.5 # probability of being 'zeroed'
    GNN_USE_DROPOUT: bool = False
    GNN_DROPOUT_P: float = 0.5 

    # model training params
    N_FOLDS: int = 10
    SAVE_STATES: bool = False
    STOP_RULE: str = 'no_improvement'
    NO_VALID_LOSS_IMPROVE_PATIENCE: int = 32
    SAVE_FINAL_MODEL: bool = True
    TRAIN_HIST_PREFIX: Optional[str] = None
    
    # burn-in num of epochs prevents early stopping before it's reached
    BURNIN_N_EPOCHS: int = 5
    N_EPOCHS: int = 256
    LEARN_RATE: float = 0.005
    ADAM_BETAS: Tuple[float] = (0.9, 0.999)
    ADAM_WEIGHT_DECAY: float = 1e-5
    LRELU_NSLOPE: float = 0.01
    # args for saving 'best' model during training, by a 
    # validation metric
    MAIN_METRIC: str = 'loss_valid'
    MAIN_METRIC_IS_BETTER: str = 'lower' # or: 'higher'
    # MAIN_METRIC_INIT_VAL: float = 1.0
    MAIN_METRIC_REL_IMPROV_THRESH: Optional[float] = 0.995

    # paths vars (full paths set in super().__post_init__)
    ROOT: str = None
    DATA_DIR: str = None
    MODEL_SAVE_DIR: str = None
    PRINT_DIR: str = None
    TRAIN_LOGS_SAVE_DIR: str = None
    DATA_SUBDIR: str = 'ellipsoids_node'
    MODEL_SAVE_SUBDIR: str = 'ellipsoids_node'
    RESULTS_SAVE_SUBDIR: str = 'ellipsoids_node'


