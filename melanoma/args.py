"""
Melanoma graph classification task 
experiment arguments.

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

    # seeds
    POINT_SAMPLING_SEED: int = 156238
    FN_ON_MANIFOLD_SEED: int = 358446
    CV_SPLIT_SEED: int = 162769
    DATALOADERS_SEED: int = 240056
    TRAIN_VALID_TEST_SPLIT_SEED: int = 253279
    # TORCH_MANUAL_SEED: int = 635585 # cv.py.151
    TORCH_MANUAL_SEEDS: Tuple[int] = (
        665877,
        515435,
        453669,
        190358,
        908537,
        578348,
        599959,
        898580,
        232237,
        349476
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
    D_MANIFOLD: int = 8
    MANIFOLD_N_AXES: int = 29
    AMBIENT_DIM: int = 29
    N_PATIENTS: int = 54
    N_OVERSAMPLES: Optional[int] = None # EB: 10
    N_MANIFOLDS: int = 54 # 540
    MANIFOLD_KEY: str = 'cell_intensities' # old: 'manifold'
    # N_PTS_ON_MANIFOLD: int = 400 # pts have diff. numbers of cell samples!
    GRAPH_TYPE: str = 'knn'
    K_OR_EPS: str | int = 'auto'
    
    # filters params
    NON_WAVELET_FILTER_TYPE: Optional[str] = 'spectral' # 'spectral', 'p'
    SPECTRAL_C: Optional[float] = 0.5
    WAVELET_TYPE: Optional[str] = 'p' # 'spectral', 'p'
    KAPPA: int = 20 # max num. eigenpairs calc'd
    MFCN_MAX_KAPPA: Optional[int] = 20 # max num. eigenpairs used in MFCN-spectral model: leave None to use all saved
    J: int = 4
    Q: int = 4 # which scat. moments to calc
    # Q_IDX = which scat. moments to use
        # (1, ) for q=2 moments only
        # (0, 1, 2, 3) for q=1...4
    Q_IDX: Tuple[int] = (0, 1, 2, 3)
    P_WAVELET_SCALES: str = 'dyadic' # 'handcrafted'
    INCLUDE_LOWPASS_WAVELET: bool = True
    HANDCRAFT_P_CMLTV_KLD_QUANTILES: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875) # (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) # (0.2, 0.4, 0.6, 0.8)
    

    # data params
    RAW_DATA_FILENAME: str = 'melanoma_manifolds_dictl.pkl' # old: 'melanoma_raw_data.pkl'
    PYG_DATA_LIST_FILENAME: str = 'melanoma_pyg_data_list.pkl'
    # DATASETS_DICT_FILENAME: str = 'melanoma_datasets.pkl'
    FEATURE_SCALING_TYPE: str = None # 'minmax'; None for no rescaling
    DROP_ZERO_VAR_FEATURES: bool = True
    LOG_TRANSFORM_TARGETS: bool = False
    DATALOADER_N_WORKERS: int = 0
    DATALOADER_DROP_LAST: bool = False

    # model params
    MODEL_NAME: str = 'model'
    GNN_TYPES: Tuple[str] = ('gcn', 'sage', 'gin', 'gat')
    TASK: str = 'binary_graph_classification'
    TARGET_NAME: str = 'response_binary' # response
    # note `F.binary_cross_entropy_with_logits` needs targets as floats
    TARGET_TENSOR_DTYPE = float32 
    STRATIF_SAMPLING_KEYS: Tuple[str] = ('response_binary', 'gross_dx_stage')
    MCN_WITHIN_FILTER_CHAN_OUT: Tuple[int] = (32, 16)
    MFCN_WITHIN_FILTER_CHAN_OUT: Tuple[int] = None # (8, 4)
    MFCN_CROSS_FILTER_COMBOS_OUT: Tuple[int] = (16, 8)
    # MFCN final channel pool options: 'max+mean', 'moments', 'max', 'mean'
    MFCN_FINAL_CHANNEL_POOLING: Optional[str] = 'max'
    MFCN_FINAL_NODE_POOLING: Optional[str] = None # 'linear', 'max', 'mean'
    
    GNN_FINAL_CHANNEL_POOLING: Optional[str] = 'max'
    GNN_FINAL_NODE_POOLING: Optional[str] = None
    NN_HIDDEN_DIMS: Tuple[int] = (128, 64, 32, 16) # (512, 64, 16, 4)
    OUTPUT_DIM: int = 1
    USE_BATCH_NORMALIZATION: bool = True
    MLP_USE_DROPOUT: bool = False
    MLP_DROPOUT_P: float = 0.5 # probability of being 'zeroed'
    GNN_USE_DROPOUT: bool = False
    GNN_DROPOUT_P: float = 0.5

    # training params
    N_FOLDS: int = 10
    # TRAIN_PROP: float = 0.8
    # note: for ex, 0.2 (vs 0.21) may underpopulate the valid set
    # VALID_PROP: float = 0.1
    # change batch size depending on memory avail; use powers of 2
    BATCH_SIZES: Tuple[int] = (8, 1024, 1024) # (128, 64, 64)
    
    N_EPOCHS: int = 256
    # burn-in num of epochs prevents early stopping before it's reached
    BURNIN_N_EPOCHS: int = 100
    LEARN_RATE: float = 0.01
    SAVE_STATES: bool = False
    STOP_RULE: str = 'no_improvement'
    NO_VALID_LOSS_IMPROVE_PATIENCE: int = 50
    SAVE_FINAL_MODEL_STATE: bool = False
    TRAIN_HIST_PREFIX: Optional[str] = None
    
    ADAM_BETAS: Tuple[float] = (0.9, 0.999)
    ADAM_WEIGHT_DECAY: float = 1e-2 # default 1e-2 in AdamW
    LRELU_NSLOPE: float = 0.01
    # args for saving 'best' model during training, by a 
    # validation metric
    MAIN_METRIC: str = 'loss_valid'
    MAIN_METRIC_IS_BETTER: str = 'lower' # or: 'higher'
    # MAIN_METRIC_INIT_VAL: float = 1.0
    # prevents new best model saves until a relative improvement
    # in main metric is reached (not arbitrarily small)
    MAIN_METRIC_REL_IMPROV_THRESH: Optional[float] = None # 0.999

    # paths vars (full paths set in super().__post_init__)
    ROOT: str = None
    DATA_DIR: str = None
    MODEL_SAVE_DIR: str = None
    PRINT_DIR: str = None
    TRAIN_LOGS_SAVE_DIR: str = None
    DATA_SUBDIR: str = 'melanoma'
    MODEL_SAVE_SUBDIR: str = 'melanoma'
    RESULTS_SAVE_SUBDIR: str = 'melanoma'


    def set_raw_data_filename(self):
        pass


