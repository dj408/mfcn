"""
Model class for a manifold filter-combine 
network (MFCN).
"""
import base_module as bm
import vanilla_nn as vnn
import data_utilities as du
import wavelets as w
import nn_utilities as nnu

from numpy import (
    nanmax,
    linspace,
    log2
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset
)
from torch import linalg as LA
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset,
    Batch
)
import os
import pickle
import warnings
from itertools import accumulate
from functools import reduce
from operator import mul
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    Callable
)


class MFCN_Module(bm.BaseModule):
    r"""
    For the following args definitions:
    Let 'n' be the number of samples / points on 
    a manifold, and 'k' be the number of eigenpairs 
    evaluated from the graph Laplacian of each manifold.

    __init__ args:
        task: string key containing 'graph' or 'node' substring
            for model task type.
        n_channels: number of signal input channels
            present in the model input ('x') tensor.
        wavelet_type: 'P' or 'spectral', indicating
            the type of wavelet filter operator used (defaults to
            'P' lazy random walk wavelets).
        non_wavelet_filter_type: 'P' or 'spectral', indicating
            the type of non-wavelet filter operator used, e.g.
            in a simpler MCN model. Leave None to default to wavelet
            filters instead.
        filter_c: optional float constant to apply to a 
            spectral filter, e.g. $e^{-\lambda} \rightarrow
            e^{-c\lambda}$ for lowpass filters. Also applies to
            dyadic wavelets.
        num_nodes_one_graph: number of nodes on the one graph,
            for node-level tasks.
        J: max index of wavelet filters.
        P_wavelets_channels_t_is: pre-computed output tensor from 
            `handcrafted_P_wavelet_scales`, containing the non-dyadic 
            indices/powers of t (diffusion step) marking the handcrafted
            wavelet boundaries; shape (n_channels, n_ts).
        cross_Wj_ns_combos_out_per_chan: m-tuple holding the desired
            number of new filtered signal combinations to
            create per channel, at each cross-filter convolution step in
            each cycle, where m is the number of cycles. Note 
            that the values here can be decreasing (from n_filters)
            to help mitigate a combinatorial explosion.
        within_Wj_ns_chan_out_per_filter: m-tuple holding the desired
            number of new cross-channel, within-filter combinations of
            input signals at each 'combine' step.
        channel_pool_moments: [graph-level tasks] if not None and 'channel_pool_key'
            is 'moments', this option computes moments of each channel of signal
            within each graph, of orders specified in its tuple (e.g., 1...4).
            This is a form of pooling across nodes (within channels), before 
            concatenating and feeding to an MLP head.
        channel_pool_key: see 'channel_pool_moments' argument description.
        node_pooling_key: 'mean', 'max', or 'linear' (for a learned linear
            combination) for pooling across final channels (within nodes).
            For node-level tasks, the output of this final pooling is the
            model output/predictions; for graph-level tasks, the output is
            fed to an MLP head.
        node_pool_linear_out_channels: number of final channels per node 
            after linear layer node pooling) to output when no fully-connected 
            head is used. Similar to 'out_channels' in pytorch geometric's
            GCN, etc., model classes.
        use_skip_cxn: whether to add a skip connection concatenating (the
            scattering moments of) the original input (before MFCN cycles) 
            to the post-MFCN input into the fully-connected network. (These
            are considered 'zeroth-order' scattering moments.)
        zero_order_scat_moms_only: bool, if True, the only features fed into 
            the fully-connected head are the (concatenated) scattering moments 
            of the original channel signals.
        mfcn_nonlin_fn: nonlinear activation function within the MFCN
            model.
        mfcn_nonlin_fn_kwargs: kwargs for 'mfcn_nonlin_fn'.
        mfcn_wts_init_fn: the torch.nn parameter initialization
            function desired to initialize within-Wj and cross-Wj trainable
            parameter weights.
        mfcn_wts_init_kwargs: kwargs to pass to the 'mfcn_wts_init_fn'.
        base_module_kwargs: kwargs to pass to the base_module.BaseModule
            parent class.
        fc_kwargs: kwargs to pass to the fully-connected network module
            (vanilla_nn.VanillaNN instance).
        verbosity: integer controlling print output volume during
            methods executions.

    """
    def __init__(
        self,
        n_channels: int,
        wavelet_type: Optional[str] = 'p',
        non_wavelet_filter_type: Optional[str] = None, # defaults to wavelet filters
        filter_c: Optional[float] = None,
        num_nodes_one_graph: Optional[int] = None,
        J: int = 4,
        P_wavelets_channels_t_is: Optional[torch.Tensor] = None,
        max_kappa: Optional[int] = None,
        include_lowpass_wavelet: bool = True,
        within_Wj_ns_chan_out_per_filter: Optional[Tuple] = (8, 4),
        cross_Wj_ns_combos_out_per_chan: Tuple = (8, 4),
        channel_pool_moments: Optional[Tuple[int]] = (1, 2, 3, 4, float('inf')),
        channel_pool_key: Optional[str] = None, # 'moments', 'max', 'mean'
        node_pooling_key: Optional[str] = None, # 'linear', 'max', 'mean'
        node_pool_linear_out_channels: int = 1,
        # use_skip_cxn: bool = False,
        # zero_order_scat_moms_only: bool = False,
        mfcn_nonlin_fn = F.relu, # F.leaky_relu,
        mfcn_nonlin_fn_kwargs: dict = {}, # {'negative_slope': 0.01},
        mfcn_wts_init_kwargs: dict = {}, # {'nonlinearity': 'leaky_relu', 'a': 0.01},
        mfcn_wts_init_fn = nn.init.kaiming_uniform_,
        base_module_kwargs: Dict[str, Any] = {},
        fc_kwargs: Dict[str, Any] = {},
        verbosity: int = 0
    ):
        """
        Initialize basic attributes of the MFCN module.
        """
        super(MFCN_Module, self).__init__(**base_module_kwargs)
        self.wavelet_type = wavelet_type.lower() if wavelet_type is not None else None
        self.non_wavelet_filter_type = non_wavelet_filter_type.lower() \
            if (non_wavelet_filter_type is not None) else None
        self.filter_c = filter_c
        self.max_kappa = max_kappa
        self.Wjs_spectral = None
        self.V_sparse = None
        self.J = J
        self.include_lowpass_wavelet = include_lowpass_wavelet
        if self.wavelet_type is not None:
            self.n_filters = (self.J + 2) if self.include_lowpass_wavelet else (self.J + 1)
        else:
            self.n_filters = 1
        self.P_wavelets_channels_t_is = P_wavelets_channels_t_is
        self.within_Wj_ns_chan_out_per_filter = within_Wj_ns_chan_out_per_filter
        self.cross_Wj_ns_combos_out_per_chan = cross_Wj_ns_combos_out_per_chan
        self.mfcn_nonlin_fn = mfcn_nonlin_fn
        self.mfcn_nonlin_fn_kwargs = mfcn_nonlin_fn_kwargs
        self.channel_pool_moments = channel_pool_moments
        self.node_pooling_key = node_pooling_key.lower() \
            if (node_pooling_key is not None) else None
        self.channel_pool_key = channel_pool_key.lower() \
            if (channel_pool_key is not None) else None
        self.verbosity = verbosity
        self.fc_kwargs = fc_kwargs
        self.fc = None

        """
        Initialize MFCN module with specified parameters.
        """
        if (within_Wj_ns_chan_out_per_filter is not None) \
        and (cross_Wj_ns_combos_out_per_chan is not None):
            if len(within_Wj_ns_chan_out_per_filter) \
            != len(cross_Wj_ns_combos_out_per_chan):
                print(
                    '''
                    ERROR: WaveletMFCN requires the same number of within- and 
                    cross-filter steps (if both are not None).
                    '''
                )
                return None

        # Initialize MFCN learnable parameters if needed
        if (within_Wj_ns_chan_out_per_filter is not None) \
        or (cross_Wj_ns_combos_out_per_chan is not None):
            self._init_mfcn_parameters(
                n_channels, within_Wj_ns_chan_out_per_filter,
                cross_Wj_ns_combos_out_per_chan, mfcn_wts_init_fn,
                mfcn_wts_init_kwargs
            )

        # Initialize node pooling parameters if needed
        if (node_pooling_key is not None) \
        and ('linear' in node_pooling_key):
            self._init_node_pooling_parameters(
                n_channels, within_Wj_ns_chan_out_per_filter,
                cross_Wj_ns_combos_out_per_chan, node_pool_linear_out_channels
            )
        

    def _init_mfcn_parameters(
        self,
        n_channels: int,
        within_Wj_ns_chan_out_per_filter: Optional[Tuple],
        cross_Wj_ns_combos_out_per_chan: Tuple,
        mfcn_wts_init_fn,
        mfcn_wts_init_kwargs: dict
    ) -> None:
        """
        Initialize MFCN learnable parameters for within-filter and 
        cross-filter combinations.
        """
        n_cycles = len(cross_Wj_ns_combos_out_per_chan) \
            if (cross_Wj_ns_combos_out_per_chan is not None) \
            else len(within_Wj_ns_chan_out_per_filter)

        if self.P_wavelets_channels_t_is is None:
            n_filters_by_cycle = [self.n_filters] * n_cycles
        else:
            custom_P_n_filters = self.P_wavelets_channels_t_is.shape[1] - 1
            if self.include_lowpass_wavelet:
                custom_P_n_filters += 1
            n_filters_by_cycle = [custom_P_n_filters] + ([self.n_filters] * (n_cycles - 1))

        if self.wavelet_type is not None:
            self.Wjs_key = 'Wjs_P' if self.wavelet_type == 'p' else 'Wjs_spectral'

        # Compute the dimensions for the within-filter parameters
        if within_Wj_ns_chan_out_per_filter is not None:
            # If there is also a cross-filter step, the two steps will
            # multiply their output channels; otherwise, the requested 
            # within-filter number of output channels will be multiplied by
            # the number of filters each cycle
            within_Wj_ns_chan_in = [n_channels] + \
                [
                    within_Wj_n_chan_out * cross_Wj_n_combos_out \
                    for within_Wj_n_chan_out, cross_Wj_n_combos_out \
                    in zip(
                        within_Wj_ns_chan_out_per_filter, 
                        cross_Wj_ns_combos_out_per_chan
                    )
                ] \
                if cross_Wj_ns_combos_out_per_chan is not None \
                else [n_channels] + \
                    [
                        (self.n_filters ** (i + 1)) * chan_out \
                        for i, chan_out in enumerate(within_Wj_ns_chan_out_per_filter)
                    ]
                # else [n_channels] + list(within_Wj_ns_chan_out_per_filter)
            
            # Initialize within-filter parameters
            self.within_Wj_params = nn.ParameterList([
                torch.nn.Parameter(
                    torch.zeros(
                        self.n_filters,
                        n_channels_in,
                        n_channels_out
                    ),
                    requires_grad=True
                ) for n_filters, n_channels_in, n_channels_out \
                in zip(
                    n_filters_by_cycle,
                    within_Wj_ns_chan_in, 
                    within_Wj_ns_chan_out_per_filter
                )
            ])
            for within_Wj in self.within_Wj_params:
                mfcn_wts_init_fn(within_Wj, **mfcn_wts_init_kwargs)
            # for i, within_Wj in enumerate(self.within_Wj_params):
            #     print(f'within_Wj {i}.shape: {within_Wj.shape}')

        # Compute the dimensions for the cross-filter parameters
        if cross_Wj_ns_combos_out_per_chan is not None:
            # If there is a within-filter step, use its output channels as input
            # to the cross-filter step; otherwise, the cross-filter steps will
            # accumulate
            cross_Wj_ns_chan_in = within_Wj_ns_chan_out_per_filter \
                if within_Wj_ns_chan_out_per_filter is not None \
                    else list(accumulate(
                    [n_channels] + list(cross_Wj_ns_combos_out_per_chan), 
                    lambda x, y: x * y
                ))[:-1]
                # else [n_channels] + list(accumulate(
                #     cross_Wj_ns_combos_out_per_chan, 
                #     lambda x, y: x * y
                # ))[:-1]

            # Initialize cross-filter parameters
            self.cross_Wj_params = nn.ParameterList([
                torch.nn.Parameter(
                    torch.zeros(
                        n_channels,
                        n_filters,
                        n_filter_combos_out, 
                    ),
                    requires_grad=True
                ) for n_channels, n_filters, n_filter_combos_out in zip(
                    cross_Wj_ns_chan_in,
                    n_filters_by_cycle,
                    cross_Wj_ns_combos_out_per_chan
                )
            ])
            for cross_Wj in self.cross_Wj_params:
                mfcn_wts_init_fn(cross_Wj, **mfcn_wts_init_kwargs)
            # for i, cross_Wj in enumerate(self.cross_Wj_params):
            #     print(f'cross_Wj {i}.shape: {cross_Wj.shape}')


    def _init_node_pooling_parameters(
        self,
        n_channels: int,
        within_Wj_ns_chan_out_per_filter: Optional[Tuple],
        cross_Wj_ns_combos_out_per_chan: Tuple,
        node_pool_linear_out_channels: int
    ) -> None:
        """
        Initialize parameters for linear node pooling, with a bias term.
        """
        if within_Wj_ns_chan_out_per_filter is not None:
            if cross_Wj_ns_combos_out_per_chan is not None:
                # both within- and cross-filter steps
                node_pool_mult = cross_Wj_ns_combos_out_per_chan[-1]
            else:
                # only within-filter step
                node_pool_mult = self.n_filters
            final_reshape_chan_dim = node_pool_mult * within_Wj_ns_chan_out_per_filter[-1]
        elif cross_Wj_ns_combos_out_per_chan is not None:
            # only cross-filter step
            final_reshape_chan_dim = n_channels * reduce(mul, cross_Wj_ns_combos_out_per_chan)
            # final_reshape_chan_dim = cross_Wj_ns_combos_out_per_chan[-1]
        else:
            # neither within- nor cross-filter steps
            final_reshape_chan_dim = n_channels * self.n_filters

        # Initialize node-pooling parameters of appropriate shape
        self.node_pool_wts = torch.nn.Parameter(
            torch.empty((final_reshape_chan_dim, node_pool_linear_out_channels))
        )
        torch.nn.init.uniform_(self.node_pool_wts, a=-0.1, b=0.1)
        self.node_pool_bias = torch.nn.Parameter(torch.zeros(1))


    def _apply_node_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        If applicable, apply node-level pooling across channels.
        
        Args:
            x: Tensor of shape (N, d) where N is number of nodes and d 
            is number of channels
            
        Returns:
            Pooled tensor
        """
        if self.node_pooling_key is None:
            return x
            
        if 'mean' in self.node_pooling_key:
            return torch.mean(x, dim=1)
        elif 'max' in self.node_pooling_key:
            return torch.max(x, dim=1).values
        elif 'linear' in self.node_pooling_key:
            # channels are linearly-combined within nodes
            # x' = wx + b, using the same learned w and b parameters for all nodes
            # (N, d) @ (d, 1) -> (N, 1)
            return torch.matmul(x, self.node_pool_wts) + self.node_pool_bias
        else:
            raise NotImplementedError(
                "Node pooling method not implemented."
                "Did you mean 'mean', 'max', or 'linear'?"
            )


    def _apply_channel_pooling(self, x: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
        """
        Apply channel-level pooling for graph-level tasks.
        
        Args:
            x: Tensor of shape (N, d) where N is total nodes and d is number of channels
            batch_index: Tensor indicating which graph each node belongs to
            num_graphs: Number of graphs in the batch
            
        Returns:
            Pooled tensor of shape (num_graphs, ...)
        """
        if ('max' in self.channel_pool_key) or ('mean' in self.channel_pool_key):
            x_i_chan_pools_max = [None] * num_graphs
            x_i_chan_pools_mean = [None] * num_graphs
            
            for i in range(num_graphs):
                if num_graphs > 1:
                    x_i_mask = (batch_index == i)
                    x_i = x[x_i_mask]
                else:
                    x_i = x
                    
                if 'max' in self.channel_pool_key:
                    x_i_chan_pools_max[i] = torch.max(x_i, dim=0).values
                if 'mean' in self.channel_pool_key:
                    x_i_chan_pools_mean[i] = torch.mean(x_i, dim=0)

            if ('max' in self.channel_pool_key) and ('mean' not in self.channel_pool_key):
                return torch.stack(x_i_chan_pools_max)
            elif ('mean' in self.channel_pool_key) and ('max' not in self.channel_pool_key):
                return torch.stack(x_i_chan_pools_mean)
            else:
                maxs = torch.stack(x_i_chan_pools_max)
                means = torch.stack(x_i_chan_pools_mean)
                return torch.stack((maxs, means), dim=1)
                
        elif 'moment' in self.channel_pool_key and (self.channel_pool_moments is not None):
            q_norms = [None] * num_graphs
            for i in range(num_graphs):
                if num_graphs > 1:
                    x_i_mask = (batch_index == i)
                    x_i = x[x_i_mask]
                else:
                    x_i = x
                
                x_i_q_norms = torch.stack([
                    LA.vector_norm(
                        x=torch.abs(x_i), 
                        ord=q, 
                        dim=0
                    ) for q in self.channel_pool_moments
                ])
                q_norms[i] = x_i_q_norms
    
            return torch.stack(q_norms)
            
        return x


    def _apply_spectral_filtering(
            self, 
            x: torch.Tensor, 
            batch: Batch | Data, num_graphs: int
    ) -> torch.Tensor:
        """
        Apply spectral-based filtering.
        
        Args:
            x: Input tensor
            batch: Batch (Data) object containing graph information,
                including batch.L_eigenvals (shape (kappa,)) and 
                batch.L_eigenvecs (shape (n_nodes, kappa))
            num_graphs: Number of graphs in the batch
            
        Returns:
            Filtered tensor, of shape (n_nodes, n_channels, n_filters), 
            or 'Ncj'
        """
        if (num_graphs > 1) or (self.Wjs_spectral is None):
            batch_L_eigenvals = batch.L_eigenvals
            if batch_L_eigenvals.dim() == 1:
                batch_L_eigenvals = batch_L_eigenvals.unsqueeze(dim=0)

            if self.wavelet_type is not None:
                self.Wjs_spectral = torch.stack([
                    w.spectral_wavelets(
                        eigenvals=L_eigenvals,
                        J=self.J,
                        include_low_pass=self.include_lowpass_wavelet,
                        spectral_c=self.filter_c,
                        device=self.device
                    ) for L_eigenvals in batch_L_eigenvals
                ], dim=0) # shape (kappa, n_filters)
            elif self.non_wavelet_filter_type is not None:
                self.Wjs_spectral = torch.stack([
                    w.spectral_lowpass_filter(
                        eigenvals=L_eigenvals,
                        c=self.filter_c,
                        device=self.device
                    ) for L_eigenvals in batch_L_eigenvals
                ], dim=0) # shape (kappa, n_filters=1)
            else:
                raise NotImplementedError(
                    f"Non-wavelet spectral filter type {self.non_wavelet_filter_type}"
                    f" not implemented."
                )

        if (num_graphs > 1) or (self.V_sparse is None):
            self.V_sparse = get_Batch_V_sparse(
                num_graphs,
                batch.batch,
                batch.L_eigenvecs,
                self.max_kappa,
                device=self.device
            )
            
        return get_Batch_spectral_Wjxs(
            num_graphs,
            x,
            batch.batch,
            self.V_sparse,
            self.Wjs_spectral,
            batch.L_eigenvecs,
            self.max_kappa
        )


    def _apply_p_wavelet_filtering(
            self,
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            i: int
    ) -> torch.Tensor:
        """
        Apply P-wavelet based filtering.
        
        Args:
            x: Input tensor
            edge_index: Edge index tensor
            i: Current cycle index
            
        Returns:
            Filtered tensor
        """
        P_sparse = get_Batch_P_sparse(edge_index, device=self.device)
        
        if (self.wavelet_type is None) and (self.non_wavelet_filter_type == 'p'):
            get_Batch_P_Wjxs_kwargs = {
                'x': x,
                'P_sparse': P_sparse,
                'scales_type': None,
                'channels_t_is': None,
                'max_t': 1,
                'J': 1,
                'include_lowpass': False
            }
        else:
            channels_t_is = self.P_wavelets_channels_t_is if (i == 0) else None
            scales_type = 'handcrafted' if (channels_t_is is not None) else 'dyadic'
            get_Batch_P_Wjxs_kwargs = {
                'x': x,
                'P_sparse': P_sparse,
                'scales_type': scales_type,
                'channels_t_is': channels_t_is,
                'J': self.J,
                'include_lowpass': self.include_lowpass_wavelet
            }

        return get_Batch_P_Wjxs(**get_Batch_P_Wjxs_kwargs)

    def _initialize_mlp_head(self, x: torch.Tensor) -> None:
        """
        Initialize the MLP head if needed.
        
        Args:
            x: Input tensor to determine input dimension
        """
        if self.fc is None:
            self.fc = vnn.VanillaNN(
                input_dim=x.shape[1],
                **self.fc_kwargs
            )
            # since MLP head is initialized on first forward pass,
            # make sure it's on the correct device
            self.fc.to(x.device)


    def forward(self, batch: Batch | Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MFCN model.
        
        Args:
            batch: Batch or Data object containing graph information
            
        Returns:
            Dictionary containing model predictions
        """
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        num_graphs = 1 if batch_index is None else batch.num_graphs
        
        # Get number of cycles
        if self.cross_Wj_ns_combos_out_per_chan is None \
        and self.within_Wj_ns_chan_out_per_filter is None:
            n_cycles = 1  # Only perform filtering if no combinations are specified
        else:
            n_cycles = len(self.cross_Wj_ns_combos_out_per_chan) \
                if self.cross_Wj_ns_combos_out_per_chan is not None \
                else len(self.within_Wj_ns_chan_out_per_filter)

        # MFCN cycles
        for i in range(n_cycles):
            if self.verbosity > 1:
                print(f'\nMFCN cycle {i}')

            # (i) Filter step (skip if no filtering is desired)

            # Wavelet models ('mfcn' in model_key)
            if (self.wavelet_type is not None):
                if ('spect' in self.wavelet_type):
                    x = self._apply_spectral_filtering(x, batch, num_graphs)
                elif (self.wavelet_type == 'p'):
                    x = self._apply_p_wavelet_filtering(x, edge_index, i)

            # Low-pass filter models ('mcn' in model_key)
            elif (self.non_wavelet_filter_type is not None):
                if ('spect' in self.non_wavelet_filter_type):
                    x = self._apply_spectral_filtering(x, batch, num_graphs)
                elif (self.non_wavelet_filter_type == 'p'):
                    x = self._apply_p_wavelet_filtering(x, edge_index, i)

            if self.verbosity > 1:
                print(f'(after filter step) x.shape: {tuple(x.shape)}')

            # if x is sparse at this point, convert to dense
            if x.is_sparse:
                x = x.to_dense()

            # If x is 2D, add a filter dimension of 1
            # if x.ndim == 2:
            #     x = x.unsqueeze(dim=-1)
            #     if self.verbosity > 1:
            #         print(f'expanding x.shape to: {x.shape}')
                
            # (ii) Combine channels step
            if self.within_Wj_ns_chan_out_per_filter is not None:
                within_Wjs_wts = self.within_Wj_params[i]
                x = torch.einsum('Ncj,jcC->NCj', x, within_Wjs_wts)
                if self.verbosity > 1:
                    print(f'(after cross-channel/within-filter step) x.shape: {tuple(x.shape)}')

            # (iii) Cross-filter combinations step
            if self.cross_Wj_ns_combos_out_per_chan is not None:
                cross_Wjs_wts = self.cross_Wj_params[i]
                x = torch.einsum('NCj,CjJ->NJC', x, cross_Wjs_wts)
                if self.verbosity > 1:
                    print(f'(after cross-filter step) x.shape: {x.shape}')

            # (iv) Nonlinear activation
            x = self.mfcn_nonlin_fn(x, **self.mfcn_nonlin_fn_kwargs)

            # (v) Reshape into shape (batch_size, new_n_channels)
            x = x.reshape(x.shape[0], -1)
            if self.verbosity > 1:
                print(f'(after reshaping) x.shape: {tuple(x.shape)}')

        # Apply node pooling (if specified)
        x = self._apply_node_pooling(x)
        if self.verbosity > 1:
            print(f'(after node pooling) x.shape: {tuple(x.shape)}')

        # Handle task-specific processing
        if 'node' in self.task.lower():
            return {'preds': x}
        elif 'graph' in self.task.lower():
            # Apply channel pooling
            x = self._apply_channel_pooling(x, batch_index, num_graphs)
            
            if x.dim() == 3:
                x = x.reshape(x.shape[0], -1)

            # Initialize (if needed) and apply MLP head
            self._initialize_mlp_head(x)
            if self.fc is not None:
                if self.verbosity > 1:
                    print(f'(fc input) x.shape: {tuple(x.shape)}')
                    # print(f'x.device: {x.device}')
                    print()
                return self.fc.forward(x)
            return {'preds': x}
        else:
            raise NotImplementedError(
                "Pooling method not implemented in MFCN."
                " Did you forget 'node' or 'graph' in the 'task' arg?"
            )


class WaveletMFCNDataset(Dataset):
    """
    Subclass of `torch.utils.data.Dataset` that
    contains inputs and targets in dictionaries,
    for abstraction that allows for a generic PyTorch
    training function.

    Note that the MFCN model requires the graphs' Laplacian eigenvectors
    in its filtering steps, since it filters signals spectrally in 
    this Fourier domain (as a linear combination of Fourier coefficients
    of L's eigenvectors, using pre-made spectral filters based on L's
    eigenvalues).

    __init__ args:
        x: tensor for x/input, first dimension
            of which indexes into one sample/input:
            hence shape (n_samples, [n_channels], n_pts_per_sample)
        Ps: (sparse coo) tensor for 'P' (the lazy random walk matrix)
            for one manifold; shape (n_samples, n_samples).
        Wjs_spectral: tensor for spectral wavelet filters, 
            shape (n_samples, [n_channels], k, n_pts_per_sample).
        L_eigenvecs: tensor for each graph's Laplacian k
            eigenvectors, of shape (n_samples, [n_channels], k, n_pts_per_sample).
        targets_dictl: list of dictionaries holding
            target(s') keys and values.

    __getitem__ returns:
        A dictionary of containing one sample's x tensor, P or spectral
        filter tensors, and a sub-dictionary of its training target(s).
    """
    def __init__(self, 
                 wavelet_type: str,
                 x: torch.Tensor,
                 targets_dictl: List[Dict[str, Any]],
                 Ps: torch.Tensor = None, # could be sparse
                 Wjs_spectral: torch.Tensor = None,
                 L_eigenvecs: torch.Tensor = None,
                ) -> None:
        super(WaveletMFCNDataset, self).__init__()
        self.wavelet_type = wavelet_type
        self.x = x
        print(f'WaveletMFCNDataset: x.shape = {self.x.shape}')
        self.Ps = Ps
        self.Wjs_spectral = Wjs_spectral
        self.L_eigenvecs = L_eigenvecs
        self.targets_dictl = targets_dictl

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data_obj_dict = {
            'x': self.x[idx],
            'target': self.targets_dictl[idx]
        }
        if self.wavelet_type == 'P':
            data_obj_dict = data_obj_dict | {
                'P': self.Ps[idx]
            }
        elif self.wavelet_type == 'spectral':
            data_obj_dict = data_obj_dict | {
                'Wjs_spectral': self.Wjs_spectral[idx],
                'L_eigenvecs': self.L_eigenvecs[idx]
            }
        return data_obj_dict



def split_and_pickle_WaveletMFCNDataset_dict(
    args,
    x: torch.Tensor,
    target_dictl: List[dict],
    Ps: torch.sparse_coo_tensor = None,
    Wjs_spectral: torch.Tensor = None,
    L_eigenvecs: torch.Tensor = None,
    set_idxs_dict: Dict[str, List[int]] = None
) -> None:
    """
    Creates 'WaveletMFCNDataset' objects, splits into
    train/valid/test sets, and saves/pickles as a 
    dictionary.

    Args:
        args: ArgsTemplate subclass with experiment
            parameters.
        x: master input/signal/function values, where 
            the first axis indexes into one sample's 
            input tensor; shape (n_samples, n_input_vals).
        target_dictl: training targets dictionaries.
        Ps: master tensor of P (lazy random walk) matrices,
            where the first axis indexes into one sample's 
            P tensor; shape (n_samples, n_input_vals, 
            n_input_vals).
        Wjs_spectral: master tensor of spectral filters,
            where the first axis indexes into one sample's 
            spectral filters tensor; (n_samples, [n_channels], 
            n_eigenpairs, n_pts_per_sample).
        L_eigenvecs: master tensor of graph Laplacian eigen-
            vectors, where the first axis indexes into one 
            sample's spectral eigenvectors tensor; 
            shape (n_samples, n_input_vals, n_eigenvectors).
        set_idxs_dict: optional dictionary of index lists 
            for train/valid/test sets. If 'None', new
            index lists will be calculated.
    Returns:
        None (pickles dataset dict).
    """
    # get train/valid/test split idxs
    if set_idxs_dict is None:
        set_idxs_dict = du.get_train_valid_test_idxs(
            seed=args.TRAIN_VALID_TEST_SPLIT_SEED,
            n=x.shape[0],
            train_prop=args.TRAIN_PROP,
            valid_prop=args.VALID_PROP
        )

    if args.WAVELET_TYPE == 'P':
        datasets_dict = {
            set: WaveletMFCNDataset(
                wavelet_type=args.WAVELET_TYPE,
                x=x[idx],
                # 'index_select' works with torch.sparse
                Ps=torch.index_select(
                    input=Ps, 
                    dim=0, 
                    index=torch.tensor(idx, dtype=torch.long)
                ),
                # Ps=Ps[idx],
                targets_dictl=[target_dictl[i] for i in idx]
            ) \
            for set, idx in set_idxs_dict.items()
        }
    elif args.WAVELET_TYPE == 'spectral':
        datasets_dict = {
            set: WaveletMFCNDataset(
                wavelet_type=args.WAVELET_TYPE,
                x=x[idx],
                Wjs_spectral=Wjs_spectral[idx],
                L_eigenvecs=L_eigenvecs[idx],
                targets_dictl=[target_dictl[i] for i in idx]
            ) \
            for set, idx in set_idxs_dict.items()
        }
    
    # pickle the dataset dict
    save_path = f'{args.DATA_DIR}/{args.DATASETS_DICT_FILENAME}'
    with open(save_path, "wb") as f:
        pickle.dump(datasets_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Datasets saved (\'{args.DATASETS_DICT_FILENAME}\').\n')



def torch_sparse_identity(size):
    indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
    values = torch.ones(size)
    return torch.sparse_coo_tensor(indices, values, (size, size))



def get_Batch_P_sparse(
    edge_index: torch.Tensor, 
    edge_weight: Optional[torch.Tensor] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    r"""
    Computes P, the diffusion operator on 
    a graph defined as $P = 0.5 (I - AD^{-1})$,
    where the graph is the disconnected batch
    graph of a torch_geometric Batch object.

    Args:
        edge_index: edge_index (e.g., from a
            pytorch_geometric Batch object).
        edge_weight: edge_weight (e.g., from a
            pytorch_geometric Batch object).
        device: string device key (e.g., 'cpu', 'cuda', 
            'mps') for placing the output tensor; if
            None, will check for cuda, else assign to cpu.
    Returns:
        Sparse P matrix tensor, of shape 
        (N, N), where N = data.x.shape[0], 
        the  total number of nodes across all
        batched graphs. Note P_sparse is 'doubly
        sparse': sparse off of block diagonals,
        where each block is itself a sparse operator
        P_i for each graph x_i.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    A_sparse = to_torch_coo_tensor(
        edge_index, 
        edge_weight
    ).to(device)
    D = A_sparse.sum(dim=1).to_dense()
    # as of Oct 2024, 'torch.sparse.spdiags' doesn't work on cuda 12.4,
    # -> use function with cpu tensors, then move resulting sparse
    # tensor to device
    D_inv = torch.sparse.spdiags(
        diagonals=(1. / D.squeeze()).to('cpu'), 
        offsets=torch.zeros(1).long().to('cpu'),
        shape=(len(D), len(D))
    ).to(device)
    I = torch_sparse_identity(len(D)).to(device)
    P_sparse = 0.5 * (I + torch.sparse.mm(A_sparse, D_inv)) # .to(device)
    return P_sparse



def get_Batch_P_Wjxs(
    x: Batch,
    P_sparse: torch.Tensor,
    scales_type: str = 'dyadic',
    channels_t_is: Optional[torch.Tensor] = None,
    max_t: int = 32,
    J: int = 4,
    include_lowpass: bool = True,
    filter_stack_dim: int = -1
) -> torch.Tensor:
    r"""
    Computes P (diffusion) wavelet filtrations
    on a disconnected graph, using recursive
    sparse matrix multiplication. That is,
    skips computing increasingly dense powers of P, 
    by these steps:
    
    1. Compute $y_t = P^t x$ recursively via $y_t = P y_{t-1}$,
       (only using P, and not its powers, which grow denser).
    2. Subtract $y_{2^{j-1}} - y_{2^{j}}$ [dyadic scales]. 
        The result is $W_j x = (P^{2^{j-1}} - P^{2^j}) x$.
        (Thus, we never form the matrices P^t, t > 1, which get 
        denser with as the power increases.)
    
    Args:
        x: stacked node-by-channel (N, c) data matrix for a 
            disconnected  batch graph of a pytorch geometric 
            Batch object. 
        P_sparse: sparse diffusion operator matrix 
            for disconnected batch graph of a pytorch
            geometric Batch object (output of 
            'get_Batch_P_sparse()').
        scales_type: 'dyadic' or 'handcrafted' or None for fixed P^1.
        channels_t_is: tensor of shape (n_channels, n_scale_split_ts)
            for calculating 'handcrafted' wavelet scales, containing the 
            indices of ts 0...max($t$). Scales are constructed uniquely
            for each channel of x from $t$s with adjacent indices in rows 
            of this tensor. If None, this function defaults to dyadic 
            scales.
        max_t: maximum power of $P$ to compute in $P^t$, for manual 
            scales.
        J: max wavelet filter order, for dyadic scales.
        include_lowpass: whether to include the 
            'lowpass' filtration, $P^{2^J} x$.
        filter_stack_dim: new dimension in which to 
           stack Wjx (filtration) tensors.
    Returns:
        Dense tensor of shape (batch_total_nodes, n_channels,
        n_filtrations) = 'Ncj'.
    """

    # print('x.device:', x.device)
    # print('Ptx.device:', Ptx.device)
    # print('P_sparse.device:', P_sparse.device)

    # dyadic scales
    if (scales_type == 'dyadic') and (channels_t_is is None):
        Ptxs = [x]
        Ptx = x
        
        # calc P^t x for t \in 1...2^J, saving ts in
        # powers of 2
        powers_to_save = 2 ** torch.arange(J + 1)
        for j in range(1, max(powers_to_save) + 1):
            Ptx = torch.sparse.mm(P_sparse, Ptx)
            if j in powers_to_save:
                Ptxs.append(Ptx)
                
        Wjxs = [Ptxs[j - 1] - Ptxs[j] for j in range(1, J + 2)]
        if include_lowpass:
            Wjxs.append(Ptxs[-1])
        Wjxs = torch.stack(Wjxs, dim=filter_stack_dim)

    # 'handcrafted' scales
    elif (scales_type == 'handcrafted') and (channels_t_is is not None):
        Ptxs = [x.to_dense()]
        Ptx = x
        
        # calc P^t x for t \in 1...T, saving all powers of t
        for j in range(1, max_t + 1):
            Ptx = torch.sparse.mm(P_sparse, Ptx)
            # print('Ptx.device:', Ptx.device)
            Ptxs.append(Ptx.to_dense())

        device = Ptxs[0].device
                
        Wjxs = torch.stack([
            torch.stack([
                # as of Nov 2024, bracket slicing doesn't work with sparse tensors
                # patch: entries of 'Ptxs' made dense above, when added to Ptxs
                Ptxs[t_is[t_i - 1]][:, c_i] - Ptxs[t_is[t_i]][:, c_i] \
                for t_i in range(1, len(t_is))
            ], dim=-1) \
            for c_i, t_is in enumerate(channels_t_is)
        ], dim=1) 
        
        # lowpass = P^T x, for all channels
        if include_lowpass:
            # print('Ptxs[-1].shape:', Ptxs[-1].shape)
            Wjxs = torch.concatenate(
                (Wjxs, Ptxs[-1].unsqueeze(dim=-1)), 
                dim=-1
            )
        # Wjxs shape (N, n_channels, n_filters)
        # print('Wjxs.shape:', Wjxs.shape)

    elif scales_type is None:
        Ptx = x
        Ptx = torch.sparse.mm(P_sparse, Ptx)
        Wjxs = Ptx.unsqueeze(dim=-1)
    else:
        raise NotImplementedError(f"No method implemented for scales_type={scales_type}")
        
    return Wjxs


def handcrafted_P_wavelet_scales(
    pyg_train_set: Dict | Data,
    task: str,
    device: str,
    T: int = 32,
    cmltv_kld_quantiles: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875), # [0.2, 0.4, 0.6, 0.8],
    include_lowpass: bool = True,
    start_from_t2: bool = True,
    reweight_klds: bool = True,
    savepath_kld_by_channel_plot: Optional[str] = None,
    kld_by_channel_plot_name: str = "handcraft_P_wav_scales_plot",
    verbosity: int = 0
) -> torch.Tensor:
    r"""

    TODO [ideas]
    [ ] parallelize better
    [ ] ensure no two t cutoffs are the same index? right now it
        can happen, which effectively drops a wavelet: (P^3 - P^3)x = 0,
        but this means features extracted by wavelets reflect the same
        infogain quantile, where some channels have more than that quantile's
        worth of infogain (the t diffusion steps aren't fine enough)
    [ ] outlier control in KLD loss calcs?
    [ ] some form of regression target 'imbalance' correction?
    
    Calculates the scales for 'handcrafted' wavelets (non-dyadic,
    with unique scales for each channel of signal on graphs); the
    output 'channels_t_is' (calculated once over all training data) 
    can then be used in 'get_Batch_P_Wjxs'.

    Give a diffusion operator $P$ and powers $P^t$ for $t \in 0...T$,
    and a graph channel signal $x$, we take normalized $P^T x$ as our 
    reference distribution, and calculate relative entropy (KL divergence/
    information gain) of each $P^t x$ versus this reference distribution.
    We then select P-wavelet scales based on $t$ cutoffs uniquely for
    each channel, based on which powers of $t$ cross the cumulative
    KL divergence thresholds passed in the 'cmltv_kld_quantiles' argument. 
    (Here, we know that relative entropies relative to $P^T x$ decrease
    with increasing powers of $t$, so each channel has a slowing cumulative
    sum; if 'start_from_t2' is true, we also automatically keep t <= 2 as
    scale cutoffs, as the greatest values of relative entropy are expected
    in these lowest powers, and corresponding wavelets should be kept by 
    default).

    Thus, instead of dyadic-scale wavelets (where $W_j x = P^{2^{j-1}} 
    - P^{2^j}) x$), we obtain, for example, wavelets unique to each channel,
    such as (P^3 - P^5)x in one, but (P^4 - P^7)x in another, with both
    capturing the same volume change in relative entropy against their 
    channel's steady state diffusion (P^T x) at the wavelet index.
    
    Args:
        pyg_train_set: either dictionary with 'train' keying a 
            pytorch geometric DataLoader object 
            containing the test set graphs in batches 
            (multiple graphs); or single-graph pytorch
            geometric Data object with a 'train' mask.
        task: string description of the modeling task, e.g.,
            'binary_classification'.
        device: string key for the torch device, e.g. 'cuda'.
        T: max power of $P$, for $P^t$ where $t \in 1...T$.
        cmltv_kld_quantiles: iterable of cumulative KLD 
            quantiles/percentile cutoffs, which powers of P
            must reach to be a wavelet scale boundary $P^t$.
        include_lowpass: boolean, whether to include the 
            lowpass wavelet $P^T$.
        start_from_t2: boolean whether to keep filters with 
            $P^1$ and $P^2$, and choose subsequent scales
            (and ignore their contribution to cumulative KLD;
            calc from $P^3...P^T$ instead). This is useful since
            these lowest powers of $t$ generally cover the largest
            steps in KLD, and perhaps should be included scale
            steps in all channels by default.
        reweight_klds: boolean whether to re-weight each graph's
            contribution to a channels' total (sum) KLD loss, e.g. 
            to rebalance KLD for unbalanced target classes.
        savepath_kld_by_channel_plot: optional save path for a plot
            of cumulative KLDs by channel.
        kld_by_channel_plot_name: filename (.png added automatically)
            for the optional KLD by channel plot.
        verbosity: integer controlling volume of print output as
            the function runs.
    Returns:
        Torch tensor containing indices of $t$s (which also happen to
        be their values in $P^t x, t \in 0...T$) for each channel in
        the graph dataset; shape (n_channels, n_ts).
    """
    # loop through batches in train set again, collecting KL divergence stats
    # for the entire train set
    klds_by_x_t_chan = []
    targets_by_xi = []
    
    if isinstance(pyg_train_set, dict):
        pyg_train_set = pyg_train_set['train']
        multiple_graphs = True
    elif isinstance(pyg_train_set, Data):
        # make iterable of 1 batch of 1 graph
        pyg_train_set = (pyg_train_set, )
        multiple_graphs = False
    
    for batch in pyg_train_set:
        batch = batch.to(device)
        x = batch.x #.to_dense()
        n_channels = batch.x.shape[1]
        edge_index = batch.edge_index    
        
        if multiple_graphs:
            num_graphs = batch.num_graphs
            targets_by_xi.extend(batch.y)
            # batch_index = batch.batch
        else: # 1 graph in single Data object
            # note that in a node-level graph task,
            # we don't mask any node signals for train vs. valid
            # set until loss calculation / evaluation time
            num_graphs = 1
            targets_by_xi.extend(batch.y[batch.train_mask])
            
        # get P_sparse
        P_sparse = get_Batch_P_sparse(
            edge_index, 
            device=device
        )
        
        # calc P^t x for t \in 1...T
        # make each Ptx in list dense here so tensor slicing below works
        Ptx = x
        Ptxs = [x.to_dense()] # densify AFTER copying sparse x for recursive mm
        for j in range(1, T + 1):
            Ptx = torch.sparse.mm(P_sparse, Ptx)
            Ptxs.append(Ptx.to_dense())
        
        # calc KL divergence of x and each P^t x versus baseline P^T x (lowpass/smoothest),
        # uniquely for each x_i (graph) and channel
        for i in range(num_graphs):
            
            if multiple_graphs:
                # subset out ith graph
                x_i_mask = (batch.batch == i)
                Ptx_is = [ptx[x_i_mask] for ptx in Ptxs] # each y_i has shape (n_i, c)
            else: # one graph
                Ptx_is = Ptxs

            if verbosity > 0:
                print('len(Ptx_is):', len(Ptx_is))
                print('Ptx_is[0].shape:', Ptx_is[0].shape)
            
            '''
            Notes on KL divergence / relative entropy: 
            - if any entry in a channel is <=0, a NaN will be created
            by log, and that NaN is then part of a sum -> sum = NaN
            - thus we normalize each P^t x into a probability vector
            first (i.e. with range 0-1 and sum = 1), and this prevents
            zeros, since KLD can't handle them.
            - we also prevent skewing relative KLDs by replacing zeros 
            with too tiny of a value by replacing zeros with the value
            halfway between the (pre-normalized) minimum channel value
            and second-lowest channel value
            '''
            # calc KLD(P^t x, P^T x) for t \in 1...(T-1), by channel
            # (T-1) -> excludes KLD(PTx_i, PTx_i)
            Ptx_i_start = 2 if start_from_t2 else 0
            PTx_i = Ptx_is[-1]
            channel_klds = [None] * n_channels
            
            for c in range(n_channels):
                # for each P^t x_i within the same channel, the reference
                # distribution P^T x_i is the same -> calc it once per channel
                PTx_ic = PTx_i[:, c]
                T_above_zero_floor = nnu.get_mid_btw_min_and_2nd_low_vector_vals(PTx_ic)
                if T_above_zero_floor <= 0:
                    T_above_zero_floor = -T_above_zero_floor
                PTx_ic = nnu.norm_1d_tensor_to_prob_mass(
                    PTx_ic, 
                    above_zero_floor=T_above_zero_floor
                )
                kld_all_ts_one_chan = [None] * (len(Ptx_is) - Ptx_i_start - 1)
                for t, Ptx_i in enumerate(Ptx_is[Ptx_i_start:-1]):
                    Ptx_ic = Ptx_i[:, c]
                    t_above_zero_floor = nnu.get_mid_btw_min_and_2nd_low_vector_vals(Ptx_ic)
                    if t_above_zero_floor <= 0:
                        t_above_zero_floor = -t_above_zero_floor
                    Ptx_ic = nnu.norm_1d_tensor_to_prob_mass(
                        Ptx_ic, 
                        above_zero_floor=t_above_zero_floor
                    )
                    kld_all_ts_one_chan[t] = (Ptx_ic * (Ptx_ic / PTx_ic).log()).sum() # scalar
                # after KLD for all powers of t for one channel have been calculated -> 
                # tensorize list and add to 'channel_klds' list, which has length n_channels,
                # and each element is a tensor of shape (num_t_powers, )
                channel_klds[c] = torch.tensor(kld_all_ts_one_chan) 

            # after processing KLDs for all ts and all channels, stack into tensor
            # of shape (num_t_powers, n_channels)
            channel_klds = torch.stack(channel_klds, dim=-1) 
            # append 'kld_all_ts_by_chan' for each graph in each batch to 'klds_by_x_t_chan',
            # a growing list of eventual length n_graphs, where each list element is a tensor 
            # of shape (num_t_powers, n_channels)
            klds_by_x_t_chan.append(channel_klds) 

    # after all graphs in all batches:
    # stack KLD values for all graphs x t powers x channels into a tensor
    klds_by_x_t_chan = torch.stack(klds_by_x_t_chan) # shape (n_graphs, num_t_powers, n_channels)
    if verbosity > 0:
        print('klds_by_x_t_chan.shape:', klds_by_x_t_chan.shape)
        print('klds_by_x_t_chan:\n', klds_by_x_t_chan)
    targets_by_xi = torch.stack(targets_by_xi) # shape (n_graphs, )
    
    # replace NaNs created (0s in log of KLD) with (max KLD value w/in same channel)
    kld_chan_maxs = nanmax(
        nanmax(klds_by_x_t_chan.numpy(), axis=1),
        axis=0
    )
    # kld_chan_maxs.shape # (n_channel, )
    
    for x_klds in klds_by_x_t_chan:
        for t_klds in x_klds:
            nan_mask = torch.isnan(t_klds)
            t_klds[nan_mask] = torch.tensor(kld_chan_maxs[nan_mask])

    
    # quantify relative KLD over all graphs, by t and channel
    
    # re-weight klds for target class balance
    if reweight_klds:
        # 0s get weight 1, 1s get relative weight 'pos_class_wt'
        if 'bin' in task.lower() and 'class' in task.lower():
            ct_1s = targets_by_xi.sum()
            pos_class_wt = (targets_by_xi.shape[0] - ct_1s) / ct_1s
            kld_weights = torch.ones(len(targets_by_xi))
            kld_weights[targets_by_xi == 1] = pos_class_wt
        else:
            # raise NotImplementedError()
            warnings.warn(f"Reweighting KLDs not implemented for task='{task}'.")
            kld_weights = None
    
        # reweight KLDs
        if kld_weights is not None:
            klds_by_x_t_chan = torch.einsum(
                'bTc,b->bTc',
                klds_by_x_t_chan,
                kld_weights
            )
        
    # sum (reweighted) KLD across graphs, by t and channel
    klds_by_t_chan = torch.sum(klds_by_x_t_chan, dim=0) # shape (T, n_channels)
    
    # get cumulative sums as t increases
    klds_cum_by_t_chan = torch.cumsum(klds_by_t_chan, dim=0) # shape (T, n_channels)
    
    # minmax scale channel cumulative KLDs, for cross-channel comparison (i.e.
    # so all channels' cumulative KLDs range from 0 to 1)
    for c in range(n_channels):
        # klds_cum_by_t_chan[:, c] = (klds_cum_by_t_chan[:, c] - min_kld) / (max_kld - min_kld)
        klds_cum_by_t_chan[:, c] = nnu.minmax_scale_1d_tensor(
            v=klds_cum_by_t_chan[:, c], 
            min_v=klds_cum_by_t_chan[:, c][0], # min cmltv kld is at start
            max_v=klds_cum_by_t_chan[:, c][-1] # max cmltv kld is at end
        )        

    # optional check: plot 'klds_cum_by_t_chan'
    if savepath_kld_by_channel_plot is not None:
        import matplotlib.pyplot as plt
        for c in range(n_channels):
            plt.plot(
                range(Ptx_i_start, T), 
                klds_cum_by_t_chan[:, c].numpy()
            )
        plt.title(
            f"Normalized cumulative (all-node) sums of KL divergences of $P^t x$"
            f"\nfor $t \in {Ptx_i_start}...(T-1)$ from $P^T x$, by channel"
        )
        plt.xlabel('$t$ in $P^t x$')
        # assuming T is a power of 2, make x ticks powers of 2
        plt.xticks([0] + [2 ** p for p in range(0, int(log2(T)) + 1)])
        # plt.xticks(range(0, T + 1, 4))
        plt.yticks(linspace(0, 1, 11))
        plt.ylabel('cmltv KLD from $P^T x$')
        plt.grid()
        # plt.show()
        os.makedirs(savepath_kld_by_channel_plot, exist_ok=True) 
        plt.savefig(f"{savepath_kld_by_channel_plot}/{kld_by_channel_plot_name}.png")
        plt.clf()
        
    # for each channel, find (indexes of) t-integer scale cutoffs, 
    # following the quantiles of cmltv KLD in 'cmltv_kld_quantiles'
    channels_t_is = torch.stack([
        torch.stack([
            torch.argwhere(klds_cum_by_t_chan[:, c] >= q)[0].squeeze() \
            # adjust t indexes returned for the 'Ptx_i_start' index
            + Ptx_i_start \
            for q in cmltv_kld_quantiles
        ]) \
        for c in range(n_channels)
    ]) # shape (n_channels, n_quantiles)
    
    # calc wavelet filters, by (P^t - P^u)x = (P^t)x - (P^u)x
    # uniquely for each channel, following 'channels_t_is'
    # (all P^ts and P^us calc'd above for all xs and channels:
    # just need to subtract using specific ts and us for each channel)
    if start_from_t2:
       channels_t_is = torch.concatenate((
            torch.stack((
                torch.zeros(n_channels), 
                torch.ones(n_channels), 
                torch.ones(n_channels) * 2
            ), dim=-1),
            channels_t_is,
            (torch.ones(n_channels) * T).unsqueeze(dim=1)
        ), dim=-1).to(torch.long)

    if verbosity > 0:
        print('channels_t_is.shape:', channels_t_is.shape) # shape (n_channels, n_quantiles)
        print('channels_t_is\n', channels_t_is)
    return channels_t_is



def get_Batch_V_sparse(
    batch_size: int,
    batch_index: torch.Tensor,
    L_eigenvecs: torch.Tensor,
    max_kappa: Optional[int] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    r"""
    Computes 

    Args:
        batch_size: number of graphs in the
            batch.
        batch_index: batch_index (e.g., from a
            pytorch_geometric Batch object). Can be
            None for a batch with 1 graph.
        L_eigenvecs: 2-d tensor holding stacked
            eigenvectors of the graph Laplacians 
            for each graph x_i; shape (N, k) =
            (total_n_nodes, n_eigenvectors).
        max_kappa: maximum number of eigenvectors to
            utilize, up to the number stored in 
            'L_eigenvecs'.
        device: string device key (e.g., 'cpu', 'cuda', 
            'mps') for placing the output tensor; if
            None, will check for cuda, else assign to cpu.
    Returns:
        Sparse block-diagonal matrix V, where blocks
        hold eigenvectors for each graph in the batch, hence
        shape (bk, N), where b is the batch size (number of
        graphs), k is the number of eigenvectors, and N is
        the total number of nodes across all graphs.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    values = []
    n_ctr = 0
    k_ctr = 0
    n_indices = []
    k_indices = []

    for i in range(batch_size):
        if batch_size > 1:
            x_i_mask = (batch_index == i)
            v = L_eigenvecs[x_i_mask] # shape (n_i, k)
        else:
            v = L_eigenvecs
            
        if max_kappa is not None:
            v = v[:, :max_kappa]
        # need to ravel individual v, not entire 'L_eigenvecs'
        values.extend(v.ravel())
        v_indices = v.nonzero() # shape (n_i * k, 2)
        
        n_indices.extend(v_indices[:, 0] + n_ctr)
        k_indices.extend(v_indices[:, 1] + k_ctr)
        
        n_ctr += v.shape[0]
        k_ctr += v.shape[1]

    # generate indices such that shape is (bk, N)
    indices = torch.stack((
        torch.tensor(k_indices),
        torch.tensor(n_indices)
    ))
    
    size = (
        k_ctr, # bk
        L_eigenvecs.shape[0] # N
    )
    
    V_sparse = torch.sparse_coo_tensor(
        indices=indices,
        values=torch.tensor(values),
        size=size,
        dtype=torch.float
    ).to(device)
    
    return V_sparse



def get_Batch_spectral_Wjxs(
    num_graphs: int,
    x: torch.Tensor,
    batch_index: torch.Tensor,
    V_sparse: torch.Tensor,
    Wjs_spectral: torch.Tensor,
    L_eigenvecs: torch.Tensor,
    max_kappa: Optional[int] = None
) -> torch.Tensor:
    r"""
    Computes spectral filtration values for a pytorch
    geometric batch of graphs with multiple channels,
    for multiple (precomputed scalar) filters in the 
    Fourier domain, defined by:
    $\sum_{i=1}^{k} w_j(\lambda_i) \hat{f}(i) \psi_i$
    for $1 \leq j \leq J$. For details, see the MFCN
    manuscript. Note $w_j(\lambda_i)$ and $\hat{f}(i)$
    are scalars.

    Args:
        num_graphs: number of graphs in the pytorch geometric 
            Batch. 
        x: stacked node-by-channel (N, c) data matrix for a 
            disconnected  batch graph of a pytorch geometric 
            Batch object. 
        batch_index: 'batch' (index set) from the pytorch 
            geometric Batch. Optional for batches with 1 graph.
        V_sparse: sparse block-diagonal 2d tensor where blocks are
            (k, n_i) matrices of each graph's eigenvectors; the
            output of 'get_Batch_V_sparse()'.
        Wjs_spectral: tensor of pre-computed spectral filter scalars;
            shape (kappa, n_filters) [1 graph; will be unsqueezed] or 
            (n_graphs, kappa, n_filters) [1 or batched graphs].
        L_eigenvecs: tensor of pre-computed eigenvalues for all graphs
            in batch stacked node-wise -> shape 
            (total_n_nodes, n_eigenvectors) = (N, k).
        max_kappa: maximum number of eigenvectors to
            utilize, up to the number stored in 
            'L_eigenvecs'.
    Returns:
        Tensor of spectrally filtered signal values, of 
        shape (total_n_nodes, n_channels, n_filters) = (N, c, j).
        Note FWV[0:n_0, 0, 0] selects the spectral convolution for
        x_0, channel_0, filter_0, which has length n_0 (n_nodes in 
        first graph).
    """
    
    # compute Fourier coefficients of each channel signal
    # on each graph, wrt k graph-Laplacian eigenvectors
    # this is done by taking the dot product of each graph's
    # eigenvectors with the channel's signal values
    # [block-diagonal] V_sparse @ x -> (bk=K, N) @ (N, c) -> (K, c)
    # -> F shape: (num_graphs * n_eigenvectors, n_channels) = 'Kc'
    # note this gets rid of 'N' and standardizes all graphs of 
    # possibly different sizes/num_nodes into the same Fourier space size
    F = torch.sparse.mm(V_sparse, x)
    # note: if x is also sparse, F will be sparse

    # PATCH: as of Oct 2024, torch sparse tensors can't use 'tensor_split',
    # so we need to convert to dense first
    F = F.to_dense()

    # split and stack F into shape 'bkc' -> inner matrix
    # cols are a graph's k Fourier coeffs for each channel
    F = torch.stack(
        torch.tensor_split(F, num_graphs, dim=0),
        dim=0
    )
    
    # 'Wjs_spectral' (stored in DataBatch object) needs shape
    # 'bkj' -> inner row is a graph's eigenpair's W(lambda_i)
    # j filter scalar values
    # -> this einsum produces FW, a 4-d tensor where each
    # inner row is a graph's channel's j filters, times 
    # each of the k Fourier coeffs -> size 'bcjk'
    # (that is, FW holds all of the $w_j(\lambda_i) \cdot \hat{f}(i)$
    # scalar combinations)

    # Wjs_spectral.shape (kappa, n_filters) [1 graph] or 
    # (n_graphs, kappa, n_filters) [batched graphs]
    # L_eigenvecs.shape (total_n_nodes, kappa) [1 graph or (stacked) batched graphs]
    if Wjs_spectral.dim() == 2:
        # have 1 graph in batch; unsqueeze to have n_graphs=1 at dim 0
        Wjs_spectral = Wjs_spectral.unsqueeze(dim=0)
        # L_eigenvecs = L_eigenvecs.unsqueeze(dim=0)
    # print('Wjs_spectral.shape:', Wjs_spectral.shape) 
    # print('L_eigenvecs.shape:', L_eigenvecs.shape) 

    # trim spectral objects to use max_kappa eigenpairs
    if max_kappa is not None:
        Wjs_spectral = Wjs_spectral[:, :max_kappa, :]
        L_eigenvecs = L_eigenvecs[:, :max_kappa]

    # calc FW -> each row_j has k Four. coeffs for filter w_j
    # (for each graph in the batch)
    FW = torch.einsum('bkc,bkj->bcjk', F, Wjs_spectral)
    # print('FW.shape:', FW.shape) 

    # 'L_eigenvecs' (stored in DataBatch object) has shape
    # (sum(n_i), k): eigenvectors of all graphs are stacked
    # vertically in blocks (like batch.x data: in case graphs
    # have different n_i, and indexable by batch.batch)
    fwvs = [None] * num_graphs
    for i in range(num_graphs):

        if num_graphs > 1:
            mask = (batch_index == i)
            # subset one graph's eigenvectors -> shape (k, n_i) (after 
            # the transpose); k rows are e-vecs of length n_i
            v = L_eigenvecs[mask].T
        else:
            v = L_eigenvecs.T

        # subset one graph's fw -> shape (c, j, k), where each 
        # row_j has k Four. coeffs for filter w_j
        fw = FW[i]
        
        # this einsum calculates the spectral convolutions for one graph
        # by multiplying e-vec $\psi_i$s by the 
        # $w_j(\lambda_i) \cdot \hat{f}(i)$ scalar product, and summing 
        # e-vecs 1...k
            # if '->cjn': each row_j is a channel's convolution using 
            # filter W_j (linear combo of eigenvectors of length n_i), 
            # all for one graph
            # if '->ncj': each element is one node's convolution value
            # for one channel and one filter, all for one graph
        fwv = torch.einsum('cjk,kn->ncj', fw, v)
        fwvs[i] = fwv

    # concat. all graphs' fwvs along first (node) axis 
    # -> shape (N, c, j), where N = sum(n_i)
    FWV = torch.concatenate(fwvs)
    return FWV

