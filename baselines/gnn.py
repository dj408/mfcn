"""
Combines this code repo's BaseModule with pytorch-geometric
modules for GCN, GIN, GAT, and GraphSAGE, for testing these
GNNs as baseline methods.

Ref:
https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models
"""
import sys
sys.path.insert(0, '../')
import base_module as bm
import vanilla_nn as vnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset,
    Batch
)
from torch_geometric.nn import (
    GCN,
    GAT,
    GraphSAGE,
    GIN
)
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool
)
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional
)


class GNN_FC(bm.BaseModule):
    """
    Implements a GCN, GAT, GraphSAGE, or GIN model
    using the PyTorch Geometric library, whose outputs
    are concatenated after max and mean pooling, 
    and then (for graph-level tasks) fed into a  
    regressor or classifier vanilla NN head.

    __init__ args:
        gnn_type: string key value for which GNN to construct.
            Options: gcn, gin, gat, sage.
        in_channels: int value of the number of original feature
            input channels. If passed -1, the pytorch-geometric
            model class will automatically infer from the first training
            batch.
        hidden_channels: number of output channels of a convolutional cycle
            of the GNN.
        num_layers: number of convolutional cycles to apply (default: 2).
        out_channels: if not None, will apply a
            final linear transformation to convert 
            hidden node embeddings to this arg's value
            [see pytorch geometric's documentation for, 
            e.g. GCN: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html]
        dropout_p: if nonzero, probability of zeroing out a node in a
            GNN layer.
        activ_fn: torch.nn.Module activation function to apply (default: ReLU).
        jk: optional jumping knowledge arg to pass to the GNN module.
        channel_pool_key: string key for the type of final channel pooling to 
            apply. Options: 'max', 'mean', 'mean and max' (in which case, mean
            and max poolins are concatenated). Useful in graph-level learning
            tasks.
        model_specific_kwargs: optional kwargs to pass to the GNN module
            specified.
        base_module_kwargs: optional kwargs to pass to BaseModule parent class.
        fc_kwargs: optional kwargs to pass to a fully-connected network head.
            If None, no fc head is created or utilized.
        verbosity: integer value controlling print output as methods of this
            class run.
    """

    def __init__(
        self,
        gnn_type: str = 'gcn',
        in_channels: int = -1,
        hidden_channels: int = 64,
        num_layers: int = 2,
        out_channels: Optional[int] = None,
        dropout_p: float = 0.,
        activ_fn: str = 'relu',
        jk: Optional[str] = None, # 'max', # 'pooling aggregator': cf. p. 6 of orig. GCN paper
        channel_pool_key: str = 'max',
        model_specific_kwargs: Optional[Dict[str, Any]] = None,
        base_module_kwargs: Dict[str, Any] = {},
        fc_kwargs: Optional[Dict[str, Any]] = None,
        verbosity: int = 0
    ):
        super(GNN_FC, self).__init__(**base_module_kwargs)
        self.gnn_type = gnn_type.lower()
        self.channel_pool_key = channel_pool_key
        self.verbosity = verbosity
        
        """
        [optional] initialize fully-connected network head
        """
        if (fc_kwargs is not None):
            pool_mult = 0
            if ('max' in self.channel_pool_key):
                pool_mult +=1 
            if ('mean' in self.channel_pool_key):
                pool_mult +=1 
            
            # fc_input_dim w/ pooling: e.g. graph-level task
            # we multiply hidden_channels by 2 for global mean+max 
            # pooling of each channel (after second conv.)
            fc_input_dim = hidden_channels * pool_mult
            self.fc = vnn.VanillaNN(
                input_dim=fc_input_dim,
                **fc_kwargs
            )
        else:
            self.fc = None
        # print(f'fc_input_dim: {fc_input_dim}')
            
        """
        initialize specific GNN model
        """
        gnn_kwargs = {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'out_channels': out_channels,
            'dropout': dropout_p,
            'act': activ_fn
        } 
        if model_specific_kwargs is not None:
            gnn_kwargs = gnn_kwargs | model_specific_kwargs
        
        if 'sage' in self.gnn_type:
            # print('Initializing GraphSAGE model...')
            self.gnn = GraphSAGE(**gnn_kwargs)
        elif 'gin' in self.gnn_type:
            # print('Initializing GIN model...')
            self.gnn = GIN(**gnn_kwargs)
        elif 'gat' in self.gnn_type:
            # print('Initializing GAT model...')
            self.gnn = GAT(**gnn_kwargs)
        elif 'gcn' in self.gnn_type:
            # print('Initializing GCN model...')
            self.gnn = GCN(**gnn_kwargs)
        # print('\tDone.')

    
    def forward(
        self, 
        batch: Batch | Data,
    ) -> Dict[str, torch.Tensor]:
        
        # extract batch graph objects
        x, edge_index, batch_index = (
            batch.x, 
            batch.edge_index, 
            batch.batch
        )

        # patch to fix SAGE and GIN error with sparse batch.x tensor
        # 'NotImplementedError: Could not run 'aten::fill_.Scalar' with 
        # arguments from the 'SparseCUDA' backend'
        if ('sage' in self.gnn_type) \
        or ('gin' in self.gnn_type):
            x = x.to_dense()
        
        # print('x.device:', x.device)
        if batch_index is None:
            batch_index = torch.ones(
                batch.num_nodes,
                dtype=torch.long
            )
            num_graphs = 1
        else:
            num_graphs = batch.num_graphs
        
        x = self.gnn(x, edge_index)
        # here, x shape = (total_n_nodes, hidden_channels) or
        # (total_n_nodes, out_channels) if out_channels is not None
        # print(f'GNN output x.shape: {x.shape}')

        # if graph-level task w/ MLP regressor/classifier head:
        # globally-pool the GNN's output channels and feed to head
        if ('graph' in self.task) \
        and (self.fc is not None) \
        and self.channel_pool_key is not None:
            if ('max' in self.channel_pool_key):
                x_maxs = global_max_pool(x, batch_index)
            if ('mean' in self.channel_pool_key):
                x_means = global_mean_pool(x, batch_index)

            if ('mean' in self.channel_pool_key) \
            and ('max' in self.channel_pool_key):
                x = torch.cat((x_means, x_maxs), dim=1)
            elif ('mean' in self.channel_pool_key):
                x = x_means
            elif ('max' in self.channel_pool_key):
                x = x_maxs
            else:
                raise NotImplementedError(
                    f"Channel pooling option '{self.channel_pool_key}' not"
                    f" implemented for GNN_FC."
                )
                
            # print(f'x.shape after mean+max pooling: {x.shape}') 
            # should be: (n_graphs_in_batch, 2*output_shape[1])
            output_dict = self.fc.forward(x.squeeze())

        # if node-level task: 
        # locally pool channels at each node
        elif 'node' in self.task:
            output_dict = {'preds': x}
        else:
            raise NotImplementedError(
                f"GNN not implemented for this task. Did"
                f" you forget to include 'graph' or 'node' in"
                f" the 'task' arg?"
            )
        return output_dict
        

    def loss(self, input_dict, output_dict):
        preds = output_dict['preds'].squeeze()
        # print('(GNN_FC.loss) preds.shape:', preds.shape)
        targets = input_dict['target']
        
        if 'reg' in self.task:
            batch_loss = F.mse_loss(preds, targets, reduction='sum')
        elif ('class' in self.task):
            if ('bin' in self.task):
                # BCEWithLogitsLoss is more stable than using 'sigmoid' 
                # activation + BCE, see:
                # https://pytorch.org/docs/stable/generated/
                # torch.nn.BCEWithLogitsLoss.html
                batch_loss = F.binary_cross_entropy_with_logits(
                    preds, 
                    targets, 
                    reduction='sum'
                )
        else:
            print(f'No loss function implemented for task=\'{self.task}\'!')
            batch_loss = None
            
        loss_dict = {
            'loss': batch_loss,
            'size': targets.shape[0]
        }
        return loss_dict

