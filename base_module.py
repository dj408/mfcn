"""
This file contains useful classes
and functions for extending the
torch.nn.Module class for model-building: 

(1) Class definition for 'BaseModule',
an extension of torch.nn.Module with
built-in loss and metrics methods for
regressor or binary classifier models.

(2) Function 'test_nn', which computes
basic metrics for regression and binary
classification models built from 
BaseModule.
"""

import nn_utilities as nnu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    Dataset, 
    DataLoader
)
from torchmetrics.regression import (
    MeanSquaredError,
    R2Score
)
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinarySpecificity,
    BinaryAUROC,
    BinaryConfusionMatrix
)
from torchmetrics import MetricCollection
from torch_geometric.data import Data
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Callable,
    Any
)




class BaseModule(nn.Module):
    """
    Subclass of torch.nn.Module, designed to work
    flexibly with the 'train_model' function (in
    train_fn.py). Has built-in loss functions and 
    metrics calculation methods specific to regression 
    and binary classification models (if not overridden).
    
    __init__ args:
        task: string key description of the model task, e.g.,
            'regression' or 'binary classification'.
        loss_fn: (optional) functional loss to use; if
            'None', will attempt to assign a default loss
            function based on the 'task' argument in 
            __init__().
        loss_fn_kwargs: for a torch.nn.functional loss.
        target_name: string key for the prediction target.
        key_prefix: string prefix for each metric column
             name in the training records. Should end in '_'.
        device: manual device onto which to move the model
            weights.
    """
    def __init__(
        self,
        task: str,
        loss_fn: Optional[Callable] = None,
        loss_fn_kwargs: Dict[str, Any] = {},
        target_name: str = None,
        key_prefix: str = '',
        device = None
    ):
        super(BaseModule, self).__init__()
        self.task = task
        self.device = device
        self.target_name = target_name
        self.key_prefix = key_prefix

        self._set_up_metrics()
        self.set_device()
        
        if loss_fn is None:
            if 'reg' in self.task.lower():
                self.loss_fn = F.mse_loss
            elif 'class' in self.task.lower():
                if 'bin' in self.task.lower():
                    # F.binary_cross_entropy_with_logits removes need
                    # for sigmoid activation after last layer, but targets
                    # need to be floats between 0 and 1
                    # https://pytorch.org/docs/stable/generated/
                    # torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
                    self.loss_fn = F.binary_cross_entropy_with_logits
            else:
                raise NotImplementedError(
                    f"No loss function implemented for task = '{self.task}'!"
                )
        else:
            self.loss_fn = loss_fn

        self.loss_fn_kwargs = {'reduction': 'mean'} \
            if loss_fn_kwargs is None else loss_fn_kwargs
        self.loss_keys = None # set in first call to update_metrics()

        self.epoch_loss_dict = {
            'train': {'size': 0.}, 
            'valid': {'size': 0.}
        }

        
    def set_device(self):
        # optional: manually enforce device
        if self.device is not None:
            # self.device = torch.device(device)
            self.to(self.device)

    
    def get_device(self):
        # find device that the model weights are on
        return next(self.parameters()).device


    def _loss(self, input_dict, output_dict):
        """
        This fn wraps loss_fn so it takes dicts storing 
        preds, targets as inputs, and outputs a loss_dict.
        Separated out in case this module is just one head
        of a multi-head model, and its loss is just one
        term of a composite loss function.
        """
        preds = output_dict['preds']
        # print('preds.shape =', preds.shape)
        targets = input_dict['target']
        # print('targets.shape =', targets.shape)
        
        # 'targets' may itself be a dict holding
        # multiple targets
        if (self.target_name is not None) \
        and (isinstance(targets, dict)):
            targets = targets[self.target_name]
        # print('targets.shape =', targets.shape)
        # print('targets =', targets)
        
        loss = self.loss_fn(
            input=preds.squeeze(),
            target=targets.squeeze(),
            **self.loss_fn_kwargs 
        ) 
        loss_dict = {
            'loss': loss,
            'size': targets.shape[0]
        }
        return loss_dict

    
    def loss(self, input_dict, output_dict):
        """
        Simply grabs preds and targets from dictionary
        containers and calls 'self._fc_loss', unless 
        overridden by subclass.
        """
        loss_dict = self._loss(input_dict, output_dict)
        return loss_dict

    
    def _set_up_metrics(self):
        """
        Convenience method to set output layer activation and 
        metrics based on model task type.
        """
        if 'reg' in self.task.lower():
            # self.fc_out_layer_activ_fn = None
            self.mse = MeanSquaredError()
            self.R2_score = R2Score()
            
        elif 'class' in self.task.lower():
            if 'bin' in self.task.lower():
                self.accuracy = BinaryAccuracy()
                self.specificity = BinarySpecificity()
                self.f1 = BinaryF1Score()
                self.f1_neg = BinaryF1Score()
                self.auroc = BinaryAUROC()
                self.class_1_pred_ct = 0
                # self.running_class_1_probs = []
            else:
                raise NotImplementedError(
                    "Multiclass classification not yet implemented"
                    " in BaseModule!"
                )

    
    def update_metrics(
        self, 
        phase,
        loss_dict,
        input_dict = None, 
        output_dict = None
    ):
        
        # on first call only: initialize loss counters
        if self.loss_keys is None:
            self.loss_keys = [
                k for k, v in loss_dict.items() \
                if 'loss' in k.lower()
            ]
            for k in self.epoch_loss_dict.keys():
                for loss_key in self.loss_keys:
                    self.epoch_loss_dict[k][loss_key] = 0.0
                    
        # for both train and valid phases, update losses and sizes
        for loss_key in self.loss_keys:
            self.epoch_loss_dict[phase][loss_key] += loss_dict[loss_key] 
        self.epoch_loss_dict[phase]['size'] += loss_dict['size'] 

        # validation metrics
        if phase == 'valid':
            preds = output_dict['preds']
            # print('preds.shape:', preds.shape)
            target = input_dict['target']
            
            # 'target' may itself be a dict containing
            # multiple targets
            if (self.target_name is not None) \
            and (isinstance(target, dict)):
                target = target[self.target_name].squeeze()
                # print('target:', target)
            
            if 'reg' in self.task.lower():
                self.R2_score.update(preds.squeeze(), target)
                self.mse.update(preds.squeeze(), target)
                
            elif 'class' in self.task.lower():
                if 'bin' in self.task.lower():
                    device = self.get_device()
                    # accuracy and f1
                    # when using BCE with logits, need to convert
                    # logit preds to 0 or 1 -> 
                    # logit = log(p/(1-p)) -> logit>0 -> p>0.5 -> predicted '1'
                    class_preds = torch.tensor(
                        [(logit > 0.0) for logit in preds],
                        dtype=torch.long,
                        device=device
                    )
                    class_targets = torch.tensor(
                        [int(t) for t in target],
                        dtype=torch.long,
                        device=device
                    )
                    self.accuracy.update(class_preds, class_targets)
                    self.f1.update(class_preds, class_targets)
                    self.f1_neg.update(
                        torch.logical_not(class_preds).to(torch.long), 
                        torch.logical_not(class_targets).to(torch.long)
                    )
                    self.specificity.update(class_preds, class_targets)
                    # auroc detects logits if preds are outside of [0, 1]
                    self.auroc.update(preds, class_targets)
                    self.class_1_pred_ct += torch.sum(preds > 0.).item()

    
    def calc_metrics(
        self,
        epoch,
        input_dict = None, 
        output_dict = None, 
        loss_dict = None
    ):

        metrics_dict = {'epoch': epoch}
        
        # include train and validation mean losses in metrics_dict
        for phase in ('train', 'valid'):
            for loss_key in self.loss_keys:
                avg_loss = self.epoch_loss_dict[phase][loss_key] \
                    / self.epoch_loss_dict[phase]['size']
                
                metrics_dict = metrics_dict \
                    | {(loss_key + '_' + phase): avg_loss.item()}

                # reset epoch loss to 0.
                self.epoch_loss_dict[phase][loss_key] = 0.
                
            self.epoch_loss_dict[phase]['size'] = 0.

        if 'reg' in self.task.lower():
            # MSE is redundant if using MSE loss in training...
            mse_score = self.mse.compute().detach().cpu().numpy().item()
            metrics_dict = metrics_dict \
                | {(self.key_prefix + 'mse_valid'): mse_score}
            self.mse.reset()

            # R^2
            R2_score = self.R2_score.compute().detach().cpu().numpy().item()
            metrics_dict = metrics_dict \
                | {(self.key_prefix + 'R2_valid'): R2_score}
            self.R2_score.reset()
            
        elif 'class' in self.task.lower():
            if 'bin' in self.task.lower():
                accuracy_score = self.accuracy.compute().detach().cpu().numpy().item()
                f1_score = self.f1.compute().detach().cpu().numpy().item()
                f1_neg_score = self.f1_neg.compute().detach().cpu().numpy().item()
                specificity_score = self.specificity.compute().detach().cpu().numpy().item()
                auroc_score = self.auroc.compute().detach().cpu().numpy().item()
                metrics_dict = metrics_dict \
                    | {(self.key_prefix + 'accuracy_valid'): accuracy_score} \
                    | {(self.key_prefix + 'f1_valid'): f1_score} \
                    | {(self.key_prefix + 'f1_neg_valid'): f1_neg_score} \
                    | {(self.key_prefix + 'specificity_valid'): specificity_score} \
                    | {(self.key_prefix + 'auroc_valid'): auroc_score} \
                    | {(self.key_prefix + 'class_1_pred_ct_valid'): self.class_1_pred_ct} 
                self.accuracy.reset()
                self.f1.reset()
                self.f1_neg.reset()
                self.specificity.reset()
                self.auroc.reset()
                self.class_1_pred_ct = 0
        
        return metrics_dict



def test_nn(
    trained_model: nn.Module,
    data_container: Dict[str, DataLoader] | Data,
    task: str,
    device: str = 'cpu',
    target_name: str = 'target',
    set_key: str = 'test',
    using_pytorch_geo: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Computes standard regression or binary
    classification metrics for the 'test' set 
    in a data_container, given a trained 
    BaseModule (or extension).

    Args:
        trained_model: trained BaseModule model.
        task: string key coding model task, e.g.,
            'regression' or 'binary classification.'
        device: device key on which the tensors live, 
            e.g. 'cpu' or 'cuda.'
        target_name: string key for the model target.
        data_container: dictionary of Data-
            Loaders, with a keyed 'test' set, or
            a pytorch geometric Data object (e.g.
            of one graph, with train/valid/test masks).
        set_key: which set ('train'/'valid'/'test') to 
            compute metrics for (default: 'test').
        using_pytorch_geo: whether the DataLoaders
            hold PyTorch Geometric datasets (i.e.
            where data are loaded as sparse block-
            diagonal matrices, requiring batch indices).
    Returns:
        2-tuple: (1) dictionary of metric scores,
        and (2) dictionary of other task-specific
        metric objects (e.g. a confusion matrix
        calculator object for classification).
    """
    print(f'Calculating metrics for {set_key} set:')

    # record raw model preds and targets, for option
    # to calculate other metrics no calculated here
    auxiliary_metrics_dict = {
        'preds': [],
        'targets': []
    }
    
    # set up metrics collections based on task
    if 'reg' in task.lower():
        metric_collection = MetricCollection({
            'mse': MeanSquaredError(),
            'R2': R2Score()
        }).to(device)
        
    elif 'class' in task.lower():
        if 'bin' in task.lower():
            metric_collection = MetricCollection({
                'acc': BinaryAccuracy(),
                'f1': BinaryF1Score(),
                'specificity': BinarySpecificity(),
                'auroc': BinaryAUROC()
            }).to(device)

            # auxiliary metrics (not part of the MetricCollection)
            f1_neg_calculator = BinaryF1Score().to(device)
            confusion_mat_calculator = BinaryConfusionMatrix().to(device)
            # auroc_calculator = BinaryAUROC().to(device)
            auxiliary_metrics_dict |= {
                'f1_neg': f1_neg_calculator,
                'confusion_matrix': confusion_mat_calculator,
                # 'auroc': auroc_calculator,
            }
        
    def update(preds, targets):
        """
        Inner function to update metrics (in containers);
        called more than once if calculating in batches.
        """
        auxiliary_metrics_dict['preds'].append(preds)
        auxiliary_metrics_dict['targets'].append(targets)
        
        # update any other auxiliary metrics (processing preds/targets as needed)
        if 'reg' in task.lower():
            pass
            
        elif 'class' in task.lower(): 
            if 'bin' in task.lower():
                targets = targets.long()
                # preds_logits = preds.clone().detach().requires_grad_(False)
                # convert logits to probabilities using sigmoid function
                # preds_probs = 1. / (1. + torch.exp(-preds_logits))
                # convert logits to 0 or 1 at a threshold of p = 0.5
                # logit = log(p / (1-p)) -> logit > 0 -> p > 0.5 -> predicted '1'
                preds_binary = (preds > 0.).long()

                # update aux metrics calculators
                f1_neg_calculator.update(
                    torch.logical_not(preds_binary).to(torch.long),
                    torch.logical_not(targets).to(torch.long)
                )
                confusion_mat_calculator.update(preds_binary, targets)
                # BinaryAUROC can take preds as probaibilities or logits
                # https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html
                # auroc_calculator.update(preds, targets)

        # update metrics collection (with unprocessed preds/targets)
        # print('preds.device:', preds.device)
        # print('targets.device:', targets.device)
        metric_collection.update(preds, targets)

    
    # get model predictions on test set
    trained_model.eval()
    with torch.set_grad_enabled(False):
        if type(data_container) is dict:
            for batch in data_container[set_key]:
                if using_pytorch_geo:
                    data = batch.to(device)
                    batch_output_dict = trained_model(data)
                    targets = batch.y
                else:
                    batch_output_dict = trained_model(batch)
                    targets = batch['target'][target_name]
                preds = batch_output_dict['preds'].squeeze()
                update(preds, targets)

        # data_container is not a dict -> we assume we have a 
        # pytorch geo Data object holding one graph, with set masks
        elif using_pytorch_geo:
            data = data_container.to(device)
            output_dict = trained_model(data)
            targets = data.y
            preds = output_dict['preds'].squeeze()
            if ('val' in set_key):
                mask = data.val_mask
            elif ('test' in set_key):
                mask = data.test_mask
            elif ('train' in set_key):
                mask = data.train_mask
            test1, test2 = preds[mask].shape, targets[mask].shape
            update(preds[mask], targets[mask])
            
        # after collecting all [batches], compute final test metrics
        # detached and sent to cpu downstream, e.g. in 'cv.run_cv'
        metric_scores_dict = metric_collection.compute()
        # metric_collection.reset()
        if auxiliary_metrics_dict is not None:
            for k, v in auxiliary_metrics_dict.items():
                if k in ('preds', 'targets'):
                    metric_scores_dict[k] = torch.concatenate(v)
                else:
                    metric_scores_dict[k] = v.compute()
        print('\tDone calculating metrics.')
        
        return metric_scores_dict  #, auxiliary_metrics_dict


