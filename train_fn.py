"""
Function to train/fine-tune a PyTorch model, 
using an Accelerator wrapper for CPU-GPU(s)
device support.

Ref:
https://huggingface.co/docs/accelerate/en/package_reference/accelerator
"""

import utilities as u
import nn_utilities as nnu
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any
)


def train_model(
    args,
    model: nn.Module,
    data_container: Dict[str, DataLoader] | Data,
    optimizer_class: type = optim.AdamW,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    grad_track_param_names: Optional[Tuple[str]] = None,
    snapshot_path: Optional[str] = None,
    save_states: bool = False,
    save_final_model_state: bool = False,
    return_best: bool = True,
    use_acc_print: bool = False,
    using_pytorch_geo: bool = False,
    verbosity: int = 0
) -> Tuple[
        Optional[nn.Module], # final model
        Optional[List[Dict[str, Any]]], # train history records
        Optional[nnu.EpochCounter] # EpochCounter object
]:
    """
    A general training function for base_module.BaseModule models, and
    potentially other pytorch models.

    Features:
    - Pass an optional `snapshot_path` to load an accelerate model snapshot and 
      resume training; else leave `None`.
    - Optimizer is initialized after first forward pass to handle dynamically created parameters.
    
    Notes:
    - The loss function must be an attrib. of the model, to get placed on the 
    device properly with Accelerate.
    - It's possible to keep training to the max number of epochs if
    validation loss kept improving by an arbitrarily small
    amount, but return a final model from a much earlier epoch, if 
    'args.MAIN_METRIC_REL_IMPROV_THRESH' is 1.0 or not None. That is,
    the model returned will be that with the weights where the valid
    loss last improved by this thresholded ratio.
    
    Args:
        args: ArgsTemplate object containing experiment arguments.
        model: the torch.nn.Module model object to train. Must have 
            'forward', 'loss', and 'update_metrics' methods that
            take dictionaries, as done in base_module.BaseModule.
        data_container: dictionary of Dataloaders by set, or pytorch_geometric 
            Data object containing the training data and set masks.
        optimizer_class: class of optimizer to use (default: AdamW)
        optimizer_kwargs: dictionary of keyword arguments for optimizer initialization
        grad_track_param_names: tuple of strings of parameter names
            for which their gradients and weights will be tracked.
            See 'nnu.log_parameter_grads_weights' for details.
        snapshot_path: directory path string from which to load a
            previously trained model, e.g. for resuming training.
        save_states: if True, save (overwrite) the best model obtained
            (by a new best main metric validation set score) as 'best' 
            in the 'snapshot_path' directory, each time a new best is 
            obtained ('checkpointing'). If False, the best model can still 
            be returned (not saved to disk) if 'return_best' is True.
        save_final_model_state: if True, the last (and likely not best)
            state of the model with weights from the last epoch reached
            will be saved (as 'final', not 'best').
        return_best: whether to return the trained model object with
            its best weights, plus the training records and epoch counter 
            objects.
        use_acc_print: if True, use Accelerator's print method, instead
            of base python's 'print'.
        using_pytorch_geo: boolean indicating whether a pytorch_geometric
            data container object is being used, instead of torch's
            base DataLoaders.
        verbosity: controls volume of print output
            as the function runs. >1 prints epoch-
            by-epoch loss and time summaries.
    Returns:
        3-tuple of (model, records, epoch_ctr) if 'return_best' is True, 
        or else (None, None, None) if error (e.g. no model weights were 
        created) or 'return_best' is False. 
    """

    """
    INNER FUNCTIONS/CLASSES
    """
    def _save_snapshot(name):
        if (args.MODEL_SAVE_DIR is not None) and (args.MODEL_SAVE_DIR != ''):
            snapshot_path = f'{args.MODEL_SAVE_DIR}/{name}'
            acc.save_state(snapshot_path)

    def _log_output(out):
        if use_acc_print:
            acc.print(out)
        else:
            with open(args.PRINT_DIR, 'a') as f:
                f.write(out + '\n')

    def _initialize_optimizer(model):
        nonlocal optimizer_kwargs
        if optimizer_kwargs is None:
            optimizer_kwargs = {
                'lr': args.LEARN_RATE,
                'betas': args.ADAM_BETAS,
                'weight_decay': args.ADAM_WEIGHT_DECAY
            }
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        optimizer = acc.prepare(optimizer)
        return optimizer

    """
    INITIALIZE DIRS, WEIGHTS, METRICS
    """
    if verbosity > 0:
        print('save_states:', save_states)
    # best_model_wts = copy.deepcopy(model.state_dict())
    if (args.MODEL_SAVE_DIR is not None) and (args.MODEL_SAVE_DIR != ""):
        os.makedirs(args.MODEL_SAVE_DIR, exist_ok=True)
        
    # store metrics by epoch in list of dicts 
    # (i.e. 'records' -> easy to convert to pd.DataFrame)
    records = []
    
    # initialize EpochCounter
    epoch_ctr = nnu.EpochCounter(0, args.MAIN_METRIC)
    best_epoch = 1


    """
    ACCELERATOR WRAPPER
    """
    acc = Accelerator(
        device_placement=args.DEVICE, 
        cpu=args.ON_CPU
    )

    # wrap model with accelerator
    model = acc.prepare(model)
    if type(data_container) is dict:
        data_container['train'], data_container['valid'] = acc.prepare(
             data_container['train'], 
             data_container['valid']
         )
    else:
        data_container = acc.prepare(data_container)
    
    # custom objects must be 'registered for checkpointing'
    acc.register_for_checkpointing(epoch_ctr)
    acc_state = AcceleratorState(cpu=args.ON_CPU)
    num_devices, device_type, distr_type = (
        acc_state.num_processes, 
        acc_state.device, 
        acc_state.distributed_type
    )

    # log key args
    out = f'{args.MODEL_NAME_TIMESTAMP}'
    out += f'\n\nParameters:\n'
    out += f'MODEL_NAME = {args.MODEL_NAME}\n'
    out += f'TASK = {args.TASK}\n'
    out += f'TARGET_NAME = {args.TARGET_NAME}\n'
    out += f'N_EPOCHS = {args.N_EPOCHS}\n'
    out += f'BURNIN_N_EPOCHS = {args.BURNIN_N_EPOCHS}\n'
    out += f'LEARN_RATE = {args.LEARN_RATE}\n'
    out += f'STOP_RULE = {args.STOP_RULE}\n'
    out += f'PATIENCE = {args.NO_VALID_LOSS_IMPROVE_PATIENCE}\n'

    out += f'\nMLP head parameters (if applicable):\n'
    out += f'NN_HIDDEN_DIMS = {args.NN_HIDDEN_DIMS}\n'
    out += f'USE_BATCH_NORMALIZATION = {args.USE_BATCH_NORMALIZATION}\n'
    out += f'MLP_USE_DROPOUT = {args.MLP_USE_DROPOUT} (p = {args.MLP_DROPOUT_P})\n'
    out += f'BATCH_SIZES (train/valid/test) = {args.BATCH_SIZES}\n'

    if 'mfcn' in args.MODEL_NAME.lower():
        out += f'\nMFCN-wavelet parameters:\n'
        out += f'WAVELET_TYPE = {args.WAVELET_TYPE}\n'
        out += f'INCLUDE_LOWPASS_WAVELET = {args.INCLUDE_LOWPASS_WAVELET}\n'
        if args.WAVELET_TYPE.lower() == 'p':
            out += f'P_WAVELET_SCALES = {args.P_WAVELET_SCALES}\n'
            if 'handcraft' in args.P_WAVELET_SCALES.lower():
                out += f'HANDCRAFT_P_CMLTV_KLD_QUANTILES = {args.HANDCRAFT_P_CMLTV_KLD_QUANTILES}\n'
        elif 'spect' in args.WAVELET_TYPE.lower():
            out += f'SPECTRAL_C = {args.SPECTRAL_C}\n'
            out += f'MFCN_MAX_KAPPA = {args.MFCN_MAX_KAPPA}\n'
        out += f'MFCN_WITHIN_FILTER_CHAN_OUT = {args.MFCN_WITHIN_FILTER_CHAN_OUT}\n'
        out += f'MFCN_CROSS_FILTER_COMBOS_OUT = {args.MFCN_CROSS_FILTER_COMBOS_OUT}\n'
        out += f'MFCN_FINAL_CHANNEL_POOLING = {args.MFCN_FINAL_CHANNEL_POOLING}\n'
        out += f'MFCN_FINAL_NODE_POOLING = {args.MFCN_FINAL_NODE_POOLING}\n'
    
    elif 'mcn' in args.MODEL_NAME.lower():
        out += f'\nMFCN-lowpass parameters:\n'
        out += f'NON_WAVELET_FILTER_TYPE = {args.NON_WAVELET_FILTER_TYPE}\n'
        out += f'MCN_WITHIN_FILTER_CHAN_OUT = {args.MCN_WITHIN_FILTER_CHAN_OUT}\n'
        if 'spect' in args.NON_WAVELET_FILTER_TYPE.lower():
            out += f'SPECTRAL_C = {args.SPECTRAL_C}\n'
            out += f'MFCN_MAX_KAPPA = {args.MFCN_MAX_KAPPA}\n'
            
    _log_output(out)

    # log training hardware info
    # print(f'AcceleratorState device: {acc_state.device}')
    distributed_str = distr_type.split('.')[0]
    out = f'Training on {num_devices} x {device_type}' \
        + f' device (distributed: {distributed_str})'
    if verbosity > 0:
        print(out)
    _log_output(out)

    
    """
    OPTIONAL: LOAD MODEL SNAPSHOT
    - to resume training from saved model state
    """
    if snapshot_path is not None:
        acc.load_state(snapshot_path)
        out = f'...resuming training from snapshot at epoch {epoch_ctr.n}'
        _log_output(out)

    
    """
    TRAINING LOOP
    """
    time_0 = time.time()
    ul_str = '-' * 12
    num_epochs_no_vl_improv = 0
    last_epoch_flag = False

    # save initial model weights
    if grad_track_param_names is not None:
        nnu.log_parameter_grads_weights(
            args=args,
            model=model,
            grad_track_param_names=grad_track_param_names,
            epoch_i=-1, 
            batch_i=-1,
            save_grads=False
        )

    # classification task: print class 1 preds proportion for
    # each epoch and phase; here, init empty counters container
    if (verbosity > 0) and ('class' in args.TASK):
        class1_preds_ctr = nnu.Class1PredsCounter()

    # loop through epochs
    for epoch in range(epoch_ctr.n + 1, epoch_ctr.n + args.N_EPOCHS + 1):
        time_epoch_0 = time.time()
        epoch_ctr += 1
        out = f'\nEpoch {epoch}/{args.N_EPOCHS}\n{ul_str}'
        _log_output(out)
        if verbosity > 1:
            print(out)

        # each epoch has a training and validation phase/phase
        for phase in ('train', 'valid'):
            training = (phase == 'train')
            model.train() if training else model.eval()

            with torch.set_grad_enabled(training):

                if using_pytorch_geo:
                    # if 'data_container' is a dictionary (with 'train' and 'valid'
                    # sets), loop through batches
                    if isinstance(data_container, dict):
                        for batch_i, batch in enumerate(data_container[phase]):

                            # PATCH: batches of size 1 cause errors in mfcn's forward()
                            # but using 'drop_last=True' in DataLoader also errors
                            if batch.num_graphs > 1:
                                output_dict = model(batch)
                                # Initialize optimizer after first forward pass in epoch 1
                                if epoch == 1 and batch_i == 0:
                                    optimizer = _initialize_optimizer(model)
                                # print("output_dict['preds'].shape", output_dict['preds'].shape)
                                input_dict = {'target': batch.y}
                                loss_dict = model.loss(input_dict, output_dict)
    
                                # classification task: update class 1 preds counts, each batch
                                if (verbosity > 0) and ('class' in args.TASK):
                                    class1_preds_ctr.update(output_dict, phase)

                    # elif 'data_container' is a torch_geometric.data.Data object,
                    # there are no batches; during loss calc, use 'train' and 'val' mask 
                    # attributes;
                    # note that in a node-level task, we assume we have full knowledge 
                    # of the ($k$-NN) graph structure and signals, and only withhold 
                    # valid/test-set node targets at loss calculation and evaluation time
                    elif isinstance(data_container, Data):
                        batch_i = 0
                        output_dict = model(data_container)
                        if epoch == 1 and batch_i == 0:
                            optimizer = _initialize_optimizer(model)
                        preds = output_dict['preds']
                        target = data_container.y[data_container.train_mask] \
                            if training else data_container.y[data_container.val_mask]
                        input_dict = {'target': target}
                        output_dict['preds'] = preds[data_container.train_mask] \
                            if training else preds[data_container.val_mask]
                        loss_dict = model.loss(input_dict, output_dict)

                        # classification task: update class 1 preds counts, once
                        if (verbosity > 0) and ('class' in args.TASK):
                            class1_preds_ctr.update(output_dict, phase)

                    # for both dicts of DataLoaders and single Data objects with masks:
                    # train phase only: backward pass and optimizer step
                    if training:
                        acc.backward(loss_dict['loss'])
                        if grad_track_param_names is not None:
                            nnu.log_parameter_grads_weights(
                                args,
                                model,
                                grad_track_param_names,
                                epoch - 1, 
                                batch_i
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    # update batch loss (test and valid) and metrics (valid only)
                    model.update_metrics(
                        phase, 
                        loss_dict, 
                        input_dict, 
                        output_dict
                    )

                # not using a pytorch geometric DataLoaders or Data object
                else: 
                    for batch_i, input_dict in enumerate(data_container[phase]):
                        output_dict = model(input_dict) # calls model.forward
                        if epoch == 1 and batch_i == 0:
                            optimizer = _initialize_optimizer(model)
                        loss_dict = model.loss(input_dict, output_dict)

                        # train phase only: backward pass and optimizer step
                        if training:
                            acc.backward(loss_dict['loss'])
                            if grad_track_param_names is not None:
                                nnu.log_parameter_grads_weights(
                                    args,
                                    model, 
                                    grad_track_param_names,
                                    epoch - 1, 
                                    batch_i
                                )
                            optimizer.step()
                            optimizer.zero_grad()
                            
                        # update batch loss (test and valid) and metrics (valid only)
                        model.update_metrics(
                            phase, 
                            loss_dict, 
                            input_dict, 
                            output_dict
                        )

                        # classification task: update class 1 preds counts, each batch
                        if (verbosity > 0) and ('class' in args.TASK):
                            class1_preds_ctr.update(output_dict, phase)

        
        # after both train and valid sets are complete
        # calc epoch losses/metrics
        epoch_hist_d = model.calc_metrics(epoch, input_dict)

        # classification task: print class 1 preds counts
        if (verbosity > 0) and ('class' in args.TASK):
            class1_preds_ctr.print_preds_counts()
            class1_preds_ctr.reset()

        # log/print losses
        train_loss = epoch_hist_d['loss_train']
        valid_loss = epoch_hist_d['loss_valid']
        epoch_time_elapsed = time.time() - time_epoch_0
        epoch_min = int(epoch_time_elapsed // 60)
        epoch_sec = epoch_time_elapsed % 60
        out = f'losses:\n\ttrain: {train_loss:.6e}' \
            + f'\n\tvalid: {valid_loss:.6e}' \
            + f'\ntime elapsed: {epoch_min}m, {epoch_sec:.2f}s'
        _log_output(out)
        if verbosity > 1:
            print(out)

        # validation phases: early stopping and train history / best model saving steps
        if (phase == 'valid'):
            
            # save initial model weights and set initial validation loss in epoch 1
            if epoch_ctr.n == 1:
                best_model_wts = copy.deepcopy(model.state_dict())
                epoch_ctr.set_best('_valid_loss', epoch, valid_loss)
            # epochs > 1: check for new best validation loss; if not,
            # increment counter of epochs w/o improvement
            elif valid_loss < epoch_ctr.best['_valid_loss']['score']:
                epoch_ctr.set_best('_valid_loss', epoch, valid_loss)
                num_epochs_no_vl_improv = 0
            else:
                num_epochs_no_vl_improv += 1

            # if burn-in period passed AND 'patience' num epochs w/o valid loss 
            # improvement reached: set 'last_epoch_flag=True': this will break 
            # epochs' for-loop at end of the current epoch
            if (epoch > args.BURNIN_N_EPOCHS) \
            and (num_epochs_no_vl_improv >= args.NO_VALID_LOSS_IMPROVE_PATIENCE) \
            and (args.STOP_RULE is not None) \
            and ('no' in args.STOP_RULE) and ('improv' in args.STOP_RULE):
                out = f'Validation loss did not improve for' \
                      + f' {num_epochs_no_vl_improv} epochs: stopping.'
                print(out)
                _log_output(out)
                last_epoch_flag = True
                
            # check for new best key validation score (by a margin)
            new_best_score, score_thresh_reached = False, False
            epoch_score = epoch_hist_d[args.MAIN_METRIC]
            # phase initial score to beat, and validation loss, after first epoch
            if epoch_ctr.n == 1:
                epoch_ctr.set_best(args.MAIN_METRIC, epoch, epoch_score)
                
            best_main_metric = epoch_ctr.best[args.MAIN_METRIC]['score']
            score_thresh = best_main_metric
            if args.MAIN_METRIC_REL_IMPROV_THRESH is not None:
                 score_thresh *= args.MAIN_METRIC_REL_IMPROV_THRESH
            if args.MAIN_METRIC_IS_BETTER == 'lower':
                score_thresh_reached = (epoch_score < score_thresh)
            elif args.MAIN_METRIC_IS_BETTER == 'higher':
                score_thresh_reached = (epoch_score > score_thresh)
    
            # if new best validation score threshold reached, record it
            if score_thresh_reached:
                new_best_score = True
                best_main_metric = epoch_hist_d[args.MAIN_METRIC]
                epoch_ctr.set_best(args.MAIN_METRIC, epoch, best_main_metric)
                epoch_key = f"epoch_{epoch}"

            # append this epoch's losses, metrics, and time elapsed to records
            # include epoch training time and reigning epoch with best validation score
            epoch_hist_d['sec_elapsed'] = epoch_time_elapsed
            epoch_hist_d['best_epoch'] = epoch_ctr.best[args.MAIN_METRIC]['epoch']
            records.append(epoch_hist_d)
    
            # if new best epoch (by main metric validation set score):
            if new_best_score:
                # print msg
                score_str = f"{args.MAIN_METRIC}={epoch_hist_d[args.MAIN_METRIC]:.10f}"
                out = f"-> New best model! {score_str}"
                _log_output(out)
                if verbosity > 0:
                    print(out)
                    
                if save_states:
                    print(f'Saving model...')
                    # save (overwrite) 'best' model and training logs to reach it
                    _save_snapshot('best')
                    u.pickle_obj(args.TRAIN_LOGS_SAVE_DIR, records)
                    if verbosity > 0:
                        _log_output(f'Model saved.')
                        
                if return_best:
                    best_model_wts = copy.deepcopy(model.state_dict())

            # last_epoch_flag has been set, break out of epochs' for-loop and
            # jump to POST-TRAINING section
            if last_epoch_flag:
                break

    """
    POST-TRAINING
    """
    # get total time elapsed
    t_min, t_sec = u.get_time_min_sec(time.time(), time_0)
    out = f'{epoch_ctr.n} epochs complete in {t_min:.0f}min, {t_sec:.1f}sec.'
    _log_output(out)
    print(out)

    # log final best validation score and epoch
    if epoch_ctr.n > args.BURNIN_N_EPOCHS:
        best_epoch = epoch_ctr.best[args.MAIN_METRIC]['epoch']
        out = f'Best {args.MAIN_METRIC}: {best_main_metric:.4f}' \
            + f' at epoch {best_epoch}'
        _log_output(out)

    # save final training log
    u.pickle_obj(args.TRAIN_LOGS_FILENAME, records, overwrite=False)
    print(f'Final training log saved.')

    # optional: save final epoch's model state
    if save_final_model_state:
        _save_snapshot('final')
        print(f'Last model state saved.')
        

    # optional: load best model weights and return tuple with history log
    if return_best:
        if best_model_wts is not None:
            out = f'Returning model with best weights (from epoch {best_epoch}).\n'
            _log_output(out)
            print(out)
            model.load_state_dict(best_model_wts)
            return (model, records, epoch_ctr)
        else:
            out = f'No best model found; no weights were saved!\n'
            _log_output(out)
            print(out)
            return (None, None, None)
    else:
        return (None, None, None)

