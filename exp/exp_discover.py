"""
Time Series Causal Discovery Experiment Module
Main training/testing loop with attention-based causal discovery.
"""

import os
import time
import warnings
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import CC_discover
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    test_params_flop,
    extract_channel_mask_from_model,
)
from utils.metrics import metric
from utils.explain_agc import compute_agc_grad_effect, agc_consistency_loss

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    """
    Main experiment class for CC_discover causal discovery model.

    Handles training with optional AGC (Attention-Gradient Consistency) loss,
    multi-layer cross-attention accumulation during testing, and
    input-gradient sensitivity map computation.
    """

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        """Build and return the CC_discover model (wrapped in DataParallel if needed)."""
        model = CC_discover.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """Return (dataset, dataloader) for the given split flag."""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """Create and return the Adam optimizer."""
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """Return the MSE loss criterion."""
        return nn.MSELoss()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def vali(self, vali_data, vali_loader, criterion):
        """
        Run one validation epoch.

        Args:
            vali_data:   Validation dataset (unused directly, kept for API consistency).
            vali_loader: Validation DataLoader.
            criterion:   Loss function.

        Returns:
            float: Average validation loss over the loader.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, setting):
        """
        Train the model, saving the best checkpoint based on validation loss.

        Supports optional AGC (Attention-Gradient Consistency) auxiliary loss,
        controlled by args.lambda_agc > 0.

        Args:
            setting (str): Experiment identifier string used for checkpoint naming.

        Returns:
            nn.Module: The model loaded with the best checkpoint weights.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data,  vali_loader  = self._get_data(flag='val')
        test_data,  test_loader  = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps    = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim    = self._select_optimizer()
        criterion      = self._select_criterion()

        print(f"Training steps per epoch: {train_steps}")

        # Learning-rate scheduler (OneCycleLR for TST strategy)
        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate,
            )
        else:
            scheduler = None

        # Cache references outside the training loop to avoid repeated attribute lookups
        core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        last_layer = core_model.model.backbone.decoder.layers[-1]
        use_agc    = (self.args.lambda_agc > 0)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                iter_count += 1
                model_optim.zero_grad()

                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Enable cross-attention caching when AGC loss is active
                last_layer.cross_attn.cache_expl = use_agc

                # Forward pass
                outputs      = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                outputs      = outputs[:, -self.args.pred_len:, :]
                batch_y_cut  = batch_y[:, -self.args.pred_len:, :]
                loss_pred    = criterion(outputs, batch_y_cut)
                reg_loss     = core_model.regularization()
                loss         = loss_pred + reg_loss

                # AGC auxiliary loss: align cross-attention with input-gradient effects
                if use_agc:
                    attn_cross = getattr(last_layer.cross_attn, 'last_attn', None)
                    v_cross    = getattr(last_layer.cross_attn, 'last_v',    None)
                    if attn_cross is not None and v_cross is not None:
                        loss_agc = 0.0
                        for ch in range(self.args.d_in):
                            y_scalar    = outputs[:, :, ch].mean(1)
                            grad_effect = compute_agc_grad_effect(
                                v_cross, y_scalar, create_graph=True
                            )
                            loss_agc += agc_consistency_loss(
                                attn_cross, grad_effect,
                                ch_idx=ch,
                                pred_len=self.args.pred_len,
                                seq_len=self.args.seq_len,
                                C=self.args.d_in,
                            )
                        loss = loss + self.args.lambda_agc * (loss_agc / self.args.d_in)

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                # Disable cross-attention caching after backward to save memory
                last_layer.cross_attn.cache_expl = False

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")

            train_loss = np.average(train_loss)
            vali_loss  = self.vali(vali_data, vali_loader, criterion)
            test_loss  = self.vali(test_data,  test_loader,  criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f}  Vali Loss: {vali_loss:.7f}  Test Loss: {test_loss:.7f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        # Load best checkpoint saved by EarlyStopping
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if not os.path.exists(best_model_path) or os.path.getsize(best_model_path) == 0:
            print(f"[WARNING] Checkpoint not found or empty: {best_model_path}")
            print("[WARNING] Using current model weights instead.")
        else:
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device, weights_only=True)
            )
        return self.model

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test(self, setting, test=0, folder_path=None, seed=None, itr=None):
        """
        Run inference on the test set, collecting predictions, attention maps,
        and input-gradient sensitivity maps.

        Optimization note: sensitivity maps reuse a single forward pass'
        computation graph for all C target channels (C backward passes),
        reducing forward-pass count from C+1 to 2 per batch.

        Args:
            setting     (str):  Experiment identifier.
            test        (int):  If 1, load checkpoint from disk before testing.
            folder_path (str):  Directory to save test outputs. Auto-generated if None.
            seed        (int):  Random seed (used for logging only).
            itr         (int):  Iteration index (used for logging only).

        Returns:
            dict with keys:
                'mse'       – mean squared error on the test set
                'preds'     – predicted values  (N, pred_len, C)
                'trues'     – ground-truth values (N, pred_len, C)
                'inputx'    – encoder inputs (N, seq_len, C)
                'attn_sums' – accumulated cross-attention tensors per layer
                'n_samples' – total number of test samples
                'input_grad'– seed-averaged sensitivity map (C, C, T) or None
        """

        def _ensure_BTC(x, d_in):
            """Ensure tensor is (B, T, C); permute if it arrives as (B, C, T)."""
            if x is None or x.dim() != 3:
                return x
            if x.shape[-1] == d_in:
                return x
            if x.shape[1] == d_in:
                return x.permute(0, 2, 1).contiguous()
            return x

        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('Loading model from checkpoint...')
            self.model.load_state_dict(
                torch.load(
                    os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                    map_location=self.device,
                )
            )

        preds, trues, inputx = [], [], []
        all_attn_sums, n_total_samples = None, 0
        all_sensitivity_maps = []

        if folder_path is None:
            date_str    = datetime.date.today().strftime("%Y%m%d")
            folder_path = f'./test_results/{date_str}/{setting}/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x = _ensure_BTC(batch_x, self.args.d_in)
                batch_y = _ensure_BTC(batch_y, self.args.d_in)

                outputs, attn_list = self.model(
                    batch_x, batch_x_mark, batch_y, batch_y_mark, return_attn=True
                )
                outputs     = outputs[:, -self.args.pred_len:, :]
                batch_y_cut = batch_y[:, -self.args.pred_len:, :]

                pred = outputs.detach().cpu().numpy()
                true = batch_y_cut.detach().cpu().numpy()

                # Accumulate cross-attention across batches (weighted by batch size)
                first = attn_list[0]
                if isinstance(first, (tuple, list)):
                    B = first[1].shape[0]
                    if all_attn_sums is None:
                        all_attn_sums = [pair[1].sum(dim=0) for pair in attn_list]
                    else:
                        for l, pair in enumerate(attn_list):
                            all_attn_sums[l] += pair[1].sum(dim=0)
                else:
                    B = first.shape[0]
                    if all_attn_sums is None:
                        all_attn_sums = [attn.sum(dim=0) for attn in attn_list]
                    else:
                        for l, attn in enumerate(attn_list):
                            all_attn_sums[l] += attn.sum(dim=0)
                n_total_samples += B

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                # Compute input-gradient sensitivity maps.
                # Strategy: build the computation graph once (1 forward pass),
                # then run C backward passes sharing the same graph.
                # This reduces the forward-pass count from C+1 to 2 per batch.
                batch_x_grad = batch_x.detach().requires_grad_(True)
                with torch.enable_grad():
                    self.model.zero_grad()
                    outputs2 = self.model(batch_x_grad, batch_x_mark, batch_y, batch_y_mark)
                    outputs2 = outputs2[:, -self.args.pred_len:, :]

                    C = self.args.d_in
                    sample_grads = []
                    for tgt_idx in range(C):
                        # retain_graph=True for all but the last channel
                        retain = (tgt_idx < C - 1)
                        grads = torch.autograd.grad(
                            outputs2[:, :, tgt_idx].sum(),
                            batch_x_grad,
                            retain_graph=retain,
                            create_graph=False,
                        )[0]
                        # Sum over batch; divide by n_total_samples at the end
                        avg_grad = grads.sum(dim=0).detach().cpu().numpy()
                        sample_grads.append(avg_grad)

                    # sensitivity_map: (C_tgt, C_src, T)
                    sensitivity_map = np.stack(sample_grads, axis=0).transpose(0, 2, 1)
                    all_sensitivity_maps.append(sensitivity_map)

        # Average sensitivity map across all batches, weighted by sample count
        avg_sensitivity_map = None
        if all_sensitivity_maps:
            avg_sensitivity_map = (
                np.sum(np.asarray(all_sensitivity_maps), axis=0) / (n_total_samples + 1e-8)
            )

        preds  = np.concatenate(preds,  axis=0)
        trues  = np.concatenate(trues,  axis=0)
        inputx = np.concatenate(inputx, axis=0)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        if self.args.inverse:
            preds = test_data.inverse_transform(
                preds.reshape(-1, self.args.d_in)
            ).reshape(-1, self.args.pred_len, self.args.d_in)
            trues = test_data.inverse_transform(
                trues.reshape(-1, self.args.d_in)
            ).reshape(-1, self.args.pred_len, self.args.d_in)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print(f'MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}')

        return {
            'mse':        mse,
            'preds':      preds,
            'trues':      trues,
            'inputx':     inputx,
            'attn_sums':  all_attn_sums,
            'n_samples':  n_total_samples,
            'input_grad': avg_sensitivity_map,
        }
