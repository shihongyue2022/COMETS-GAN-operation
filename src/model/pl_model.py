import math
import os
import pickle
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch.optim import Optimizer

import wandb
from common.utils import corr
from dataset.pipeline import Pipeline
from model.modules.discriminator.cnn import CNNDiscriminator
from model.modules.generator.tcn import TCNGenerator
from utils.probe_logger import CovarianceProbeLogger, ProbeConfig

def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'bin').exists():
            return parent
    return current.parents[-1]



class MyLightningModule(LightningModule):

    def __init__(
        self, encoder_length: int, decoder_length: int,
        generator: DictConfig, discriminator: DictConfig, n_critic: int,
        stock_names, is_prices: bool, is_volumes: bool,
        pipeline_price: Pipeline, pipeline_volume: Pipeline,
        path_storage: str,
        metrics_probe: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.path_storage = path_storage

        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

        self.pipeline_price = pipeline_price
        self.pipeline_volume = pipeline_volume

        self.is_prices = is_prices
        self.is_volumes = is_volumes
        self.n_stocks = len(stock_names)
        n_features = self.n_stocks*2 if is_prices and is_volumes else self.n_stocks

        self.stock_names = stock_names
        self.feature_names = list()
        if self.is_prices:
            self.feature_names.extend([f'{s}_price' for s in stock_names])
        if self.is_volumes:
            self.feature_names.extend([f'{s}_volume' for s in stock_names])

        self.generator: TCNGenerator = instantiate(
            generator, n_features=n_features, n_stocks=self.n_stocks,
            is_prices=is_prices, is_volumes=is_volumes,
        )

        self.discriminator: CNNDiscriminator = instantiate(
            discriminator, n_features=n_features,
        )

        self.n_critic = n_critic

        self.mse = nn.MSELoss(reduction='none')

        self.generation_length = 3000

        # ---- probe metrics configuration ----
        self.probe_enabled = False
        self.probe_logger: Optional[CovarianceProbeLogger] = None
        self.probe_num_samples = 0
        self.probe_mc_mode = "flatten"
        self._probe_samples_batches: List[torch.Tensor] = []
        self._probe_targets_batches: List[torch.Tensor] = []

        if metrics_probe is not None and metrics_probe.get("enable", False) and self.is_prices:
            outdir_cfg = metrics_probe.get("outdir")
            eval_cfg = metrics_probe.get("eval_dir")
            predictor_name = metrics_probe.get("predictor_name", "COMETS-GAN")
            sampler_name = metrics_probe.get("sampler_name", "")
            mc_mode = metrics_probe.get("mc_mode", "flatten")
            probe_tag = metrics_probe.get("tag")
            self.probe_num_samples = int(metrics_probe.get("num_samples", 64))
            self.probe_mc_mode = mc_mode

            if outdir_cfg is None:
                repo_root = _find_repo_root()
                outdir_path = repo_root / "assets" / "covcmp_comets"
            else:
                outdir_path = Path(outdir_cfg)

            eval_path = Path(eval_cfg) if eval_cfg else None

            probe_cfg = ProbeConfig(
                predictor_name=predictor_name,
                sampler_name=sampler_name,
                mc_mode=mc_mode,
                outdir=outdir_path,
                eval_dir=eval_path,
                probe_tag=probe_tag,
            )

            self.probe_logger = CovarianceProbeLogger(
                config=probe_cfg,
                stock_names=list(stock_names),
                pred_len=self.decoder_length,
            )
            self.probe_enabled = True

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, n_features, encoder_length]
        out = self.generator(x, noise)
        # out.shape = [batch_size, n_features, decoder_length]
        return out

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()
        opt_g: Optimizer
        opt_d: Optimizer

        x, y_real = batch["x"], batch["y"]
        # x.shape [B, n_features, encoder_length]
        # y_real.shape [B, n_features, decoder_length]

        noise = torch.randn((x.shape[0], 1, self.encoder_length), device=self.device)
        # noise.shape = [B, 1, encoder_length]

        # Train discriminator
        if batch_idx > 0 and batch_idx % self.n_critic == 0:
            y_pred = self(x, noise)
            real_validity = self.discriminator(x, y_real)
            fake_validity = self.discriminator(x, y_pred)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            self.log("loss/discriminator", d_loss, prog_bar=True)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        # Train generator
        else:
            y_pred = self(x, noise)
            g_loss = -torch.mean(self.discriminator(x, y_pred))
            self.log("loss/generator", g_loss, prog_bar=True)
            if self.n_stocks > 1:
                self.log_corr_dist(y_real, y_pred)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

    def log_corr_dist(self, y_real: torch.Tensor, y_pred: torch.Tensor) -> None:
        corr_real, corr_pred = corr(y_real), corr(y_pred)
        metric_names = [f"corr_dist/{'-'.join(x)}" for x in combinations(self.feature_names, 2)]
        corr_distances = self.mse(corr_real, corr_pred).mean(dim=0)
        d = {metric: corr_dist.item() for metric, corr_dist in zip(metric_names, corr_distances)}
        self.log_dict(d, prog_bar=False)
        self.log('corr_dist/mean', corr_distances.mean(), prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.probe_enabled:
            self._probe_samples_batches = []
            self._probe_targets_batches = []

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if not self.probe_enabled or self.probe_logger is None:
            return
        if not self._probe_samples_batches:
            return

        samples_tensor = torch.cat(self._probe_samples_batches, dim=1).cpu()
        targets_tensor = torch.cat(self._probe_targets_batches, dim=0).cpu()

        samples_np_list: List[np.ndarray] = []
        samples_np = samples_tensor.numpy()
        for sample_block in samples_np:
            for seq in sample_block:
                arr = self.pipeline_price.inverse_transform(seq.T).T
                samples_np_list.append(arr.T)

        if not samples_np_list:
            return

        samples_nla = np.stack(samples_np_list, axis=0)

        targets_np = targets_tensor.numpy()
        target_list: List[np.ndarray] = []
        for seq in targets_np:
            arr = self.pipeline_price.inverse_transform(seq.T).T
            target_list.append(arr.T)

        if not target_list:
            return

        truth_series = np.mean(target_list, axis=0) if len(target_list) > 1 else target_list[0]

        pred_len = self.decoder_length
        if samples_nla.shape[1] > pred_len:
            samples_nla = samples_nla[:, -pred_len:, :]
        if truth_series.shape[0] > pred_len:
            truth_series = truth_series[-pred_len:, :]

        samples_nla = np.nan_to_num(samples_nla, copy=False)
        truth_series = np.nan_to_num(truth_series, copy=False)

        try:
            self.probe_logger.log_epoch(
                epoch=self.current_epoch + 1,
                samples_nla=samples_nla,
                truth_series=truth_series,
            )
        except Exception as exc:
            print(f"[probe] failed to log metrics: {exc}")

        self._probe_samples_batches = []
        self._probe_targets_batches = []

    def _collect_probe_batch(self, batch_x: torch.Tensor) -> None:
        if not self.probe_enabled or self.probe_num_samples <= 0:
            return
        batch_x = batch_x.detach()
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(0)
        total_len = batch_x.shape[-1]
        required = self.encoder_length + self.decoder_length
        if total_len < required:
            return

        start = total_len - required
        end = total_len - self.decoder_length
        context = batch_x[:, :, start:end].contiguous().to(self.device)
        target = batch_x[:, :, -self.decoder_length:].contiguous().to(self.device)

        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(self.probe_num_samples):
                noise = torch.randn((context.shape[0], 1, self.encoder_length), device=self.device)
                y_pred = self(context, noise)
                preds.append(y_pred.detach().cpu())

        if preds:
            samples = torch.stack(preds, dim=0)
            self._probe_samples_batches.append(samples)
            self._probe_targets_batches.append(target.detach().cpu())

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> None:
        x = batch['x']

        if self.probe_enabled:
            self._collect_probe_batch(x)

        x_hat = x[:, :, :self.encoder_length]

        prediction_iterations = math.ceil(self.generation_length / self.decoder_length)

        for _ in range(prediction_iterations):
            noise = torch.randn(1, 1, self.encoder_length, device=self.device)
            o = self(x_hat[:, :, -self.encoder_length:], noise)
            x_hat = torch.cat((x_hat, o), dim=2)
        
        x_hat = x_hat.squeeze().detach().cpu().numpy()
        x = x.squeeze().detach().cpu().numpy()[:, :x_hat.shape[-1]]

        x_hat_price = self.pipeline_price.inverse_transform(x_hat[:self.n_stocks].T).T if self.is_prices else None
        x_price = self.pipeline_price.inverse_transform(x[:self.n_stocks].T).T if self.is_prices else None
        if self.is_volumes:
            x_hat_volume = self.pipeline_volume.inverse_transform(x_hat[self.n_stocks:].T).T
            x_volume = self.pipeline_volume.inverse_transform(x[self.n_stocks:].T).T
        else:
            x_hat_volume = None
            x_volume = None

        path = f'{self.path_storage}/synthetic/epoch={self.current_epoch}'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/sample.pkl', 'wb') as f:
            d = dict()
            if self.is_prices:
                d.update(x_hat_price=x_hat_price, x_price=x_price)
            if self.is_volumes:
                d.update(x_hat_volume=x_hat_volume, x_volume=x_volume)
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

        def make_plot(x, x_hat, type='price'):
            fig, axes = plt.subplots(2, 2, figsize=(7, 5))
            axes = axes.ravel()
            label = True
            for ax, r, s, f in zip(axes, x, x_hat, self.stock_names):
                ax.plot(r, label='Real' if label else None, alpha=.5 if type == 'volume' else 1)
                ax.plot(s, label='Synthetic' if label else None, alpha=.5 if type == 'volume' else 1)
                label = False
                ax.set_title(f)
            fig.legend()
            fig.tight_layout()
            plt.close(fig)
            return fig

        if self.is_prices:
            fig_price = make_plot(x_price, x_hat_price)
            title_wandb = f'prices/Epoch:{self.current_epoch}'
            self.logger.experiment.log({title_wandb: wandb.Image(fig_price)})
        if self.is_volumes:
            fig_volume = make_plot(x_volume, x_hat_volume, type='volume')
            title_wandb = f'volumes/Epoch:{self.current_epoch}'
            self.logger.experiment.log({title_wandb: wandb.Image(fig_volume)})

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=1e-4)
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=3e-4)
        return opt_g, opt_d
