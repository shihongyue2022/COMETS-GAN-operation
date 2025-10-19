from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sys

def _discover_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'bin').exists():
            return parent
    return current.parents[-1]

_REPO_ROOT = _discover_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from typing import Dict, List, Optional

import numpy as np

try:
    from bin._matrix_metrics import compute_all_matrix_metrics, METRIC_KEYS_ORDER
except Exception:  # pragma: no cover
    from _matrix_metrics import compute_all_matrix_metrics, METRIC_KEYS_ORDER

try:
    from bin._series_metrics import compute_series_metrics
except Exception:  # pragma: no cover
    from _series_metrics import compute_series_metrics


@dataclass
class ProbeConfig:
    predictor_name: str = "COMETS-GAN"
    sampler_name: str = ""
    mc_mode: str = "flatten"
    outdir: Path = Path("covcmp_comets")
    eval_dir: Optional[Path] = None
    probe_tag: Optional[str] = None


class CovarianceProbeLogger:
    def __init__(
        self,
        config: ProbeConfig,
        stock_names: List[str],
        pred_len: int,
    ) -> None:
        self.cfg = config
        self.stock_names = stock_names
        self.pred_len = pred_len

        self.root = self.cfg.outdir.expanduser().resolve()
        self.history_dir = self.root / "probe_history"
        self.eval_dir = self.cfg.eval_dir.expanduser().resolve() if self.cfg.eval_dir else None
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if self.eval_dir is not None:
            self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.history_dir / "summary_history.csv"
        self.matrix_path = self.history_dir / "matrix_history.csv"
        self.series_path = self.history_dir / "series_history.csv"
        self._ensure_headers()

        self.version_index = self._next_version_index()

    def _ensure_headers(self) -> None:
        if not self.summary_path.exists():
            with self.summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "epochs", "probe_tag", "predictor_name", "sampler_name", "mc_mode",
                    "outdir", "predictor", "assets", "mse_mean", "mae_mean", "mape_mean", "smape_mean",
                    "cover80_mean", "cover95_mean",
                ])
        if not self.matrix_path.exists():
            with self.matrix_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "epochs", "probe_tag", "predictor_name", "sampler_name", "mc_mode",
                    "outdir", "name", "predictor", "sampler", "assets", "pred_len",
                ] + METRIC_KEYS_ORDER)
        if not self.series_path.exists():
            with self.series_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "epochs", "probe_tag", "predictor_name", "sampler_name", "mc_mode",
                    "outdir", "mse", "mae", "mape", "smape", "cover80", "cover95", "asset", "item_id",
                    "predictor", "samples", "pred_len",
                ])

    def _next_version_index(self) -> int:
        existing = [p for p in self.root.glob("version_*") if p.is_dir()]
        if not existing:
            return 0
        indices = []
        for p in existing:
            try:
                idx = int(p.name.split("_")[-1])
            except ValueError:
                continue
            indices.append(idx)
        return max(indices) + 1 if indices else 0

    def _make_version_dir(self) -> Path:
        path = self.root / f"version_{self.version_index}"
        path.mkdir(parents=True, exist_ok=True)
        self.version_index += 1
        if self.eval_dir is not None:
            copy_dir = self.eval_dir / path.name
            copy_dir.mkdir(parents=True, exist_ok=True)
        return path

    def log_epoch(
        self,
        epoch: int,
        samples_nla: np.ndarray,
        truth_series: np.ndarray,
        probe_tag: Optional[str] = None,
    ) -> None:
        if samples_nla.ndim != 3:
            raise ValueError("samples_nla must be (num_samples, pred_len, num_assets)")
        if truth_series.shape != (self.pred_len, len(self.stock_names)):
            raise ValueError("truth_series shape mismatch with pred_len/assets")

        ts = datetime.utcnow().isoformat(timespec="seconds")
        tag = probe_tag or self.cfg.probe_tag or f"comets_epoch{epoch:04d}"
        version_dir = self._make_version_dir()

        np.save(version_dir / "samples.npy", samples_nla)
        np.save(version_dir / "truth.npy", truth_series)

        eval_version_dir = self.eval_dir / version_dir.name if self.eval_dir is not None else None
        if eval_version_dir is not None:
            eval_version_dir.mkdir(parents=True, exist_ok=True)
            np.save(eval_version_dir / "samples.npy", samples_nla)
            np.save(eval_version_dir / "truth.npy", truth_series)

        matrix_rows, summary_row, series_rows = self._compute_tables(samples_nla, truth_series, ts, epoch, tag, version_dir)

        with self.matrix_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "epochs", "probe_tag", "predictor_name", "sampler_name", "mc_mode",
                "outdir", "name", "predictor", "sampler", "assets", "pred_len",
            ] + METRIC_KEYS_ORDER)
            for row in matrix_rows:
                writer.writerow(row)

        with self.summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "epochs", "probe_tag", "predictor_name", "sampler_name", "mc_mode",
                "outdir", "predictor", "assets", "mse_mean", "mae_mean", "mape_mean", "smape_mean",
                "cover80_mean", "cover95_mean",
            ])
            writer.writerow(summary_row)

        with self.series_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "epochs", "probe_tag", "predictor_name", "sampler_name", "mc_mode",
                "outdir", "mse", "mae", "mape", "smape", "cover80", "cover95", "asset", "item_id",
                "predictor", "samples", "pred_len",
            ])
            for row in series_rows:
                writer.writerow(row)

        metrics_snapshot = {
            "timestamp": ts,
            "epoch": epoch,
            "probe_tag": tag,
            "matrix": matrix_rows,
            "summary": summary_row,
            "series": series_rows,
        }
        with (version_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_snapshot, f, ensure_ascii=False, indent=2)
        if eval_version_dir is not None:
            with (eval_version_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_snapshot, f, ensure_ascii=False, indent=2)

    # ----------------- helpers -----------------
    def _compute_tables(
        self,
        samples_nla: np.ndarray,
        truth_series: np.ndarray,
        timestamp: str,
        epoch: int,
        probe_tag: str,
        version_dir: Path,
    ) -> tuple[List[Dict[str, float]], Dict[str, float], List[Dict[str, float]]]:
        num_assets = len(self.stock_names)
        num_samples = samples_nla.shape[0]

        C_true = np.cov(truth_series, rowvar=False)

        mean_curve = samples_nla.mean(axis=0)
        C_pred_time = np.cov(mean_curve, rowvar=False)

        if self.cfg.mc_mode == "sample_mean":
            sample_means = samples_nla.mean(axis=1)  # (S, num_assets)
            C_pred_mc = np.cov(sample_means, rowvar=False)
        else:  # flatten
            flattened = samples_nla.reshape(samples_nla.shape[0] * samples_nla.shape[1], num_assets)
            C_pred_mc = np.cov(flattened, rowvar=False)

        matrix_rows = []
        for name, C_hat in ("pred_time", C_pred_time), ("pred_mc", C_pred_mc):
            metrics = compute_all_matrix_metrics(C_true, C_hat)
            row = {
                "timestamp": timestamp,
                "epochs": epoch,
                "probe_tag": probe_tag,
                "predictor_name": self.cfg.predictor_name,
                "sampler_name": self.cfg.sampler_name,
                "mc_mode": self.cfg.mc_mode,
                "outdir": str(version_dir),
                "name": name,
                "predictor": self.cfg.predictor_name,
                "sampler": self.cfg.sampler_name,
                "assets": num_assets,
                "pred_len": self.pred_len,
            }
            for key in METRIC_KEYS_ORDER:
                row[key] = metrics.get(key, float("nan"))
            matrix_rows.append(row)

        series_rows: List[Dict[str, float]] = []
        mse_vals = []
        mae_vals = []
        mape_vals = []
        smape_vals = []
        cover80_vals = []
        cover95_vals = []

        for idx, name in enumerate(self.stock_names):
            y_true = truth_series[:, idx]
            samples_asset = samples_nla[:, :, idx]
            series_metrics = compute_series_metrics(y_true, samples_asset)
            series_rows.append({
                "timestamp": timestamp,
                "epochs": epoch,
                "probe_tag": probe_tag,
                "predictor_name": self.cfg.predictor_name,
                "sampler_name": self.cfg.sampler_name,
                "mc_mode": self.cfg.mc_mode,
                "outdir": str(version_dir),
                "mse": series_metrics["mse"],
                "mae": series_metrics["mae"],
                "mape": series_metrics["mape"],
                "smape": series_metrics["smape"],
                "cover80": series_metrics["cover80"],
                "cover95": series_metrics["cover95"],
                "asset": idx,
                "item_id": name,
                "predictor": self.cfg.predictor_name,
                "samples": num_samples,
                "pred_len": self.pred_len,
            })
            mse_vals.append(series_metrics["mse"])
            mae_vals.append(series_metrics["mae"])
            mape_vals.append(series_metrics["mape"])
            smape_vals.append(series_metrics["smape"])
            cover80_vals.append(series_metrics["cover80"])
            cover95_vals.append(series_metrics["cover95"])

        summary_row = {
            "timestamp": timestamp,
            "epochs": epoch,
            "probe_tag": probe_tag,
            "predictor_name": self.cfg.predictor_name,
            "sampler_name": self.cfg.sampler_name,
            "mc_mode": self.cfg.mc_mode,
            "outdir": str(version_dir),
            "predictor": self.cfg.predictor_name,
            "assets": num_assets,
            "mse_mean": float(np.mean(mse_vals)) if mse_vals else float("nan"),
            "mae_mean": float(np.mean(mae_vals)) if mae_vals else float("nan"),
            "mape_mean": float(np.mean(mape_vals)) if mape_vals else float("nan"),
            "smape_mean": float(np.mean(smape_vals)) if smape_vals else float("nan"),
            "cover80_mean": float(np.mean(cover80_vals)) if cover80_vals else float("nan"),
            "cover95_mean": float(np.mean(cover95_vals)) if cover95_vals else float("nan"),
        }

        return matrix_rows, summary_row, series_rows
