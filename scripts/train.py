from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import GNCEConfig, load_config
from src.data_utils import prepare_gnce_data
from src.models import TripleModel


logger = logging.getLogger("gnce.train")


def _setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )


def _get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _q_error(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = torch.clamp(pred, min=eps)
    t = torch.clamp(true, min=eps)
    return torch.maximum(p / t, t / p)


@torch.no_grad()
def _run_eval_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    model.eval()
    total_n = 0
    total_loss_sum = 0.0
    total_mae_sum = 0.0
    total_qerr_sum = 0.0

    by_ds: Dict[str, Dict[str, float]] = {}

    for meta, batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        y = batch.y.to(torch.float32)
        log_y = torch.log(torch.clamp(y, min=1e-7)).to(torch.float64)

        pred_log = model(
            batch.x.to(torch.float64),
            batch.edge_index,
            getattr(batch, "edge_type", None),
            batch.edge_attr.to(torch.float64),
            batch=batch.batch,
        ).to(torch.float64)

        pred = torch.exp(pred_log).squeeze(-1).to(torch.float32)
        n = int(y.numel())
        if n <= 0:
            continue


        loss_sum = float(F.mse_loss(pred_log.squeeze(-1), log_y, reduction="sum").item())
        mae_sum = float(torch.sum(torch.abs(pred - y)).item())
        qerr_sum = float(torch.sum(_q_error(pred, y)).item())

        total_n += n
        total_loss_sum += loss_sum
        total_mae_sum += mae_sum
        total_qerr_sum += qerr_sum

        ds_name = getattr(meta, "dataset_name", "unknown")
        slot = by_ds.setdefault(ds_name, {"n": 0.0, "loss_sum": 0.0, "mae_sum": 0.0, "qerr_sum": 0.0})
        slot["n"] += float(n)
        slot["loss_sum"] += loss_sum
        slot["mae_sum"] += mae_sum
        slot["qerr_sum"] += qerr_sum

    overall = {
        "loss": (total_loss_sum / total_n) if total_n else float("nan"),
        "mae": (total_mae_sum / total_n) if total_n else float("nan"),
        "q_error": (total_qerr_sum / total_n) if total_n else float("nan"),
    }

    by_ds_out: Dict[str, Dict[str, float]] = {}
    for ds, slot in by_ds.items():
        n = float(slot["n"])
        by_ds_out[ds] = {
            "loss": (float(slot["loss_sum"]) / n) if n else float("nan"),
            "mae": (float(slot["mae_sum"]) / n) if n else float("nan"),
            "q_error": (float(slot["qerr_sum"]) / n) if n else float("nan"),
        }

    return overall, by_ds_out


def _save_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
) -> Tuple[List[float], List[float], List[int], List[str]]:
    model.eval()

    preds: List[float] = []
    gts: List[float] = []
    sizes: List[int] = []
    dataset_names: List[str] = []

    with torch.no_grad():
        for meta, batch in tqdm(loader, desc="predict", leave=False):
            batch = batch.to(device)
            y = batch.y.to(torch.float32)

            pred_log = model(
                batch.x.to(torch.float64),
                batch.edge_index,
                getattr(batch, "edge_type", None),
                batch.edge_attr.to(torch.float64),
                batch=batch.batch,
            ).to(torch.float64)

            pred = torch.exp(pred_log).squeeze(-1).to(torch.float32)
            preds.extend(pred.detach().cpu().numpy().tolist())
            gts.extend(y.detach().cpu().numpy().tolist())
            dataset_names.extend([getattr(meta, "dataset_name", "unknown")] * int(y.numel()))

            num_triples = getattr(batch, "num_triples", None)
            if num_triples is None:
                sizes.extend([0] * int(y.numel()))
            else:
                sizes.extend(torch.as_tensor(num_triples).detach().cpu().numpy().astype(int).tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "preds.npy", np.array(preds))
    np.save(out_dir / "gts.npy", np.array(gts))
    np.save(out_dir / "sizes.npy", np.array(sizes))
    (out_dir / "dataset_names.json").write_text(json.dumps(dataset_names, indent=2), encoding="utf-8")
    return preds, gts, sizes, dataset_names


def _plot_train_val_curves(history: List[Dict[str, float]], out_dir: Path) -> None:
    """
    Writes/overwrites:
      - loss.png
      - mae.png
      - q_error.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _series(key: str) -> List[float]:
        return [float(h.get(key, float("nan"))) for h in history]

    epochs = [int(h.get("epoch", i + 1)) for i, h in enumerate(history)]

    def _plot(metric_name: str, train_key: str, val_key: str, fname: str) -> None:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, _series(train_key), label="train", linewidth=2)
        plt.plot(epochs, _series(val_key), label="val", linewidth=2)
        plt.xlabel("epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name}: train vs val")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    _plot("loss (MSE on log-cardinality)", "train_loss", "val_loss", "loss.png")
    _plot("MAE (cardinality)", "train_mae", "val_mae", "mae.png")
    _plot("q-error", "train_q_error", "val_q_error", "q_error.png")


def _plot_log_scatter(true_cards: List[float], pred_cards: List[float], out_path: Path) -> None:
    """
    Scatter plot: log(true) vs log(pred).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(true_cards, dtype=np.float64)
    p = np.asarray(pred_cards, dtype=np.float64)
    eps = 1e-7
    lt = np.log10(np.clip(t, eps, None))
    lp = np.log10(np.clip(p, eps, None))

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(lt, lp, s=8, alpha=0.25, edgecolors="none")
    lo = float(np.nanmin([np.nanmin(lt), np.nanmin(lp)]))
    hi = float(np.nanmax([np.nanmax(lt), np.nanmax(lp)]))
    if np.isfinite(lo) and np.isfinite(hi):
        plt.plot([lo, hi], [lo, hi], "--", linewidth=1.5, color="black", alpha=0.6, label="y=x")
    plt.xlabel("log10(true cardinality)")
    plt.ylabel("log10(pred cardinality)")
    plt.title("Validation: log(true) vs log(pred)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train(config: GNCEConfig, run_dir: Path):
    log_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_dir)

    device = _get_device(config.device)
    _set_seeds(config.seed)

    logger.info(f"Device: {device} (cuda_available={torch.cuda.is_available()})")
    logger.info(f"Run dir: {run_dir}")

    # Save config snapshot
    cfg_out = run_dir / "config.json"
    cfg_out.write_text(
        json.dumps(
            {
                **asdict(config),
                "data": [asdict(dc) for dc in config.data],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    data = prepare_gnce_data(config)
    train_loader = data.train_loader
    val_loader = data.validation_loader
    test_loader = data.test_loader

    model = TripleModel().to(device).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []
    history_by_dataset: Dict[str, List[Dict[str, float]]] = {}
    dataset_names = sorted({dc.dataset_name for dc in config.data})

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total_n = 0
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_qerr_sum = 0.0

        train_by_ds: Dict[str, Dict[str, float]] = {}

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{config.epochs}", leave=False)
        for meta, batch in pbar:
            batch = batch.to(device)
            y = batch.y.to(torch.float32)
            log_y = torch.log(torch.clamp(y, min=1e-7)).to(torch.float64)

            optimizer.zero_grad(set_to_none=True)
            pred_log = model(
                batch.x.to(torch.float64),
                batch.edge_index,
                getattr(batch, "edge_type", None),
                batch.edge_attr.to(torch.float64),
                batch=batch.batch,
            ).to(torch.float64)

            loss = F.mse_loss(pred_log.squeeze(-1), log_y)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())

            with torch.no_grad():
                pred = torch.exp(pred_log).squeeze(-1).to(torch.float32)
                n = int(y.numel())
                if n > 0:
                    loss_sum = float(F.mse_loss(pred_log.squeeze(-1), log_y, reduction="sum").item())
                    mae_sum = float(torch.sum(torch.abs(pred - y)).item())
                    qerr_sum = float(torch.sum(_q_error(pred, y)).item())

                    train_total_n += n
                    train_loss_sum += loss_sum
                    train_mae_sum += mae_sum
                    train_qerr_sum += qerr_sum

                    ds_name = getattr(meta, "dataset_name", "unknown")
                    slot = train_by_ds.setdefault(ds_name, {"n": 0.0, "loss_sum": 0.0, "mae_sum": 0.0, "qerr_sum": 0.0})
                    slot["n"] += float(n)
                    slot["loss_sum"] += loss_sum
                    slot["mae_sum"] += mae_sum
                    slot["qerr_sum"] += qerr_sum

            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        train_loss = (train_loss_sum / train_total_n) if train_total_n else float("nan")
        train_mae = (train_mae_sum / train_total_n) if train_total_n else float("nan")
        train_q_error = (train_qerr_sum / train_total_n) if train_total_n else float("nan")

        metrics: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_q_error": train_q_error,
        }

        val_metrics, val_by_ds = _run_eval_epoch(model, val_loader, device=device)
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={metrics['train_loss']:.6f} train_mae={metrics['train_mae']:.6f} train_q={metrics['train_q_error']:.6f} | "
            f"val_loss={metrics.get('val_loss', float('nan')):.6f} val_mae={metrics.get('val_mae', float('nan')):.6f} val_q={metrics.get('val_q_error', float('nan')):.6f}"
        )

        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info(f"New best model saved (val_loss={best_val_loss:.6f})")

        history.append(metrics)

        # Plots after every epoch
        _plot_train_val_curves(history, out_dir=run_dir / "plots")
        val_preds, val_gts, _val_sizes, val_ds_names = _save_predictions(
            model, val_loader, device=device, out_dir=run_dir / "val_predictions" / f"epoch_{epoch:04d}"
        )
        _plot_log_scatter(
            true_cards=val_gts,
            pred_cards=val_preds,
            out_path=run_dir / "scatter" / f"epoch_{epoch:04d}.png",
        )

        # Per-dataset histories + plots (curves + scatter)
        for ds in dataset_names:
            # Train per-ds
            tslot = train_by_ds.get(ds, {"n": 0.0, "loss_sum": 0.0, "mae_sum": 0.0, "qerr_sum": 0.0})
            tn = float(tslot["n"])
            ds_train_loss = (float(tslot["loss_sum"]) / tn) if tn else float("nan")
            ds_train_mae = (float(tslot["mae_sum"]) / tn) if tn else float("nan")
            ds_train_q = (float(tslot["qerr_sum"]) / tn) if tn else float("nan")

            # Val per-ds
            vslot = val_by_ds.get(ds, {"loss": float("nan"), "mae": float("nan"), "q_error": float("nan")})
            ds_val_loss = float(vslot.get("loss", float("nan")))
            ds_val_mae = float(vslot.get("mae", float("nan")))
            ds_val_q = float(vslot.get("q_error", float("nan")))

            history_by_dataset.setdefault(ds, []).append(
                {
                    "epoch": float(epoch),
                    "train_loss": ds_train_loss,
                    "train_mae": ds_train_mae,
                    "train_q_error": ds_train_q,
                    "val_loss": ds_val_loss,
                    "val_mae": ds_val_mae,
                    "val_q_error": ds_val_q,
                }
            )

            _plot_train_val_curves(history_by_dataset[ds], out_dir=run_dir / "plots" / "by_dataset" / ds)

            ds_true: List[float] = []
            ds_pred: List[float] = []
            for name, t, p in zip(val_ds_names, val_gts, val_preds):
                if name == ds:
                    ds_true.append(t)
                    ds_pred.append(p)
            if ds_true:
                _plot_log_scatter(
                    true_cards=ds_true,
                    pred_cards=ds_pred,
                    out_path=run_dir / "scatter" / "by_dataset" / ds / f"epoch_{epoch:04d}.png",
                )

        if epoch % config.save_every == 0 or epoch == config.epochs:
            (run_dir / "training_progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            (run_dir / "training_progress_by_dataset.json").write_text(
                json.dumps(history_by_dataset, indent=2), encoding="utf-8"
            )
            torch.save(model.state_dict(), ckpt_dir / f"checkpoint_epoch_{epoch}.pt")

    # Final test with best model
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))

    test_metrics, _test_by_ds = _run_eval_epoch(model, test_loader, device=device)
    logger.info(f"Final test: loss={test_metrics['loss']:.6f} mae={test_metrics['mae']:.6f} q={test_metrics['q_error']:.6f}")

    _save_predictions(model, test_loader, device=device, out_dir=run_dir / "test_predictions")
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gnce_local.yaml")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run directory name (default: run_<timestamp>)")
    args = parser.parse_args()

    config = load_config(args.config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{args.run_name}_{timestamp}" if args.run_name else f"run_{timestamp}"
    run_dir = Path(config.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train(config, run_dir)


if __name__ == "__main__":
    main()
