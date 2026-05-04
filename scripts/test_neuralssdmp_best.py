"""Test the best Neural SS-DMP run (by test r2) for a dataset and render test full-seq viz."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.append(SRC)

from neuralssdmp.cli import (  # noqa: E402
    _get_target_stats,
    _infer_input_type,
    _resolve_full_sequence_cfg,
    _safe_torch_load,
    _save_full_sequence,
)
from neuralssdmp.config import load_yaml
from neuralssdmp.data.datamodule import build_datasets  
from neuralssdmp.models import NeuralDMPModel  
from neuralssdmp.train.trainer import evaluate, get_device, train_one_epoch  
from neuralssdmp.utils.seed import set_seed 


def _set(cfg: dict[str, Any], key: str, value: Any) -> None:
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_dir(paths: list[str]) -> str | None:
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths[-1]


def _find_latest_checkpoint(search_root: str) -> str | None:
    if not os.path.isdir(search_root):
        return None
    skip_dirs = {"data", ".git", "__pycache__", ".tmp"}
    best_path = None
    best_mtime = -1.0
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if "last.pt" in files:
            path = os.path.join(root, "last.pt")
            mtime = os.path.getmtime(path)
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = path
    return best_path


def _resolve_ckpt_path(ckpt_arg: str | None) -> str | None:
    if not ckpt_arg:
        return None
    path = os.path.abspath(ckpt_arg)
    if os.path.isdir(path):
        candidate = os.path.join(path, "last.pt")
        return candidate if os.path.isfile(candidate) else None
    return path if os.path.isfile(path) else None


def _find_overrides_near(path: str) -> str | None:
    if not path:
        return None
    if os.path.isfile(path):
        base = os.path.dirname(path)
    else:
        base = path
    candidates = [
        base,
        os.path.dirname(base),
        os.path.dirname(os.path.dirname(base)),
        os.path.dirname(os.path.dirname(os.path.dirname(base))),
    ]
    for cand in candidates:
        if not cand:
            continue
        overrides_path = os.path.join(cand, "overrides.json")
        if os.path.isfile(overrides_path):
            return overrides_path
    return None


def _find_best_run(dataset: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    outputs_root = os.path.join(ROOT, "outputs")
    if not os.path.isdir(outputs_root):
        raise RuntimeError("outputs directory not found.")

    candidates = []
    for name in os.listdir(outputs_root):
        if not name.startswith("sweep_"):
            continue
        if "baseline" in name or "baselines" in name:
            continue
        if dataset not in name:
            continue
        candidates.append(os.path.join(outputs_root, name))
    sweep_root = _latest_dir(candidates)
    if sweep_root is None:
        raise RuntimeError(f"No sweep directory found for dataset '{dataset}'.")

    best = None
    best_r2 = -1.0e18
    best_overrides = None
    best_metrics = None

    for variant in os.listdir(sweep_root):
        variant_dir = os.path.join(sweep_root, variant)
        if not os.path.isdir(variant_dir):
            continue
        overrides_path = os.path.join(variant_dir, "overrides.json")
        if not os.path.isfile(overrides_path):
            continue
        overrides = _load_json(overrides_path)
        ds_list = overrides.get("data.datasets")
        if ds_list is not None and dataset not in ds_list:
            continue
        run_dirs = [
            os.path.join(variant_dir, d)
            for d in os.listdir(variant_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(variant_dir, d))
        ]
        run_dir = _latest_dir(run_dirs)
        if run_dir is None:
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        metrics = _load_json(metrics_path)
        if "r2" not in metrics:
            continue
        r2 = float(metrics["r2"])
        if r2 > best_r2:
            best_r2 = r2
            best = run_dir
            best_overrides = overrides
            best_metrics = metrics

    if best is None or best_overrides is None or best_metrics is None:
        raise RuntimeError(f"No valid runs with r2 found in {sweep_root}.")
    return best, best_overrides, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="m2")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (last.pt) or a run directory.")
    parser.add_argument("--overrides", type=str, default=None, help="Optional overrides.json to apply.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--heldout-only",
        dest="heldout_only",
        action="store_true",
        help="Use held-out split for train/val/test and compute normalization from held-out only (default).",
    )
    group.add_argument(
        "--use-heldin",
        dest="heldout_only",
        action="store_false",
        help="Use held-in for train/val stats and held-out for test (original behavior).",
    )
    parser.set_defaults(heldout_only=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--fallback-epochs", type=int, default=1)
    args = parser.parse_args()

    run_dir = None
    overrides: Dict[str, Any] = {}
    prev_metrics: Dict[str, Any] | None = None
    ckpt_path = _resolve_ckpt_path(args.ckpt)
    if ckpt_path:
        run_dir = os.path.dirname(ckpt_path)
        print(f"Using checkpoint: {ckpt_path}")
        overrides_path = args.overrides or _find_overrides_near(run_dir)
        if overrides_path:
            overrides = _load_json(overrides_path)
            print(f"Loaded overrides: {overrides_path}")
    else:
        try:
            run_dir, overrides, prev_metrics = _find_best_run(args.dataset)
            print(f"Best run (by test r2) for {args.dataset}: {run_dir}")
            if prev_metrics is not None:
                print(f"Best r2 from sweep: {prev_metrics.get('r2')}")
        except RuntimeError as exc:
            print(f"warning: {exc}")
            latest_ckpt = _find_latest_checkpoint(ROOT)
            if latest_ckpt:
                run_dir = os.path.dirname(latest_ckpt)
                ckpt_path = latest_ckpt
                print(f"warning: using latest checkpoint: {latest_ckpt}")
                overrides_path = args.overrides or _find_overrides_near(run_dir)
                if overrides_path:
                    overrides = _load_json(overrides_path)
                    print(f"Loaded overrides: {overrides_path}")
            else:
                print("warning: no checkpoint found, falling back to a short train run.")

    cfg = load_yaml("configs/default.yaml")
    if overrides:
        for key, value in overrides.items():
            _set(cfg, key, value)
    else:
        _set(cfg, "data.datasets", [args.dataset])
        if run_dir is None:
            _set(cfg, "train.num_epochs", max(int(args.fallback_epochs), 1))

    if args.heldout_only:
        _set(cfg, "data.datasets", [args.dataset])
        dataset_cfg = cfg["data"].get(args.dataset, {})
        if "falcon_split" in dataset_cfg:
            _set(cfg, f"data.{args.dataset}.split_strategy", "falcon")
            _set(cfg, f"data.{args.dataset}.falcon_split.train_split", "held-out-calib")
            _set(cfg, f"data.{args.dataset}.falcon_split.test_split", "held-out-calib")
            _set(cfg, f"data.{args.dataset}.falcon_split.val_frac", 0.0)
        else:
            print(f"warning: --heldout-only requested but dataset '{args.dataset}' has no falcon_split.")

    set_seed(cfg["project"]["seed"])

    device = get_device(cfg)
    train_ds, val_ds, test_ds, dt = build_datasets(cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    sample = next(iter(test_loader))
    input_dim = sample["x"].shape[-1]
    input_dim_psd = None
    if "x_psd" in sample:
        input_dim_psd = sample["x_psd"].shape[-1]
    out_dim = sample["y_future"].shape[-1]

    model_cfg = dict(cfg["model"])
    model_cfg["feature_mode"] = cfg.get("preprocess", {}).get("ecog_features", "raw")
    model_cfg["input_type"] = _infer_input_type(cfg)
    if model_cfg["input_type"] == "spike":
        model_cfg["feature_mode"] = "raw"
    model_cfg["psd"] = cfg.get("preprocess", {}).get("psd", {})
    model = NeuralDMPModel(input_dim, out_dim, model_cfg, input_dim_psd=input_dim_psd).to(device)

    if run_dir is not None:
        if ckpt_path is None:
            ckpt_path = os.path.join(run_dir, "last.pt")
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
        model.load_state_dict(_safe_torch_load(ckpt_path, map_location=device))
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=True,
        )
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        epochs = max(int(args.fallback_epochs), 1)
        for epoch in range(epochs):
            tr_loss = train_one_epoch(model, train_loader, optim, dt, cfg, device)
            print(f"fallback epoch {epoch+1} train {tr_loss:.4f}")

    te_loss, te_metrics, te_sequences = evaluate(model, test_loader, dt, cfg, device)
    print(f"test loss {te_loss:.4f} metrics {te_metrics}")

    target_mean, target_std = _get_target_stats(cfg)
    if args.out_dir is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(ROOT, "outputs", f"test_best_{args.dataset}_{tag}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    full_seq_cfg = _resolve_full_sequence_cfg(cfg)
    if full_seq_cfg["enabled"]:
        _save_full_sequence(
            model,
            test_ds,
            dt,
            out_dir,
            None,
            device,
            cfg,
            split_name="test",
            target_mean=target_mean,
            target_std=target_std,
            full_seq_cfg=full_seq_cfg,
        )

    metrics_out = os.path.join(out_dir, "test_metrics_best.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump({"test_loss": te_loss, **te_metrics}, f, indent=2)
    print(f"saved metrics: {metrics_out}")
    print(f"saved plots: {os.path.join(out_dir, 'plots')}")


if __name__ == "__main__":
    main()
