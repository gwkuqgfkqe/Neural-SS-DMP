# Neural SS-DMP

Anonymous code repository for the Neural SS-DMP experiments on the M2 dataset.
This repository contains the code and configuration files only. The M2 data is
not included. The trained checkpoint exceeds the 25 MB upload limit for the
anonymous code repository and is provided in the supplementary materials
submitted with the manuscript.

## Contents

```text
README.md

configs/
  default.yaml                  # Default training and evaluation config

data/                           # M2 dataset location after download

outputs/
  test_best_m2/
    test_metrics_best.json      # Evaluation metrics from the supplied checkpoint
    plots/
      test_full_seq_*.png       # Full-sequence prediction plots

scripts/
  test_neuralssdmp_best.py      # Evaluation script

src/neuralssdmp/
  __init__.py
  cli.py                        # Training entry point
  config.py                     # YAML loading and command-line overrides
  data/
    __init__.py
    base.py                     # Dataset base classes
    datamodule.py               # Dataset assembly, preprocessing, and splits
    m2.py                       # M2 loader
    h1.py                       # H1 loader
    mouse.py                    # Mouse loader
    bci_iv.py                   # BCI IV loader
    utils.py                    # Data preprocessing utilities
  models/
    __init__.py
    dmp.py                      # DMP rollout module
    encoder.py                  # Neural encoders
    model.py                    # Neural SS-DMP model wrapper
  train/
    __init__.py
    dynamics.py                 # Subject dynamics fitting
    losses.py                   # Training losses
    metrics.py                  # Evaluation metrics
    trainer.py                  # Training and evaluation loops
    visualizer.py               # Plotting utilities
  utils/
    __init__.py
    seed.py                     # Random seed utility

requirements.txt                # Python dependencies
pyproject.toml                  # Project metadata
```

Empty `data/` and `outputs/` directories may not appear after cloning. Create
them before adding data or checkpoints.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Download the M2 dataset from DANDI:

```text
https://dandiarchive.org/dandiset/000953
```

Place the files under:

```text
data/M2/000953/
```

The M2 loader expects NWB files in:

```text
data/M2/000953/sub-MonkeyN-held-in-calib/*.nwb
data/M2/000953/sub-MonkeyN-held-out-calib/*.nwb
```

## Evaluate the Supplementary Checkpoint

After obtaining the trained checkpoint from the submitted supplementary
materials, place it at:

```text
outputs/run_pretrained/last.pt
```

Run:

```bash
python scripts/test_neuralssdmp_best.py --dataset m2 --ckpt outputs/run_pretrained/last.pt
```

The evaluation script writes:

```text
outputs/test_best_m2_<timestamp>/
  test_metrics_best.json
  plots/test_full_seq_*.png
```

## Train from Scratch

```bash
python src/neuralssdmp/cli.py --config configs/default.yaml --set data.datasets=[m2]
```

Training writes:

```text
outputs/run_<timestamp>/
  last.pt
  metrics.json
  plots/
```

`last.pt` is the final-epoch model. The training code records `val_best_loss`
in `metrics.json`, but it does not write a separate `best.pt`.

To evaluate a newly trained model:

```bash
python scripts/test_neuralssdmp_best.py --dataset m2 --ckpt outputs/run_<timestamp>/last.pt --use-heldin
```

## Split and Checkpoint Notes

With `configs/default.yaml`, training uses `held-in-calib` for train/validation
and `held-out-calib` for test. The validation split is 20% of `held-in-calib`.

`scripts/test_neuralssdmp_best.py` defaults to `--heldout-only`, which sets both
train and test split sources to `held-out-calib` before building datasets. Use
`--use-heldin` to keep the held-in/held-out split from `configs/default.yaml`.

Pass `--ckpt` for reproducible evaluation. If `--ckpt` is omitted, the test
script first searches `outputs/sweep_*`, then searches the project for the
newest `last.pt`, and finally falls back to a one-epoch in-memory training run
if no checkpoint is found.
