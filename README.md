# Neural SS-DMP

Structured neural decoding (DMP + subject dynamics) with training, testing, and visualization.

This cleaned repo only includes the M2 dataset (no Mouse/BCI/H1 data bundled).

## Project layout (current repo)
```
configs/
  default.yaml                  # default training/testing config

data/
  M2/000953/                    # M2 dataset (held-in/held-out NWB)

outputs/                        # run outputs (created by scripts)

scripts/
  test_neuralssdmp_best.py      # test best sweep model (by test r2) + full-seq viz

src/neuralssdmp/
  cli.py                        # Neural SS-DMP training entrypoint
  config.py                     # YAML loading + overrides
  data/                         # dataset loaders + preprocessing
  models/                       # encoder + DMP
  train/                        # training/eval/visualization
  utils/                        # utilities

requirements.txt                # Python dependencies
pyproject.toml                  # tooling metadata
```

## Quick start (M2 best-model test)
Run the best-model test (auto-creates a new outputs folder):
```
python scripts/test_neuralssdmp_best.py --dataset m2
```

Outputs:
```
outputs/test_best_m2_<timestamp>/
  test_metrics_best.json
  plots/test_full_seq_*.png
```

## Quick start (train M2)
Train a Neural SS-DMP model on M2 and save outputs to `outputs/`:
```
python src/neuralssdmp/cli.py --config configs/default.yaml --set data.datasets=[m2]
```

## Notes
- `test_neuralssdmp_best.py` scans `outputs/` for a sweep folder and picks the best run by **test r2**.
- If `outputs/` is empty, the script will fall back to a short training run (default 1 epoch) before testing.
