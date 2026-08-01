# Transolver for Discretization-Independent Operator Learning

This repository contains a compact Transolver training/evaluation framework for
irregular-grid operator learning. The current focus is simple, portable
experiments that start from a prebuilt `.pt` cache and produce checkpoints,
metrics, and visualizations under `runs/`.

The code keeps the original model implementations under `models/`, while the
user-facing launchers live in `scripts/transolver/public/`.

## What is included

- Quadrature-aware normalization and loss support through per-node weights.
- 1D and 2D irregular-grid smoke-test datasets under `data/`.
- YAML-driven train/eval scripts under `tests/configs/`.
- Public training and evaluation entrypoints:
  - `scripts.transolver.public.train_dp`
  - `scripts.transolver.public.train_ddp`
  - `scripts.transolver.public.evaluate`
- Evaluation outputs grouped into `best/`, `worst/`, and `other/` folders, plus
  summary and timing reports.

## Repository layout

```text
.
├── data/                         # Small synthetic .pt caches for tests
├── models/                       # Transolver model definitions
├── runs/                         # Training/evaluation outputs
├── scripts/
│   └── transolver/
│       ├── core/                 # Config parsing, cache loading, plotting helpers
│       ├── eval/                 # Evaluation implementation
│       ├── public/               # Stable command-line entrypoints
│       ├── train/                # Training loop, metrics, scheduling
│       └── viz/                  # Visualization helpers and demos
├── tests/
│   ├── configs/                  # Portable smoke-test YAML configs
│   ├── train_eval_1d_irregular.sh
│   ├── train_eval_2d_irregular.sh
│   └── test_weighted_normalizer.py
├── utils/                        # Normalization, quadrature, losses
└── requirements.txt
```

## Installation

Create an environment and install dependencies:

```bash
conda create -n transolver python=3.8 -y
conda activate transolver
pip install -r requirements.txt
```

For Linux with NVIDIA GPUs, install a PyTorch build matching your CUDA version
if the default `pip install torch` is not suitable for your machine.

## Data format

Training expects a PyTorch `.pt` file saved as a dictionary.

Required fields:

```text
pos: [B, N, D] float32   node coordinates, D=1/2/3
fx : [B, N, F] float32   input field/features
y  : [B, N, C] float32   target field
```

Optional fields:

```text
mask       : [B, N] bool      valid-node mask for padded data
node_weight: [B, N] float32   quadrature / nodal weights
case_names : list[str]        sample names
geom_id    : [B] int64        geometry id for merged datasets
metadata   : dict             optional split metadata
```

If `node_weight` is absent and `use_node_weight: 1` is enabled, the loader can
compute Delaunay lumped nodal weights from `pos`.

## Quick smoke tests

The repository includes small synthetic datasets:

```bash
bash tests/train_eval_1d_irregular.sh
bash tests/train_eval_2d_irregular.sh
```

If your conda environment has a different name, override it:

```bash
TRANSOLVER_CONDA_ENV=my_env bash tests/train_eval_1d_irregular.sh
```

Or run the Python entrypoints directly:

```bash
python -m scripts.transolver.public.train_dp \
  --config tests/configs/train_eval_1d_irregular.yaml

python -m scripts.transolver.public.evaluate \
  --config tests/configs/train_eval_1d_irregular.yaml \
  --load_ckpt runs/synth_1d_n64_q/checkpoints/synth_1d_n64_q_best_eval.pt
```

## Training on your own `.pt` cache

Copy one of the test configs and edit these fields:

```yaml
key:
  save_name: "my_run"
  run_dir: "runs/my_run"

data:
  cache_path: "data/my_dataset.pt"
  coord_norm_path: "runs/my_run/coord_norm.pt"
  use_node_weight: 1
  output_cols: ["u"]

model:
  name: "Transolver_2D"

train:
  batch_size: 8
  epochs: 100
  lr: 0.001
```

Then launch:

```bash
python -m scripts.transolver.public.train_dp --config path/to/config.yaml
```

For multi-GPU distributed training:

```bash
python -m scripts.transolver.public.train_ddp --config path/to/config.yaml
```

## Quadrature weights

For irregular grids, pointwise averaging can make dense regions dominate the
loss and normalization statistics. Enabling node weights makes the training
objective closer to a continuous-domain integral:

```yaml
data:
  use_node_weight: 1
  persist_node_weight: 0
  node_weight_max_edge: 0.0
```

Recommended behavior:

- If the dataset already contains `node_weight`, the loader uses it directly.
- If not, Delaunay lumped weights are computed from `pos`.
- Use `mask` when samples are padded to a common node count.

## Evaluation outputs

Evaluation writes results under the run directory, usually:

```text
runs/<run_name>/eval/paper_eval_<timestamp>/
├── best/
├── worst/
├── other/
├── summary_report.txt
├── time.txt
└── high_stress_*.txt
```

The `best/` and `worst/` folders contain selected case plots. The `other/`
folder contains additional plots such as distribution and high-stress
visualizations.

## Portability notes

The code is designed to run from the repository root with relative paths in
YAML configs. When moving to Linux or another machine, normally only these
items need attention:

1. Install a PyTorch build compatible with the new machine.
2. Put `.pt` datasets under `data/`, or update `data.cache_path` in the config.
3. Set `TRANSOLVER_CONDA_ENV` if using the provided shell scripts with a
   different conda environment name.
4. Keep output paths relative, for example `runs/my_run`, to avoid machine-
   specific paths.

## Development checks

Run the lightweight normalizer tests:

```bash
python -m unittest discover -s tests -p "test_weighted_normalizer.py"
```

Check the command-line entrypoints:

```bash
python -m scripts.transolver.public.train_dp --help
python -m scripts.transolver.public.evaluate --help
```
