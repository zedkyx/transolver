from __future__ import annotations

import argparse
import os
from datetime import datetime

import yaml


LEGACY_REPO_PREFIX = "/root/transolver"


def _repo_root() -> str:
    # scripts/transolver/core/argparser.py -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _project_root() -> str:
    return _repo_root()


def _rewrite_legacy_paths(value, repo_root: str):
    if isinstance(value, str):
        if LEGACY_REPO_PREFIX in value:
            return value.replace(LEGACY_REPO_PREFIX, repo_root)
        return value
    if isinstance(value, list):
        return [_rewrite_legacy_paths(item, repo_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_legacy_paths(item, repo_root) for key, item in value.items()}
    return value


def _apply_config_items(args, items, parser, arg_types, arg_defaults, section: str = ""):
    for key, value in items.items():
        if section == "model" and key == "name":
            if hasattr(args, "model"):
                current = getattr(args, "model")
                default = arg_defaults.get("model")
                if current == default or current is None or current == "":
                    setattr(args, "model", value)
            continue

        if hasattr(args, key):
            expected_type = arg_types.get(key)
            if expected_type and value is not None:
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                except (ValueError, TypeError):
                    pass
            current = getattr(args, key)
            default = arg_defaults.get(key)
            if current == default or current is None or current == "":
                setattr(args, key, value)


def _expand_run_dir_templates(args):
    run_dir = getattr(args, "run_dir", "") or ""
    save_name = str(getattr(args, "save_name", "") or "")
    if run_dir and ("{timestamp}" in run_dir or "{save_name}" in run_dir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = run_dir.replace("{save_name}", save_name).replace("{timestamp}", ts)
    return args


def load_yaml_config(args, parser=None):
    """
    Merge YAML configuration into args.
    Sections: data, model, train, eval, key (key applied last).
    """
    if not args.config:
        return args
    
    if not os.path.exists(args.config):
        print(f"[!] Warning: Config file not found: {args.config}")
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        return args

    repo_root = os.environ.get("TRANSOLVER_ROOT", _repo_root())
    config = _rewrite_legacy_paths(config, repo_root)

    # Get types and defaults from parser to ensure correct casting
    arg_types = {}
    arg_defaults = {}
    if parser:
        for action in parser._actions:
            for opt in action.option_strings:
                if opt.startswith("--"):
                    arg_types[opt[2:]] = action.type
            if action.dest:
                arg_defaults[action.dest] = action.default

    sections = ["data", "model", "train", "eval", "key"]
    for section in sections:
        if section in config and config[section]:
            _apply_config_items(
                args, config[section], parser, arg_types, arg_defaults, section=section
            )

    # Global transfer learning block (fx channel remap when loading ckpt).
    # Kept as a nested dict on args.transfer — not flattened into CLI flags.
    if config.get("transfer"):
        args.transfer = config["transfer"]

    return _expand_run_dir_templates(args)


def build_argparser() -> argparse.ArgumentParser:
    """
    Refined Argument Parser for cache (Cache-Only Training).
    Removed legacy CSV/filtering parameters to keep the system clean.
    """
    parser = argparse.ArgumentParser("Transolver cache Trainer")

    root = _project_root()

    # 1. Paths & Setup
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")
    parser.add_argument("--run_dir", type=str, default="", help="Directory for outputs")
    parser.add_argument("--cache_path", type=str, default="", help="Path to pre-built cache.pt")
    parser.add_argument("--coord_norm_path", type=str, default="", help="Path to coord_norm.pt (auto-derived if empty)")
    parser.add_argument("--save_name", type=str, default="transolver_experiment", help="Experiment identifier")
    parser.add_argument("--load_ckpt", type=str, default="", help="Path to checkpoint to resume or eval")
    parser.add_argument(
        "--transfer_json",
        type=str,
        default="",
        help="Optional JSON string/path overriding yaml transfer: {enabled,pos_feat_dim,new_init,fx_map}",
    )

    # 2. Data Partitioning
    parser.add_argument("--train_split", type=float, default=0.8, help="Train ratio (random frame split, or geom count if split_by=geom)")
    parser.add_argument(
        "--split_by",
        type=str,
        default="random",
        choices=["random", "geom", "case"],
        help="random: 按帧随机; geom: 按几何 geom_id 整组划分; case: metadata train_cases/test_cases",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument(
        "--train_geom_ids",
        nargs="*",
        type=int,
        default=None,
        help="显式训练几何 id（与 test_geom_ids 一起使用；split_by=geom）",
    )
    parser.add_argument(
        "--test_geom_ids",
        nargs="*",
        type=int,
        default=None,
        help="显式测试几何 id（与 train_geom_ids 一起使用；split_by=geom）",
    )
    parser.add_argument("--output_cols", nargs="+", default=["Mises"], help="Names of output physical quantities")
    parser.add_argument("--padding", type=int, default=0, help="1: Enable padding mask for zero-padded points")
    parser.add_argument("--padding_value", type=float, default=0.0, help="Padding value for masked points")
    parser.add_argument(
        "--use_node_weight",
        type=int,
        default=0,
        help="1: pass Delaunay/cache node_weight into Physics-Attention slice aggregation",
    )
    parser.add_argument(
        "--persist_node_weight",
        type=int,
        default=0,
        help="1: if cache lacks node_weight, compute and write it back into cache.pt (no subsample/filter)",
    )
    parser.add_argument(
        "--node_weight_max_edge",
        type=float,
        default=0.0,
        help="Optional Delaunay max edge length filter (0=disabled)",
    )
    parser.add_argument("--pos_norm_type", type=str, default="min-max", choices=["min-max", "mean-std"])
    parser.add_argument("--fx_norm_type", type=str, default="mean-std", choices=["min-max", "mean-std"])
    parser.add_argument("--y_norm_type", type=str, default="mean-std", choices=["min-max", "mean-std"])
    parser.add_argument("--empty_fx", type=int, default=0, help="1: ignore cache fx, use fun_dim=0 (time via time_input only)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Temporal subsample: keep every N-th frame (1=all)")
    parser.add_argument("--frame_offset", type=int, default=0, help="Start frame index before striding")
    parser.add_argument("--max_frames", type=int, default=0, help="Cap frames after stride (0=no limit)")

    # 3. Model Architecture (Transolver)
    parser.add_argument("--model", type=str, default="Transolver_1D")
    parser.add_argument("--use_multi_net", type=int, default=1, help="1: Independent sub-nets per channel, 0: Single net for all channels")
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--slice_num", type=int, default=32)
    parser.add_argument("--ref", type=int, default=8)
    parser.add_argument("--unified_pos", type=int, default=1)
    parser.add_argument("--time_input", type=int, default=0, help="1: Transolver Time_Input + timestep_embedding(T), requires cache t")
    parser.add_argument("--use_spectral", type=int, default=0, help="1: Enable FNO-style spectral filtering in Physics Attention")
    parser.add_argument("--spectral_modes", type=int, default=16, help="Number of Fourier modes to keep")
    parser.add_argument("--spectral_hi_freq_boost", type=float, default=1.0,
                        help="High-frequency boost strength in SpectralLayer (0.0 disables)")
    parser.add_argument("--use_edge_conv", type=int, default=0, help="1: Enable local edge convolution before attention")
    parser.add_argument("--H", type=int, default=256, help="Image height for Transolver_2D (structured mesh)")
    parser.add_argument("--W", type=int, default=256, help="Image width for Transolver_2D (structured mesh)")

    # SATO Specific Parameters
    parser.add_argument("--use_sato", type=int, default=0, help="1: Enable SATO mode")
    parser.add_argument("--sato_index_path", type=str, default="", help="Path to precomputed SATO indices")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size for serialized attention")
    parser.add_argument("--shift", type=int, default=1, help="Number of serialization shifts")
    parser.add_argument("--orders", nargs="+", default=["z", "z-trans", "hilbert", "hilbert-trans"], help="Serialization orders")

    # 4. Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--sdf_weight_alpha", type=float, default=0.0, help="Alpha for SDF-weighted loss")
    parser.add_argument("--grad_loss_weight", type=float, default=0.0, help="Weight for pointwise gradient loss")
    parser.add_argument("--grad_k", type=int, default=8, help="kNN size for gradient estimation on point clouds")
    parser.add_argument("--grad_eps", type=float, default=1e-6, help="Stability epsilon for local least squares")
    parser.add_argument("--knn_enable", type=int, default=1, help="1: Enable kNN cache for gradient loss")
    parser.add_argument("--knn_cache_path", type=str, default="", help="Path to cached kNN indices (auto-derived if empty)")
    parser.add_argument("--knn_cache_rebuild", type=int, default=0, help="1: Force rebuild kNN cache")
    parser.add_argument("--knn_block_size", type=int, default=512, help="Block size for kNN cache building to reduce memory")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step", "plateau"],
        help="LR scheduler: none/cosine/step/plateau",
    )
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for cosine/plateau schedulers")
    parser.add_argument("--lr_step_size", type=int, default=500, help="StepLR: epochs per decay step")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR: multiplicative decay factor")
    parser.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=50,
        help="ReduceLROnPlateau: eval epochs without improvement before LR decay",
    )
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5, help="ReduceLROnPlateau: LR decay factor")
    parser.add_argument("--early_stop", type=int, default=1, help="1: stop when eval plateaus (see early_stop_patience)")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=100,
        help="Stop after this many epochs without >early_stop_min_delta_rel eval improvement",
    )
    parser.add_argument(
        "--early_stop_min_delta_rel",
        type=float,
        default=0.001,
        help="Minimum relative eval improvement to reset early-stop counter (0.001 = 0.1%%)",
    )

    # 5. Distributed / Hardware Setup
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    parser.add_argument("--gpu", type=str, default="0", help="GPU IDs for DP mode")
    parser.add_argument("--preload_data_to_gpu", action="store_true", help="Keep all data on GPU memory")

    # 6. Evaluation & Logging Control
    parser.add_argument("--viz_interval", type=int, default=100000, help="Plotting disabled by default")
    parser.add_argument("--eval_interval", type=int, default=100, help="Interval for saving best model")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Limit batches per epoch (0=all)")
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--auto_eval", type=int, default=1, help="1: run evaluate.py after training (rank 0 only)")
    parser.add_argument("--num_plot", type=int, default=3, help="Best/worst cases to plot and report in evaluate")
    parser.add_argument("--high_stress_analysis", type=int, default=1, help="1: high-stress region error analysis in evaluate")
    parser.add_argument("--per_channel_stress", type=int, default=1, help="1: per-channel stress filter in high_stress_analysis")
    parser.add_argument("--stress_channel_idx", type=int, default=0, help="Stress channel index when per_channel_stress=0")
    parser.add_argument("--stress_lower_ratio", type=float, default=2 / 3, help="Lower stress ratio for high_stress_analysis")
    parser.add_argument("--stress_upper_ratio", type=float, default=1.0, help="Upper stress ratio for high_stress_analysis")
    parser.add_argument("--error_thresholds", type=str, default="0.01,0.05,0.10,0.20", help="Comma-separated error thresholds")
    parser.add_argument(
        "--rel_err_denoms",
        type=str,
        default="",
        help="High-stress relative error denominators: 'local', 'max', or 'local,max'. "
             "Empty=auto (displacement→max only; stress→local+max)",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 step to verify setup")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")

    return parser
