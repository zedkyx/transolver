from __future__ import annotations

import os
import sys
import json
import subprocess
import time
import shutil
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.utils.data.distributed import DistributedSampler
except Exception:  # pragma: no cover
    DistributedSampler = None

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import UnitTransformer, IdentityTransformer
from scripts.transolver.core.cache_loader import (
    load_cache_data,
    resolve_train_test_indices,
    resolve_coord_norm_path,
    frame_subsample_active,
    _load_cache,
)
from scripts.transolver.core.plot_utils import (
    plot_loss_curves,
    plot_loss_curves_channels,
    plot_samples,
    save_loss_curves_npy,
)

from scripts.transolver.train.ddp_utils import init_ddp_if_needed, cleanup_ddp
from scripts.transolver.train.modeling import MultiNetWrapper
from scripts.transolver.train.loops import run_epoch, build_padding_mask
from scripts.transolver.train.checkpointing import save_state_dict, maybe_load_checkpoint, load_state_dict_into
from scripts.transolver.train.scheduling import (
    build_lr_scheduler,
    current_lr,
    is_meaningful_improvement,
)


def _get_script_root() -> str:
    # scripts/transolver/train/ -> scripts/transolver/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _run_auto_eval(args, run_root: str, ckpt_dir: str) -> None:
    if not int(getattr(args, "auto_eval", 1)):
        return
    if getattr(args, "smoke_test", False) or getattr(args, "eval_only", False):
        return

    ckpt_path = os.path.join(ckpt_dir, f"{args.save_name}_best_eval.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[!] auto_eval skipped: checkpoint not found: {ckpt_path}")
        return

    repo_root = _repo_root()
    eval_script = os.path.join(repo_root, "scripts", "transolver", "eval", "evaluate.py")
    if not os.path.isfile(eval_script):
        print(f"[!] auto_eval skipped: eval script not found: {eval_script}")
        return

    config_path = os.path.join(run_root, "config.yaml")
    if not os.path.isfile(config_path):
        config_path = getattr(args, "config", "") or ""

    cmd = [
        sys.executable,
        eval_script,
        "--run_dir",
        run_root,
        "--load_ckpt",
        ckpt_path,
        "--save_name",
        str(args.save_name),
        "--num_plot",
        str(int(getattr(args, "num_plot", 3))),
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if getattr(args, "cache_path", ""):
        cmd.extend(["--cache_path", str(args.cache_path)])
    out_cols = getattr(args, "output_cols", None)
    if out_cols:
        if isinstance(out_cols, (list, tuple)):
            cmd.extend(["--output_cols", *[str(c) for c in out_cols]])
        else:
            cmd.extend(["--output_cols", str(out_cols)])
    if getattr(args, "coord_norm_path", ""):
        cmd.extend(["--coord_norm_path", str(args.coord_norm_path)])
    if int(getattr(args, "high_stress_analysis", 1)):
        cmd.extend(["--high_stress_analysis", "1"])
    if int(getattr(args, "per_channel_stress", 1)):
        cmd.extend(["--per_channel_stress", "1"])
    cmd.extend(
        [
            "--stress_channel_idx",
            str(int(getattr(args, "stress_channel_idx", 0))),
            "--stress_lower_ratio",
            str(float(getattr(args, "stress_lower_ratio", 2 / 3))),
            "--stress_upper_ratio",
            str(float(getattr(args, "stress_upper_ratio", 1.0))),
            "--error_thresholds",
            str(getattr(args, "error_thresholds", "0.01,0.05,0.10,0.20")),
        ]
    )

    log_path = os.path.join(run_root, "logs", "eval.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"[*] auto_eval: starting evaluate.py (log -> {log_path})")
    print(f"[*] auto_eval: eval output -> {os.path.join(run_root, 'eval')}/")
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    with open(log_path, "w", encoding="utf-8") as log_f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
    ret = proc.wait()
    if ret != 0:
        print(f"[!] auto_eval failed (exit {ret}), see {log_path}")
    else:
        print(f"[✓] auto_eval finished, log: {log_path}")


def _teardown_viz_cuda(device: torch.device) -> None:
    """Close matplotlib figures and sync CUDA to reduce rare exit-time segfaults."""
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def run(args) -> None:
    """
    Unified cache trainer for both DP and DDP.
    DP/DDP differences are limited to:
    - sampler (DistributedSampler)
    - model wrapping (DDP vs DataParallel/None)
    - rank0-only logging/saving/viz
    """
    # DP: CUDA_VISIBLE_DEVICES must be set BEFORE the first torch.cuda call.
    # Otherwise a shell/scheduler may have restricted visibility (e.g. only GPU 0),
    # and later assignment in this function is ignored — DataParallel then only uses cuda:0.
    if not getattr(args, "ddp", False):
        gpu_str = str(getattr(args, "gpu", "") or "").strip()
        if gpu_str:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    ctx = init_ddp_if_needed(args)
    device = ctx.device
    is_main = ctx.is_main_process

    # --- Output Path Alignment ---
    run_dir = getattr(args, "run_dir", "")
    if run_dir:
        # If run_dir is explicitly provided (recommended), use it directly
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        plot_dir = os.path.join(run_dir, "plot")
        run_root = run_dir
    else:
        # Legacy/Fallback: Use script-relative paths (might be symlinks)
        script_root = _get_script_root()
        ckpt_dir = os.path.join(script_root, "checkpoints")
        plot_dir = os.path.join(script_root, "plot")
        # Try to resolve run_root from symlink
        if os.path.islink(ckpt_dir):
            run_root = os.path.dirname(os.path.realpath(ckpt_dir))
        else:
            run_root = script_root

    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Backup config.yaml to the run root directory
        config_path = getattr(args, "config", "")
        if config_path and os.path.exists(config_path):
            target_config = os.path.join(run_root, "config.yaml")
            shutil.copy(config_path, target_config)
            print(f"[*] Configuration backed up to: {target_config}")
            print(f"[*] Checkpoints will be saved to: {ckpt_dir}")
            print(f"[*] Plots will be saved to: {plot_dir}")

    # DP: use logical device indices 0..N-1 (after CUDA_VISIBLE_DEVICES remapping)
    gpu_ids = []
    num_gpus = 0
    if not ctx.enabled:
        if device.type == "cuda":
            num_gpus = int(torch.cuda.device_count())
            gpu_ids = list(range(num_gpus))
            if is_main:
                vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                print(f"Using {num_gpus} visible GPU(s), DataParallel device_ids={gpu_ids}")
                if vis:
                    print(f"    CUDA_VISIBLE_DEVICES={vis}")

    # --- Data Loading ---
    pos, fx, y, input_dim, output_dim, _, sato_indices, case_t, node_weight = load_cache_data(args)

    time_input = bool(int(getattr(args, "time_input", 0)))
    use_node_weight = bool(int(getattr(args, "use_node_weight", 0)))
    if use_node_weight and node_weight is None:
        raise RuntimeError("use_node_weight=1 but node_weight is None after load_cache_data")
    if time_input:
        if case_t is None:
            raise ValueError("time_input=1 需要 cache 中的 't' 字段")
        if getattr(args, "model", "Transolver_1D") == "SATO":
            raise ValueError("SATO 不支持 Time_Input，请改用 model: Transolver_1D")
        if is_main:
            print(f"[*] time_input=1: 使用 timestep_embedding(T)，T 来自 cache.t，范围 [{case_t.min():.4f}, {case_t.max():.4f}]")

    padding_enabled = bool(int(getattr(args, "padding", 0)))
    padding_value = float(getattr(args, "padding_value", 0.0))

    # --- Split ---
    cache = _load_cache(getattr(args, "cache_path", ""))
    B = pos.shape[0]
    case_names = getattr(args, "filtered_case_names", None)
    geom_id = getattr(args, "filtered_geom_id", None)
    train_idx, test_idx, split_mode = resolve_train_test_indices(
        cache,
        train_split=float(getattr(args, "train_split", 0.8)),
        seed=int(getattr(args, "seed", 42)),
        num_samples=B,
        case_names=case_names,
        geom_id=geom_id,
        split_by=str(getattr(args, "split_by", "random")),
        train_geom_ids=getattr(args, "train_geom_ids", None),
        test_geom_ids=getattr(args, "test_geom_ids", None),
    )
    if frame_subsample_active(args) and is_main:
        split_mode += ", frame_subsample(stride={}, offset={}, max_frames={})".format(
            int(getattr(args, "frame_stride", 1)),
            int(getattr(args, "frame_offset", 0)),
            int(getattr(args, "max_frames", 0)),
        )
    if is_main:
        print(f"[*] Dataset split: {split_mode}")
        if str(getattr(args, "split_by", "random")).lower() == "geom":
            if meta := cache.get("metadata", {}):
                if meta.get("train_cases") is not None:
                    print(f"    train cases: {meta.get('train_cases')}")
                    print(f"    test cases : {meta.get('test_cases')}")
                if meta.get("train_geom_ids") is not None:
                    print(f"    train_geom_ids: {meta.get('train_geom_ids')}")
                    print(f"    test_geom_ids : {meta.get('test_geom_ids')}")
    pos_train, fx_train, y_train = pos[train_idx], fx[train_idx], y[train_idx]
    pos_test, fx_test, y_test = pos[test_idx], fx[test_idx], y[test_idx]
    nw_train = nw_test = None
    if use_node_weight and node_weight is not None:
        nw_train = node_weight[train_idx]
        nw_test = node_weight[test_idx]
        if is_main:
            print(
                f"[*] use_node_weight=1: train q "
                f"min={nw_train[nw_train > 0].min().item():.3e} "
                f"max={nw_train.max().item():.3e}"
            )
    if padding_enabled:
        mask_train = build_padding_mask(pos_train, fx_train, y_train, padding_value)
        mask_test = build_padding_mask(pos_test, fx_test, y_test, padding_value)
        if is_main:
            total_train_points = mask_train.numel()
            valid_train_points = mask_train.sum().item()
            padding_train_points = total_train_points - valid_train_points
            total_test_points = mask_test.numel()
            valid_test_points = mask_test.sum().item()
            padding_test_points = total_test_points - valid_test_points
            print(f"[*] Padding enabled (padding_value={padding_value}), will build mask for valid points")
            print(f"    Train set: {valid_train_points} valid points, {padding_train_points} padding points ({100*padding_train_points/total_train_points:.2f}%)")
            print(f"    Test set: {valid_test_points} valid points, {padding_test_points} padding points ({100*padding_test_points/total_test_points:.2f}%)")
    else:
        mask_train = None
        mask_test = None

    # --- Normalization (coords, fx, y) ---
    def get_normalizer(data, norm_type, *, node_weight=None, mask=None):
        if norm_type == "min-max":
            if mask is not None:
                return IdentityTransformer(data[mask])
            return IdentityTransformer(data)
        return UnitTransformer(data, node_weight=node_weight, mask=mask)

    coord_stats_path = resolve_coord_norm_path(args)
    subsample_active = frame_subsample_active(args)
    if is_main:
        if subsample_active:
            print(
                f"[*] frame subsample: normalizers from subsampled train set -> {coord_stats_path}"
            )
        elif not getattr(args, "coord_norm_path", ""):
            print(f"[*] coord_norm_path not provided, using default: {coord_stats_path}")

    load_pos_from_file = (
        coord_stats_path
        and os.path.exists(coord_stats_path)
        and not subsample_active
    )
    if load_pos_from_file:
        try:
            saved = torch.load(coord_stats_path, weights_only=True)
        except TypeError:
            saved = torch.load(coord_stats_path)
        pos_min = saved["min"].view(1, 1, -1)
        pos_max = saved["max"].view(1, 1, -1)
        pos_normalizer = IdentityTransformer(
            min_val=pos_min.squeeze(0).squeeze(0),
            max_val=pos_max.squeeze(0).squeeze(0),
        )
    else:
        # 计算坐标归一化参数（如果padding_enabled，只使用有效点）
        if padding_enabled and mask_train is not None:
            # mask_train: [B, N], pos_train: [B, N, 2]
            # 使用mask过滤出所有有效点
            pos_train_valid = pos_train[mask_train]  # [N_valid, 2]
            if is_main:
                total_points = pos_train.shape[0] * pos_train.shape[1]
                valid_points = pos_train_valid.shape[0]
                padding_points = total_points - valid_points
                print(f"[*] Computing pos normalizer ({args.pos_norm_type}) from valid points only: {valid_points} valid points, {padding_points} padding points excluded ({100*padding_points/total_points:.2f}%)")
            pos_normalizer = get_normalizer(pos_train_valid, args.pos_norm_type)
        else:
            pos_normalizer = get_normalizer(pos_train, args.pos_norm_type)
        
        if coord_stats_path and is_main:
            os.makedirs(os.path.dirname(coord_stats_path), exist_ok=True)
            # 保存坐标归一化参数
            norm_stats = {
                "min": pos_normalizer.min.detach().cpu(),
                "max": pos_normalizer.max.detach().cpu()
            }
            torch.save(norm_stats, coord_stats_path)
            print(f"[*] Saved coord normalization stats to {coord_stats_path}")

    # 计算fx/y归一化参数（如果padding_enabled，只使用有效点）
    t_normalizer = None
    t_train_n = t_test_n = None
    normalization_weight = nw_train if use_node_weight else None
    weighted_fx_norm = normalization_weight is not None and args.fx_norm_type != "min-max"
    weighted_y_norm = normalization_weight is not None and args.y_norm_type != "min-max"
    if input_dim > 0:
        if normalization_weight is not None:
            fx_normalizer = get_normalizer(
                fx_train,
                args.fx_norm_type,
                node_weight=normalization_weight,
                mask=mask_train,
            )
            y_normalizer = get_normalizer(
                y_train,
                args.y_norm_type,
                node_weight=normalization_weight,
                mask=mask_train,
            )
            if is_main:
                print(
                    "[*] Computing quadrature-aware fx/y normalization: "
                    "node weights normalized per sample, then samples averaged equally"
                )
        elif padding_enabled and mask_train is not None:
            fx_train_valid = fx_train[mask_train]
            y_train_valid = y_train[mask_train]
            if is_main:
                total_points = fx_train.shape[0] * fx_train.shape[1]
                valid_points = fx_train_valid.shape[0]
                padding_points = total_points - valid_points
                print(f"[*] Computing fx/y normalization ({args.fx_norm_type}/{args.y_norm_type}) from valid points only: {valid_points} valid points, {padding_points} padding points excluded ({100*padding_points/total_points:.2f}%)")
                print(f"    fx_train_valid shape: {fx_train_valid.shape}, y_train_valid shape: {y_train_valid.shape}")
            fx_train_valid_3d = fx_train_valid.unsqueeze(0)
            y_train_valid_3d = y_train_valid.unsqueeze(0)
            fx_normalizer = get_normalizer(fx_train_valid_3d, args.fx_norm_type)
            y_normalizer = get_normalizer(y_train_valid_3d, args.y_norm_type)
            if is_main:
                if hasattr(fx_normalizer, "mean"):
                    print(f"    fx_normalizer (mean-std): mean range [{fx_normalizer.mean.min():.6f}, {fx_normalizer.mean.max():.6f}], std range [{fx_normalizer.std.min():.6f}, {fx_normalizer.std.max():.6f}]")
                else:
                    print(f"    fx_normalizer (min-max): min range [{fx_normalizer.min.min():.6f}, {fx_normalizer.min.max():.6f}], max range [{fx_normalizer.max.min():.6f}, {fx_normalizer.max.max():.6f}]")
                if hasattr(y_normalizer, "mean"):
                    print(f"    y_normalizer (mean-std): mean range [{y_normalizer.mean.min():.6f}, {y_normalizer.mean.max():.6f}], std range [{y_normalizer.std.min():.6f}, {y_normalizer.std.max():.6f}]")
                else:
                    print(f"    y_normalizer (min-max): min range [{y_normalizer.min.min():.6f}, {y_normalizer.min.max():.6f}], max range [{y_normalizer.max.min():.6f}, {y_normalizer.max.max():.6f}]")
        else:
            fx_normalizer = get_normalizer(fx_train, args.fx_norm_type)
            y_normalizer = get_normalizer(y_train, args.y_norm_type)
    else:
        fx_normalizer = None
        if normalization_weight is not None:
            y_normalizer = get_normalizer(
                y_train,
                args.y_norm_type,
                node_weight=normalization_weight,
                mask=mask_train,
            )
        elif padding_enabled and mask_train is not None:
            y_train_valid = y_train[mask_train]
            y_train_valid_3d = y_train_valid.unsqueeze(0)
            y_normalizer = get_normalizer(y_train_valid_3d, args.y_norm_type)
        else:
            y_normalizer = get_normalizer(y_train, args.y_norm_type)
        if is_main:
            print(f"[*] input_dim=0 (empty_fx): 跳过 fx 归一化")

    if time_input and case_t is not None:
        t_train_3d = case_t[train_idx].view(-1, 1, 1)
        t_normalizer = get_normalizer(t_train_3d, args.fx_norm_type)
        t_train_n = t_normalizer.encode(case_t[train_idx].view(-1, 1, 1)).squeeze(-1).squeeze(-1)
        t_test_n = t_normalizer.encode(case_t[test_idx].view(-1, 1, 1)).squeeze(-1).squeeze(-1)
        if is_main:
            print(f"[*] t_normalizer ({args.fx_norm_type}): train range [{t_train_n.min():.4f}, {t_train_n.max():.4f}]")

    # 保存fx/y/t归一化参数到coord_norm文件（如果coord_stats_path存在）
    if coord_stats_path and is_main:
        try:
            # 尝试加载现有文件（如果存在）
            saved = torch.load(coord_stats_path, map_location="cpu", weights_only=True)
        except (TypeError, FileNotFoundError):
            try:
                saved = torch.load(coord_stats_path, map_location="cpu")
            except FileNotFoundError:
                saved = {}
        
        # 更新或添加fx/y/t归一化参数
        if fx_normalizer is not None:
            if hasattr(fx_normalizer, "mean"):
                saved["fx_mean"] = fx_normalizer.mean.detach().cpu()
                saved["fx_std"] = fx_normalizer.std.detach().cpu()
            else:
                saved["fx_min"] = fx_normalizer.min.detach().cpu()
                saved["fx_max"] = fx_normalizer.max.detach().cpu()
        else:
            saved.pop("fx_mean", None)
            saved.pop("fx_std", None)
            saved.pop("fx_min", None)
            saved.pop("fx_max", None)

        if t_normalizer is not None:
            if hasattr(t_normalizer, "mean"):
                saved["t_mean"] = t_normalizer.mean.detach().cpu()
                saved["t_std"] = t_normalizer.std.detach().cpu()
            else:
                saved["t_min"] = t_normalizer.min.detach().cpu()
                saved["t_max"] = t_normalizer.max.detach().cpu()

        if hasattr(y_normalizer, "mean"):
            saved["y_mean"] = y_normalizer.mean.detach().cpu()
            saved["y_std"] = y_normalizer.std.detach().cpu()
        else:
            saved["y_min"] = y_normalizer.min.detach().cpu()
            saved["y_max"] = y_normalizer.max.detach().cpu()
        saved["normalization_node_weighted"] = {
            "fx": bool(weighted_fx_norm and fx_normalizer is not None),
            "y": bool(weighted_y_norm),
            "semantics": "per-sample unit mass, equal sample average",
        }
        
        torch.save(saved, coord_stats_path)
        print(f"[*] Saved fx/y normalization stats to {coord_stats_path}")
        
        # Print fx normalization stats based on actual type
        if "fx_mean" in saved:
            print(f"    fx (mean-std): mean shape={saved['fx_mean'].shape}, std shape={saved['fx_std'].shape}")
            print(f"    fx_mean range: [{saved['fx_mean'].min():.6f}, {saved['fx_mean'].max():.6f}], fx_std range: [{saved['fx_std'].min():.6f}, {saved['fx_std'].max():.6f}]")
        elif "fx_min" in saved:
            print(f"    fx (min-max): min shape={saved['fx_min'].shape}, max shape={saved['fx_max'].shape}")
            print(f"    fx_min range: [{saved['fx_min'].min():.6f}, {saved['fx_min'].max():.6f}], fx_max range: [{saved['fx_max'].min():.6f}, {saved['fx_max'].max():.6f}]")
        
        # Print y normalization stats based on actual type
        if "y_mean" in saved:
            print(f"    y (mean-std): mean shape={saved['y_mean'].shape}, std shape={saved['y_std'].shape}")
            print(f"    y_mean range: [{saved['y_mean'].min():.6f}, {saved['y_mean'].max():.6f}], y_std range: [{saved['y_std'].min():.6f}, {saved['y_std'].max():.6f}]")
        elif "y_min" in saved:
            print(f"    y (min-max): min shape={saved['y_min'].shape}, max shape={saved['y_max'].shape}")
            print(f"    y_min range: [{saved['y_min'].min():.6f}, {saved['y_min'].max():.6f}], y_max range: [{saved['y_max'].min():.6f}, {saved['y_max'].max():.6f}]")

    pos_train_n = pos_normalizer.encode(pos_train)
    fx_train_n = fx_normalizer.encode(fx_train) if fx_normalizer is not None else fx_train
    y_train_n = y_normalizer.encode(y_train)
    pos_test_n = pos_normalizer.encode(pos_test)
    fx_test_n = fx_normalizer.encode(fx_test) if fx_normalizer is not None else fx_test
    y_test_n = y_normalizer.encode(y_test)

    if device.type == "cuda":
        pos_normalizer.cuda()
        if fx_normalizer is not None:
            fx_normalizer.cuda()
        if t_normalizer is not None:
            t_normalizer.cuda()
        y_normalizer.cuda()

    # Optional preload to GPU (keeps existing semantics)
    preload = bool(getattr(args, "preload_data_to_gpu", False))
    if preload and device.type == "cuda":
        pos_train_n = pos_train_n.to(device)
        fx_train_n = fx_train_n.to(device)
        y_train_n = y_train_n.to(device)
        pos_test_n = pos_test_n.to(device)
        fx_test_n = fx_test_n.to(device)
        y_test_n = y_test_n.to(device)
        if padding_enabled:
            mask_train = mask_train.to(device)
            mask_test = mask_test.to(device)
        if use_node_weight and nw_train is not None:
            nw_train = nw_train.to(device)
            nw_test = nw_test.to(device)
        dl_num_workers = 0
        dl_pin_memory = False
        if is_main:
            print(f"Preloaded all data tensors to GPU: {device}")
    else:
        dl_num_workers = 0
        dl_pin_memory = False

    # --- kNN cache for EdgeConv and gradient loss (separate conditions) ---
    knn_idx = None
    knn_valid = None
    knn_idx_train = None
    knn_valid_train = None
    knn_idx_test = None
    knn_valid_test = None
    
    grad_w = float(getattr(args, "grad_loss_weight", 0.0))
    use_edge_conv = bool(int(getattr(args, "use_edge_conv", 0)))
    knn_enable = bool(int(getattr(args, "knn_enable", 1)))
    
    # Check if kNN is needed for EdgeConv
    need_knn_for_edgeconv = use_edge_conv and knn_enable
    # Check if kNN is needed for gradient loss
    need_knn_for_grad = (grad_w > 0.0) and knn_enable
    # Build kNN cache if needed for either purpose
    need_knn = need_knn_for_edgeconv or need_knn_for_grad
    
    if need_knn:
        from scripts.transolver.train.metrics import build_knn_cache

        cache_path = getattr(args, "cache_path", "")
        k = int(getattr(args, "grad_k", 8))
        knn_cache_path = getattr(args, "knn_cache_path", "")
        if not knn_cache_path:
            base = cache_path.replace(".pt", "") if cache_path else "knn_cache"
            knn_cache_path = f"{base}_knn_k{k}.pt"
            args.knn_cache_path = knn_cache_path

        rebuild = bool(int(getattr(args, "knn_cache_rebuild", 0)))
        if (not rebuild) and knn_cache_path and os.path.exists(knn_cache_path):
            try:
                saved = torch.load(knn_cache_path, weights_only=True)
            except TypeError:
                saved = torch.load(knn_cache_path)
            if isinstance(saved, dict) and saved.get("k") == k:
                knn_idx = saved.get("knn_idx")
                knn_valid = saved.get("knn_valid")
        
        if (knn_idx is None) or (knn_valid is None):
            block_size = int(getattr(args, "knn_block_size", 512))
            pos_norm_cpu = IdentityTransformer(
                min_val=pos_normalizer.min.detach().cpu(),
                max_val=pos_normalizer.max.detach().cpu(),
            )
            pos_all_n = pos_norm_cpu.encode(pos.cpu())
            if padding_enabled:
                mask_all = torch.zeros(pos.shape[0], pos.shape[1], dtype=mask_train.dtype, device="cpu")
                mask_all[train_idx] = mask_train.cpu()
                mask_all[test_idx] = mask_test.cpu()
            else:
                mask_all = None
            with torch.no_grad():
                knn_idx, knn_valid = build_knn_cache(pos_all_n, k=k, mask=mask_all, block_size=block_size)
            if knn_cache_path:
                os.makedirs(os.path.dirname(knn_cache_path) or ".", exist_ok=True)
                torch.save({"k": k, "knn_idx": knn_idx, "knn_valid": knn_valid}, knn_cache_path)

        if preload and device.type == "cuda":
            if knn_idx is not None:
                knn_idx = knn_idx.to(device)
            if knn_valid is not None:
                knn_valid = knn_valid.to(device)

        if knn_idx is not None and knn_valid is not None:
            knn_idx_train = knn_idx[train_idx]
            knn_valid_train = knn_valid[train_idx]
            knn_idx_test = knn_idx[test_idx]
            knn_valid_test = knn_valid[test_idx]

    # --- Build datasets with or without kNN data ---
    def _build_ds(p, f, y, m, nw, k_idx, k_val, ord, inv, t_n=None):
        tensors = [p, f, y]
        if m is not None:
            tensors.append(m)
        if nw is not None:
            tensors.append(nw)
        if k_idx is not None:
            tensors.append(k_idx)
        if k_val is not None:
            tensors.append(k_val)
        if ord is not None:
            tensors.append(ord)
        if inv is not None:
            tensors.append(inv)
        if t_n is not None:
            tensors.append(t_n)
        return TensorDataset(*tensors)

    if sato_indices is not None:
        order_train, inverse_train = sato_indices["order"][train_idx], sato_indices["inverse"][train_idx]
        order_test, inverse_test = sato_indices["order"][test_idx], sato_indices["inverse"][test_idx]
    else:
        order_train = inverse_train = order_test = inverse_test = None

    train_ds = _build_ds(
        pos_train_n, fx_train_n, y_train_n, mask_train, nw_train,
        knn_idx_train, knn_valid_train, order_train, inverse_train,
        t_train_n if time_input else None,
    )
    test_ds = _build_ds(
        pos_test_n, fx_test_n, y_test_n, mask_test, nw_test,
        knn_idx_test, knn_valid_test, order_test, inverse_test,
        t_test_n if time_input else None,
    )

    if ctx.enabled:
        if DistributedSampler is None:
            raise RuntimeError("DistributedSampler unavailable but args.ddp is True")
        train_sampler = DistributedSampler(train_ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True)
        test_sampler = DistributedSampler(test_ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=dl_num_workers, pin_memory=dl_pin_memory)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler, num_workers=dl_num_workers, pin_memory=dl_pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=dl_num_workers, pin_memory=dl_pin_memory)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=dl_num_workers, pin_memory=dl_pin_memory)

    # --- Model ---
    model = MultiNetWrapper(args, input_dim=input_dim, output_dim=output_dim).to(device)
    if is_main:
        n_params_total = sum(p.numel() for p in model.parameters())
        n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[*] Model parameters: total={n_params_total:,}, "
            f"trainable={n_params_trainable:,} ({n_params_total / 1e6:.3f}M)"
        )

    if ctx.enabled:
        # MultiNet / SATO 等路径下，单次 forward 可能未用到全部子网参数；DDP 默认会报错，需开启 unused 检测
        model = DDP(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=True,
        )
        if is_main:
            print(f"Wrapped model with DDP for {ctx.world_size} GPUs (find_unused_parameters=True)")
    else:
        if device.type == "cuda" and num_gpus > 1:
            if is_main:
                print(f"Wrapping model with DataParallel for {num_gpus} GPUs")
            model = nn.DataParallel(model, device_ids=gpu_ids)
            device = torch.device(f"cuda:{gpu_ids[0]}")

    if getattr(args, "load_ckpt", ""):
        transfer = getattr(args, "transfer", None)
        transfer_json = str(getattr(args, "transfer_json", "") or "").strip()
        if transfer_json:
            if os.path.isfile(transfer_json):
                with open(transfer_json, "r", encoding="utf-8") as f:
                    transfer = json.load(f)
            else:
                transfer = json.loads(transfer_json)
            args.transfer = transfer
        maybe_load_checkpoint(
            model,
            args.load_ckpt,
            device=device,
            is_main_process=is_main,
            transfer=transfer,
            args=args,
        )

    # --- Eval-only ---
    if getattr(args, "eval_only", False):
        if is_main:
            eval_dir = os.path.join(plot_dir, args.save_name, "eval_only")
            plot_samples(model, test_loader, y_normalizer, device, eval_dir, prefix="eval")
        cleanup_ddp(ctx)
        _teardown_viz_cuda(device)
        return

    # --- Smoke test ---
    if getattr(args, "smoke_test", False):
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                idx = 0
                pos_b = batch[idx].to(device, non_blocking=True); idx += 1
                fx_b = batch[idx].to(device, non_blocking=True); idx += 1
                y_b = batch[idx].to(device, non_blocking=True); idx += 1
                if padding_enabled and mask_train is not None:
                    idx += 1
                if use_node_weight:
                    idx += 1
                if need_knn:
                    idx += 2
                if sato_indices is not None:
                    idx += 2
                t_b = batch[idx].to(device, non_blocking=True) if time_input else None
                model_fx = fx_b if fx_b.shape[-1] > 0 else None
                nw_b = None
                # rebuild index for node_weight if present
                if use_node_weight:
                    # batch layout: pos,fx,y,[mask],[nw],...
                    nw_i = 3 + (1 if padding_enabled else 0)
                    nw_b = batch[nw_i].to(device, non_blocking=True)
                out = y_normalizer.decode(
                    model(pos_b, fx=model_fx, T=t_b, quadrature_weights=nw_b)
                )
                y_orig = y_normalizer.decode(y_b)
                if is_main:
                    from scripts.transolver.train.metrics import relative_l2
                    n_ch = out.shape[-1]
                    ch_rels = [relative_l2(out[..., i:i+1], y_orig[..., i:i+1]) for i in range(n_ch)]
                    avg_rel = torch.stack(ch_rels).mean()
                    print(f"Smoke Test RelL2(avg channels): {avg_rel.item():.6f}")
                break
        cleanup_ddp(ctx)
        _teardown_viz_cuda(device)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler, lr_scheduler_mode = build_lr_scheduler(optimizer, args)
    early_stop_enabled = int(getattr(args, "early_stop", 1))
    early_stop_patience = int(getattr(args, "early_stop_patience", 100))
    early_stop_min_delta_rel = float(getattr(args, "early_stop_min_delta_rel", 0.001))
    epochs_without_improvement = 0
    stopped_early = False

    train_losses, eval_losses, epochs_list = [], [], []
    train_losses_ch, eval_losses_ch = [], []
    best_eval_loss = float("inf")
    best_epoch = 0

    def _relative_l2_masked(pred, target, mask):
        from scripts.transolver.train.metrics import relative_l2
        if mask is None:
            return relative_l2(pred, target)
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask_f = mask.to(pred.dtype).unsqueeze(-1)
        diff = (pred - target) * mask_f
        denom = target * mask_f
        num = torch.sum(diff ** 2, dim=(1, 2))
        den = torch.sum(denom ** 2, dim=(1, 2))
        rel = torch.sqrt(num + 1e-12) / (torch.sqrt(den + 1e-12))
        return rel.mean()

    def loss_fn(out_dec, y_dec, fx_b_enc, fx_norm, mask_b=None, pos_b=None, knn_idx=None, knn_valid=None):
        """
        Directly call TestLoss for each channel and return the mean.
        This follows the user request to keep it simple and explicit.
        """
        from scripts.transolver.train.metrics import mse_mae

        n_ch = out_dec.shape[-1]
        per_ch_losses = []
        mse_vals = []
        mae_vals = []
        for i in range(n_ch):
            # relative_l2 calls TestLoss.rel(p=2)
            l = _relative_l2_masked(out_dec[..., i : i + 1], y_dec[..., i : i + 1], mask_b)
            per_ch_losses.append(l)
            mse_i, mae_i = mse_mae(
                out_dec[..., i : i + 1], y_dec[..., i : i + 1], mask_b
            )
            mse_vals.append(mse_i)
            mae_vals.append(mae_i)
        
        rel_ch = torch.stack(per_ch_losses)
        rel_mean = rel_ch.mean()
        per_ch_t = rel_ch
        mean_loss = rel_mean
        grad_mean = None
        grad_ch = None

        grad_w = float(getattr(args, "grad_loss_weight", 0.0))
        if grad_w > 0.0 and pos_b is not None:
            from scripts.transolver.train.metrics import gradient_loss

            grad_k = int(getattr(args, "grad_k", 8))
            grad_eps = float(getattr(args, "grad_eps", 1e-6))
            grad_mean, grad_ch = gradient_loss(
                pos_b,
                out_dec,
                y_dec,
                k=grad_k,
                mask=mask_b,
                eps=grad_eps,
                knn_idx=knn_idx,
                knn_valid=knn_valid,
            )
            per_ch_t = rel_ch + grad_w * grad_ch
            mean_loss = per_ch_t.mean()
        
        metrics = {
            "rel": rel_mean,
            "grad": grad_mean,
            "rel_ch": rel_ch,
            "grad_ch": grad_ch,
            "mse": torch.stack(mse_vals).mean(),
            "mae": torch.stack(mae_vals).mean(),
        }
        return mean_loss, per_ch_t, metrics

    if is_main:
        print(f"Starting cache training for {args.epochs} epochs... (ddp={ctx.enabled})")
        print(
            f"[*] LR scheduler: {lr_scheduler_mode}, init_lr={args.lr:.6g}, lr_min={getattr(args, 'lr_min', 1e-6):.6g}"
        )
        if early_stop_enabled:
            print(
                f"[*] Early stop: patience={early_stop_patience} epochs, "
                f"min_delta_rel={early_stop_min_delta_rel:.4f} ({early_stop_min_delta_rel * 100:.2f}%)"
            )
        else:
            print("[*] Early stop: disabled")

    def _save_plot_loss_curves():
        save_dir = os.path.join(plot_dir, args.save_name)
        plot_loss_curves(
            train_losses,
            eval_losses,
            epochs_list,
            save_dir,
            best_eval_info=(best_epoch, best_eval_loss),
        )
        if train_losses_ch and eval_losses_ch:
            plot_loss_curves_channels(
                train_losses_ch,
                eval_losses_ch,
                epochs_list,
                save_dir,
                channel_names=args.output_cols,
            )
        save_loss_curves_npy(
            train_losses,
            eval_losses,
            epochs_list,
            save_dir,
            train_losses_ch=train_losses_ch if train_losses_ch else None,
            eval_losses_ch=eval_losses_ch if eval_losses_ch else None,
            channel_names=getattr(args, "output_cols", None),
        )

    for epoch in range(args.epochs):
        t0 = time.time()
        if ctx.enabled:
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        train_loss, train_loss_ch, train_rel, train_grad, train_rel_ch, train_grad_ch, train_mse, train_mae = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            y_normalizer=y_normalizer,
            fx_normalizer=fx_normalizer,
            loss_fn=loss_fn,
            optimizer=optimizer,
            max_grad_norm=getattr(args, "max_grad_norm", None),
            max_batches=getattr(args, "max_train_batches", 0),
            padding_enabled=padding_enabled,
            padding_value=padding_value,
            use_knn=need_knn,
            use_sato=(sato_indices is not None),
            use_time_input=time_input,
            use_node_weight=use_node_weight,
        )

        eval_interval = int(getattr(args, "eval_interval", 100))
        eval_rel = float("nan")
        eval_rel_ch = None
        eval_grad = float("nan")
        eval_grad_ch = None
        eval_mse = float("nan")
        eval_mae = float("nan")
        did_eval = (epoch + 1) % eval_interval == 0
        if did_eval:
            _, _, eval_rel, eval_grad, eval_rel_ch, eval_grad_ch, eval_mse, eval_mae = run_epoch(
                model=model,
                loader=test_loader,
                device=device,
                y_normalizer=y_normalizer,
                fx_normalizer=fx_normalizer,
                loss_fn=loss_fn,
                optimizer=None,
                max_grad_norm=None,
                max_batches=getattr(args, "max_eval_batches", 0),
                padding_enabled=padding_enabled,
                padding_value=padding_value,
                use_knn=need_knn,
                use_sato=(sato_indices is not None),
                use_time_input=time_input,
                use_node_weight=use_node_weight,
            )

            if is_meaningful_improvement(best_eval_loss, eval_rel, early_stop_min_delta_rel):
                best_eval_loss = eval_rel
                best_epoch = epoch
                epochs_without_improvement = 0
                save_state_dict(model, os.path.join(ckpt_dir, f"{args.save_name}_best_eval.pt"), is_main_process=is_main)
                if is_main:
                    print(f"  [SAVE] Best Eval: {best_eval_loss:.6f} @ epoch {epoch + 1}")
            else:
                epochs_without_improvement += eval_interval
                if is_main and best_eval_loss < float("inf"):
                    if (
                        epochs_without_improvement >= early_stop_patience
                        or epochs_without_improvement % max(1, min(early_stop_patience // 4, 50)) == 0
                    ):
                        print(
                            f"  [EARLY-STOP] No >{early_stop_min_delta_rel * 100:.2f}% eval improvement "
                            f"({epochs_without_improvement}/{early_stop_patience} epochs)"
                        )

            if lr_scheduler is not None and lr_scheduler_mode == "plateau":
                lr_scheduler.step(eval_rel)

        if lr_scheduler is not None and lr_scheduler_mode == "epoch":
            lr_scheduler.step()

        dt = time.time() - t0
        lr_now = current_lr(optimizer)
        if is_main:
            if did_eval:
                print(
                    f"Epoch {epoch}: "
                    f"TrainRel={train_rel:.6f}, TrainMSE={train_mse:.6e}, TrainMAE={train_mae:.6e}, "
                    f"TrainGrad={train_grad:.6f}, "
                    f"EvalRel={eval_rel:.6f}, EvalMSE={eval_mse:.6e}, EvalMAE={eval_mae:.6e}, "
                    f"EvalGrad={eval_grad:.6f}, "
                    f"LR={lr_now:.6g}, Time={dt:.2f}s"
                )
            else:
                print(
                    f"Epoch {epoch}: "
                    f"TrainRel={train_rel:.6f}, TrainMSE={train_mse:.6e}, TrainMAE={train_mae:.6e}, "
                    f"TrainGrad={train_grad:.6f}, "
                    f"LR={lr_now:.6g}, Time={dt:.2f}s"
                )

            train_losses.append(train_rel)
            eval_losses.append(eval_rel)
            epochs_list.append(epoch)
            if train_rel_ch is not None:
                train_losses_ch.append(train_rel_ch)
                if eval_rel_ch is None:
                    eval_losses_ch.append([float("nan")] * len(train_rel_ch))
                else:
                    eval_losses_ch.append(eval_rel_ch)

        if (epoch + 1) % int(getattr(args, "viz_interval", 100)) == 0:
            if is_main:
                _save_plot_loss_curves()
                # Note: plot_samples remains disabled to maintain training speed and avoid device issues.
                # Use evaluate.py for full sample evaluation.
                pass

        if (
            early_stop_enabled
            and did_eval
            and epochs_without_improvement >= early_stop_patience
        ):
            stopped_early = True
            if is_main:
                print(
                    f"\n[EARLY-STOP] Triggered at epoch {epoch + 1}: "
                    f"no >{early_stop_min_delta_rel * 100:.2f}% eval improvement for "
                    f"{epochs_without_improvement} epochs. "
                    f"Best eval={best_eval_loss:.6f} @ epoch {best_epoch + 1}"
                )
            break

    save_state_dict(model, os.path.join(ckpt_dir, f"{args.save_name}_latest.pt"), is_main_process=is_main)
    if is_main:
        if stopped_early:
            print(f"Training stopped early at epoch {epoch + 1}/{args.epochs}.")
        else:
            print("Training finished.")
        print(f"Best eval checkpoint: epoch {best_epoch + 1}, EvalRel={best_eval_loss:.6f}")
        _save_plot_loss_curves()

    cleanup_ddp(ctx)
    _teardown_viz_cuda(device)

    if is_main:
        _run_auto_eval(args, run_root, ckpt_dir)
