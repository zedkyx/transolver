"""
cache data loader (cache-only).

The project has been refactored to decouple "data processing / cache building"
from "training". In the container, training should ONLY rely on a pre-built
PyTorch cache at `args.cache_path`.

Expected cache format (torch.save dict):
  - pos: (B, N, 2) float32
  - fx : (B, N, F) float32
  - y  : (B, N, C) float32
Optional:
  - case_params: (B, 13) float32
  - case_names: list[str] length B
  - geom_id: (B,) int64 — merged multi-geometry cache
  - mask: (B, N) bool
  - node_weight: (B, N) float32 — Delaunay lumped nodal area (or FEM mass)
  - metadata.train_cases / metadata.test_cases for case-level split
  - metadata.train_geom_ids / metadata.test_geom_ids for geometry-level split
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch

from utils.quadrature import lumped_area_delaunay_batch


def resolve_output_cols(output_cols, num_channels: int) -> list[str]:
    """Align channel names with y.shape[-1]; pad missing names as Ch{i}."""
    if output_cols is None:
        cols: list[str] = []
    elif isinstance(output_cols, str):
        cols = output_cols.split()
    else:
        cols = list(output_cols)
    if len(cols) == 1 and cols and " " in cols[0]:
        cols = cols[0].split()
    while len(cols) < num_channels:
        cols.append(f"Ch{len(cols)}")
    return cols


def _case_y_l2_norm(y: torch.Tensor) -> torch.Tensor:
    """Per-case L2 norm of y over nodes and channels. Shape: [B]."""
    return torch.sqrt((y.float() ** 2).sum(dim=(1, 2)))


def find_degenerate_y_cases(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Return indices of cases whose ||y||_2 is below eps (all channels combined).

    These samples make relative-L2 metrics explode because the denominator ~ 0.
    """
    norms = _case_y_l2_norm(y)
    return (norms < eps).nonzero(as_tuple=True)[0]


def filter_degenerate_y_cases(
    pos: torch.Tensor,
    fx: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-6,
    case_params: torch.Tensor | None = None,
    sato_indices: dict | None = None,
    case_t: torch.Tensor | None = None,
    case_names: Sequence[str] | None = None,
    geom_id: torch.Tensor | None = None,
    node_weight: torch.Tensor | None = None,
    cache_mask: torch.Tensor | None = None,
    cache_path: str = "",
    filter_enabled: bool = True,
) -> tuple:
    """
    Detect near-zero ||y|| cases and optionally drop them before train/test split.

    Returns filtered tensors + num_removed + keep_idx.
    """
    bad_idx = find_degenerate_y_cases(y, eps=eps)
    num_bad = int(bad_idx.numel())
    if num_bad == 0:
        return pos, fx, y, case_params, sato_indices, case_t, 0, None, geom_id, node_weight, cache_mask

    norms = _case_y_l2_norm(y)[bad_idx]
    label = os.path.basename(cache_path) if cache_path else "cache"
    print(f"[!] Degenerate y cases in {label}: {num_bad} sample(s) with ||y||_2 < {eps:g}")
    for j, idx in enumerate(bad_idx.tolist()):
        name_suffix = ""
        if case_names is not None and 0 <= idx < len(case_names):
            name_suffix = f", case_name={case_names[idx]!r}"
        print(f"    [{j + 1}/{num_bad}] index={idx}, ||y||_2={norms[j].item():.6e}{name_suffix}")

    if not filter_enabled:
        print("    filter_degenerate_y=0: kept all samples (TrainRel may be misleading)")
        return pos, fx, y, case_params, sato_indices, case_t, 0, None, geom_id, node_weight, cache_mask

    keep = _case_y_l2_norm(y) >= eps
    keep_idx = keep.nonzero(as_tuple=True)[0]
    pos = pos[keep_idx]
    fx = fx[keep_idx]
    y = y[keep_idx]
    if case_params is not None:
        case_params = case_params[keep_idx]
    if sato_indices is not None:
        sato_indices = {
            key: val[keep_idx] for key, val in sato_indices.items()
        }
    if case_t is not None:
        case_t = case_t[keep_idx]
    if geom_id is not None:
        geom_id = geom_id[keep_idx]
    if node_weight is not None:
        node_weight = node_weight[keep_idx]
    if cache_mask is not None:
        cache_mask = cache_mask[keep_idx]
    print(f"    filter_degenerate_y=1: removed {num_bad}, remaining {int(keep_idx.numel())} sample(s)")
    return pos, fx, y, case_params, sato_indices, case_t, num_bad, keep_idx, geom_id, node_weight, cache_mask


def _resolve_weight_mask(
    pos: torch.Tensor,
    cache_mask: torch.Tensor | None,
    *,
    padding_enabled: bool,
    padding_value: float,
) -> torch.Tensor | None:
    if cache_mask is not None:
        return cache_mask.bool()
    if padding_enabled:
        pad = torch.tensor(padding_value, dtype=pos.dtype)
        return ~((pos == pad).all(dim=-1))
    return None


def compute_or_load_node_weight(
    pos: torch.Tensor,
    *,
    node_weight: torch.Tensor | None,
    cache_mask: torch.Tensor | None,
    use_node_weight: bool,
    padding_enabled: bool = False,
    padding_value: float = 0.0,
    max_edge: float | None = None,
    persist_path: str = "",
) -> torch.Tensor | None:
    """Return [B,N] node_weight when use_node_weight, else None."""
    if not use_node_weight:
        return None

    if node_weight is not None:
        nw = node_weight.float()
        if nw.shape != pos.shape[:2]:
            raise ValueError(f"node_weight shape {tuple(nw.shape)} != pos {tuple(pos.shape[:2])}")
        print(
            f"[*] node_weight from cache: shape={tuple(nw.shape)}, "
            f"min={nw[nw > 0].min().item() if (nw > 0).any() else 0:.3e}, "
            f"max={nw.max().item():.3e}, mean={nw.mean().item():.3e}"
        )
        return nw

    mask = _resolve_weight_mask(
        pos, cache_mask, padding_enabled=padding_enabled, padding_value=padding_value
    )
    print("[*] node_weight missing in cache; computing Delaunay lumped area ...")
    q_np = lumped_area_delaunay_batch(
        pos.numpy(),
        None if mask is None else mask.numpy(),
        max_edge=max_edge,
        show_progress=True,
    )
    nw = torch.from_numpy(q_np)
    print(
        f"[*] node_weight computed: shape={tuple(nw.shape)}, "
        f"min={nw[nw > 0].min().item() if (nw > 0).any() else 0:.3e}, "
        f"max={nw.max().item():.3e}, mean={nw.mean().item():.3e}"
    )

    if persist_path:
        persist_node_weight_to_cache(persist_path, nw, source_pos_shape=tuple(pos.shape))
    return nw


def persist_node_weight_to_cache(
    cache_path: str,
    node_weight: torch.Tensor,
    *,
    source_pos_shape: tuple | None = None,
) -> None:
    """Write/overwrite node_weight in an existing cache.pt (full file rewrite)."""
    if not cache_path:
        raise ValueError("persist_node_weight requires cache_path")
    cache = _load_cache(cache_path)
    pos = cache.get("pos")
    if pos is None:
        raise KeyError(f"cache missing pos: {cache_path}")
    # Only persist when shapes match the on-disk cache (no subsample/filter applied).
    if tuple(pos.shape[:2]) != tuple(node_weight.shape):
        print(
            f"[!] skip persist node_weight: cache pos {tuple(pos.shape)} "
            f"!= weight {tuple(node_weight.shape)} "
            f"(likely frame subsample / degenerate filter). "
            f"Re-run without subsample to write into cache."
        )
        return
    if source_pos_shape is not None and tuple(source_pos_shape) != tuple(pos.shape):
        print(
            f"[!] skip persist node_weight: in-memory pos {source_pos_shape} "
            f"!= cache pos {tuple(pos.shape)}"
        )
        return
    cache["node_weight"] = node_weight.detach().cpu().float()
    meta = cache.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["node_weight_method"] = "delaunay_lumped_area"
    cache["metadata"] = meta
    torch.save(cache, cache_path)
    print(f"[*] persisted node_weight -> {cache_path}")


def frame_subsample_active(args) -> bool:
    stride = int(getattr(args, "frame_stride", 1))
    offset = int(getattr(args, "frame_offset", 0))
    max_frames = int(getattr(args, "max_frames", 0))
    return stride > 1 or offset > 0 or max_frames > 0


def default_coord_norm_path(
    cache_path: str,
    *,
    frame_stride: int = 1,
    frame_offset: int = 0,
    max_frames: int = 0,
    user_path: str = "",
) -> str:
    """Derive coord_norm path; subsampled runs use a separate stats file."""
    if not cache_path:
        return user_path
    auto_default = cache_path.replace(".pt", "_coord_norm.pt")
    if user_path and user_path != auto_default:
        return user_path
    if frame_stride <= 1 and frame_offset <= 0 and max_frames <= 0:
        return auto_default
    base = cache_path.replace(".pt", "")
    tag = f"_coord_norm_sub_s{frame_stride}_o{frame_offset}"
    if max_frames > 0:
        tag += f"_n{max_frames}"
    return f"{base}{tag}.pt"


def apply_frame_subsample(
    pos: torch.Tensor,
    fx: torch.Tensor,
    y: torch.Tensor,
    *,
    frame_stride: int = 1,
    frame_offset: int = 0,
    max_frames: int = 0,
    case_params: torch.Tensor | None = None,
    sato_indices: dict | None = None,
    case_t: torch.Tensor | None = None,
    case_names: Sequence[str] | None = None,
    geom_id: torch.Tensor | None = None,
    node_weight: torch.Tensor | None = None,
    cache_mask: torch.Tensor | None = None,
    cache_path: str = "",
) -> tuple:
    """Temporal subsample along batch dim B."""
    stride = int(frame_stride)
    offset = int(frame_offset)
    max_frames = int(max_frames)
    if stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {stride}")
    if offset < 0:
        raise ValueError(f"frame_offset must be >= 0, got {offset}")

    if stride == 1 and offset == 0 and max_frames <= 0:
        cn = list(case_names) if case_names is not None else None
        return pos, fx, y, case_params, sato_indices, case_t, cn, geom_id, node_weight, cache_mask, False

    b = int(pos.shape[0])
    idx = torch.arange(offset, b, stride, dtype=torch.long)
    if max_frames > 0:
        idx = idx[:max_frames]
    if idx.numel() == 0:
        raise ValueError(
            f"frame subsample empty: B={b}, stride={stride}, offset={offset}, max_frames={max_frames}"
        )

    pos = pos[idx]
    fx = fx[idx]
    y = y[idx]
    if case_params is not None:
        case_params = case_params[idx]
    if sato_indices is not None:
        sato_indices = {key: val[idx] for key, val in sato_indices.items()}
    if case_t is not None:
        case_t = case_t[idx]
    if geom_id is not None:
        geom_id = geom_id[idx]
    if node_weight is not None:
        node_weight = node_weight[idx]
    if cache_mask is not None:
        cache_mask = cache_mask[idx]
    cn = None
    if case_names is not None:
        cn = [case_names[i] for i in idx.tolist()]

    label = os.path.basename(cache_path) if cache_path else "cache"
    t_msg = ""
    if case_t is not None and case_t.numel() > 0:
        t_msg = f", t=[{case_t.min():.4f}, {case_t.max():.4f}]"
    print(
        f"[*] frame subsample ({label}): B {b} -> {int(idx.numel())} "
        f"(stride={stride}, offset={offset}, max_frames={max_frames or 'all'}){t_msg}"
    )
    return pos, fx, y, case_params, sato_indices, case_t, cn, geom_id, node_weight, cache_mask, True


def _load_cache(cache_path: str) -> dict:
    if not cache_path:
        raise ValueError("cache_path is empty. Provide --cache_path pointing to a pre-built cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"cache_path not found: {cache_path}")

    try:
        obj = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(cache_path, map_location="cpu")

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported cache format: expected dict, got {type(obj)} at {cache_path}")
    return obj


def load_cache_data(args) -> Tuple:
    """
    Load cache tensors from a pre-built cache.

    Returns:
      pos, fx, y, input_dim, output_dim, case_params, sato_indices, case_t, node_weight
      node_weight is [B,N] when use_node_weight=1, else None.
    """
    if getattr(args, "rebuild_cache", False):
        raise RuntimeError(
            "This environment is configured for cache-only training. "
            "Please build cache offline (outside the training container) and pass --cache_path."
        )

    cache_path = getattr(args, "cache_path", "")
    cache = _load_cache(cache_path)

    for k in ("pos", "fx", "y"):
        if k not in cache:
            raise KeyError(f"cache missing key '{k}'. Found keys={sorted(cache.keys())}")

    pos = cache["pos"]
    fx = cache["fx"]
    y = cache["y"]
    case_params = cache.get("case_params", None)
    node_weight = cache.get("node_weight", None)
    cache_mask = cache.get("mask", None)

    case_t = None
    if "t" in cache:
        case_t = cache["t"].float().view(-1)
    elif fx.shape[-1] >= 1:
        case_t = fx[:, 0, 0].float().clone()

    empty_fx = bool(int(getattr(args, "empty_fx", 0)))
    if empty_fx:
        fx = torch.zeros(pos.shape[0], pos.shape[1], 0, dtype=pos.dtype)
        if case_t is None:
            raise ValueError("empty_fx=1 需要 cache 中的 't' 字段（time_input 用）")
        print(f"[*] empty_fx=1: fx 置为空 (B,N,0)，时间来自 case_t，范围 [{case_t.min():.4f}, {case_t.max():.4f}]")

    # Load SATO indices if requested
    sato_indices = None
    if getattr(args, "use_sato", False):
        sato_index_path = getattr(args, "sato_index_path", "")
        if not sato_index_path and getattr(args, "cache_path", ""):
            sato_index_path = args.cache_path.replace(".pt", "_sato_index.pt")
            setattr(args, "sato_index_path", sato_index_path)

        if sato_index_path and os.path.exists(sato_index_path):
            sato_indices = torch.load(sato_index_path, map_location="cpu")
            print(f"[*] Loaded SATO indices from {sato_index_path}")
        else:
            print(f"Warning: SATO indices not found at {sato_index_path}, but use_sato is True.")

    if not (isinstance(pos, torch.Tensor) and isinstance(fx, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("cache tensors must be torch.Tensor: pos/fx/y")

    if pos.ndim != 3 or fx.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors. Got pos={pos.shape}, fx={fx.shape}, y={y.shape}")
    if pos.shape[:2] != fx.shape[:2] or pos.shape[:2] != y.shape[:2]:
        raise ValueError(f"Shape mismatch among pos/fx/y: pos={pos.shape}, fx={fx.shape}, y={y.shape}")

    pos = pos.float()
    fx = fx.float()
    y = y.float()
    if node_weight is not None:
        node_weight = node_weight.float()
        if node_weight.shape != pos.shape[:2]:
            raise ValueError(
                f"cache node_weight shape {tuple(node_weight.shape)} != pos {tuple(pos.shape[:2])}"
            )
    if cache_mask is not None:
        cache_mask = cache_mask.bool()
        if cache_mask.shape != pos.shape[:2]:
            raise ValueError(
                f"cache mask shape {tuple(cache_mask.shape)} != pos {tuple(pos.shape[:2])}"
            )

    case_names = cache.get("case_names", None)
    geom_id = cache.get("geom_id", None)
    if geom_id is not None:
        geom_id = geom_id.long().view(-1)

    # Persist only when full cache is used (no subsample / no degenerate drop).
    can_persist = (
        int(getattr(args, "frame_stride", 1)) <= 1
        and int(getattr(args, "frame_offset", 0)) <= 0
        and int(getattr(args, "max_frames", 0)) <= 0
    )

    pos, fx, y, case_params, sato_indices, case_t, case_names, geom_id, node_weight, cache_mask, subsampled = (
        apply_frame_subsample(
            pos,
            fx,
            y,
            frame_stride=int(getattr(args, "frame_stride", 1)),
            frame_offset=int(getattr(args, "frame_offset", 0)),
            max_frames=int(getattr(args, "max_frames", 0)),
            case_params=case_params,
            sato_indices=sato_indices,
            case_t=case_t,
            case_names=case_names,
            geom_id=geom_id,
            node_weight=node_weight,
            cache_mask=cache_mask,
            cache_path=cache_path,
        )
    )
    setattr(args, "frame_subsample_applied", subsampled)
    if subsampled:
        can_persist = False

    filter_deg = bool(int(getattr(args, "filter_degenerate_y", 1)))
    deg_eps = float(getattr(args, "y_degenerate_eps", 1e-6))
    pos, fx, y, case_params, sato_indices, case_t, num_removed, keep_idx, geom_id, node_weight, cache_mask = (
        filter_degenerate_y_cases(
            pos,
            fx,
            y,
            eps=deg_eps,
            case_params=case_params,
            sato_indices=sato_indices,
            case_t=case_t,
            case_names=case_names,
            geom_id=geom_id,
            node_weight=node_weight,
            cache_mask=cache_mask,
            cache_path=cache_path,
            filter_enabled=filter_deg,
        )
    )
    if keep_idx is not None:
        can_persist = False
    if keep_idx is not None and case_names is not None:
        setattr(args, "filtered_case_names", [case_names[i] for i in keep_idx.tolist()])
    elif case_names is not None:
        setattr(args, "filtered_case_names", list(case_names))
    if geom_id is not None:
        setattr(args, "filtered_geom_id", geom_id)

    use_nw = bool(int(getattr(args, "use_node_weight", 0)))
    persist = bool(int(getattr(args, "persist_node_weight", 0)))
    max_edge = getattr(args, "node_weight_max_edge", None)
    if max_edge is not None and float(max_edge) <= 0:
        max_edge = None
    elif max_edge is not None:
        max_edge = float(max_edge)

    had_cached_nw = node_weight is not None
    node_weight = compute_or_load_node_weight(
        pos,
        node_weight=node_weight,
        cache_mask=cache_mask,
        use_node_weight=use_nw,
        padding_enabled=bool(int(getattr(args, "padding", 0))),
        padding_value=float(getattr(args, "padding_value", 0.0)),
        max_edge=max_edge,
        persist_path=(cache_path if (persist and use_nw and not had_cached_nw and can_persist) else ""),
    )

    input_dim = int(fx.shape[-1])
    output_dim = int(y.shape[-1])
    return pos, fx, y, input_dim, output_dim, case_params, sato_indices, case_t, node_weight


def get_train_test_indices(num_samples: int, train_split: float = 0.8, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate deterministic random indices for train/test split.
    
    Args:
        num_samples: Total number of samples (B)
        train_split: Proportion of samples for training
        seed: Random seed for reproducibility
        
    Returns:
        train_idx: torch.Tensor of training indices
        test_idx: torch.Tensor of testing indices
    """
    import numpy as np
    
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * train_split)
    # Avoid torch.from_numpy(): mixed numpy/torch installs can fail isinstance checks.
    train_idx = torch.tensor(indices[:split_idx].tolist(), dtype=torch.long)
    test_idx = torch.tensor(indices[split_idx:].tolist(), dtype=torch.long)
    
    return train_idx, test_idx


def get_train_test_indices_by_case(
    case_names: Sequence[str],
    train_cases: Sequence[str],
    test_cases: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sample indices by case name (all timesteps of a case stay together)."""
    train_set = set(str(c) for c in train_cases)
    test_set = set(str(c) for c in test_cases)
    overlap = train_set & test_set
    if overlap:
        raise ValueError("train_cases and test_cases overlap: {}".format(sorted(overlap)))

    train_idx, test_idx = [], []
    unknown = set()
    for i, name in enumerate(case_names):
        name = str(name)
        if name in train_set:
            train_idx.append(i)
        elif name in test_set:
            test_idx.append(i)
        else:
            unknown.add(name)
    if unknown:
        raise ValueError(
            "Samples from cases not in train/test lists: {} (train={}, test={})".format(
                sorted(unknown), sorted(train_set), sorted(test_set)
            )
        )
    if not train_idx:
        raise ValueError("No training samples after case split")
    if not test_idx:
        raise ValueError("No test samples after case split")

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def _normalize_geom_id_list(values) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            return None
        return [int(x) for x in values]
    return [int(values)]


def get_train_test_indices_by_geom_id(
    geom_id: torch.Tensor,
    *,
    train_geom_ids: Sequence[int] | None = None,
    test_geom_ids: Sequence[int] | None = None,
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """
    按几何 id 划分：同一 geom 的全部时间帧只进 train 或 test，互不重叠。
    """
    import numpy as np

    gid = geom_id.long().view(-1)
    unique = sorted(int(x) for x in torch.unique(gid).tolist())
    if len(unique) < 2:
        raise ValueError("geom split 需要至少 2 个几何，当前 unique geom_id={}".format(unique))

    train_list = _normalize_geom_id_list(train_geom_ids)
    test_list = _normalize_geom_id_list(test_geom_ids)

    if train_list is not None and test_list is not None:
        train_set = set(train_list)
        test_set = set(test_list)
    else:
        ids = list(unique)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(ids)
        n_train = max(1, int(len(ids) * float(train_split)))
        n_train = min(n_train, len(ids) - 1)
        train_set = set(ids[:n_train])
        test_set = set(ids[n_train:])

    overlap = train_set & test_set
    if overlap:
        raise ValueError("train_geom_ids 与 test_geom_ids 重叠: {}".format(sorted(overlap)))

    unknown = set(unique) - train_set - test_set
    if unknown:
        raise ValueError("存在未分配的几何 geom_id: {}".format(sorted(unknown)))

    train_idx, test_idx = [], []
    for i, g in enumerate(gid.tolist()):
        if g in train_set:
            train_idx.append(i)
        elif g in test_set:
            test_idx.append(i)
    if not train_idx or not test_idx:
        raise ValueError(
            "geom split 后 train/test 为空: train_geoms={}, test_geoms={}".format(
                sorted(train_set), sorted(test_set)
            )
        )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
        sorted(train_set),
        sorted(test_set),
    )


def resolve_train_test_indices(
    cache: dict,
    train_split: float = 0.8,
    seed: int = 42,
    num_samples: int | None = None,
    case_names: Sequence[str] | None = None,
    geom_id: torch.Tensor | None = None,
    split_by: str = "random",
    train_geom_ids: Sequence[int] | None = None,
    test_geom_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Resolve train/test indices.

    Priority:
      1. split_by=geom + geom_id（或 cache.geom_id）
      2. metadata train_cases + test_cases with per-sample case_names
      3. random split by train_split ratio
    """
    meta = cache.get("metadata", {}) or {}
    split_by = str(split_by or "random").lower()

    if geom_id is None and "geom_id" in cache:
        geom_id = cache["geom_id"]
        if isinstance(geom_id, torch.Tensor):
            geom_id = geom_id.long().view(-1)

    train_geom_ids = _normalize_geom_id_list(train_geom_ids)
    test_geom_ids = _normalize_geom_id_list(test_geom_ids)

    if split_by == "geom":
        if train_geom_ids is None:
            train_geom_ids = _normalize_geom_id_list(meta.get("train_geom_ids"))
        if test_geom_ids is None:
            test_geom_ids = _normalize_geom_id_list(meta.get("test_geom_ids"))
        if geom_id is None:
            raise ValueError("split_by=geom 需要 cache 中的 geom_id 字段")
        train_idx, test_idx, train_geoms, test_geoms = get_train_test_indices_by_geom_id(
            geom_id,
            train_geom_ids=train_geom_ids,
            test_geom_ids=test_geom_ids,
            train_split=train_split,
            seed=seed,
        )
        mode = "geom_split({} train / {} test geoms, {} train / {} test frames, seed={})".format(
            len(train_geoms),
            len(test_geoms),
            int(train_idx.numel()),
            int(test_idx.numel()),
            seed,
        )
        mode += " train_geom_ids={} test_geom_ids={}".format(train_geoms, test_geoms)
        return train_idx, test_idx, mode

    train_cases = meta.get("train_cases", None)
    test_cases = meta.get("test_cases", None)
    cn = case_names if case_names is not None else cache.get("case_names", None)

    if split_by == "case" and train_cases is not None and test_cases is not None and cn is not None:
        train_idx, test_idx = get_train_test_indices_by_case(cn, train_cases, test_cases)
        mode = "case_split({} train / {} test cases, {} train / {} test samples)".format(
            len(train_cases),
            len(test_cases),
            int(train_idx.numel()),
            int(test_idx.numel()),
        )
        return train_idx, test_idx, mode

    if train_cases is not None and test_cases is not None and cn is not None:
        train_idx, test_idx = get_train_test_indices_by_case(cn, train_cases, test_cases)
        mode = "case_split({} train / {} test cases, {} train / {} test samples)".format(
            len(train_cases),
            len(test_cases),
            int(train_idx.numel()),
            int(test_idx.numel()),
        )
        return train_idx, test_idx, mode

    cache_b = int(cache["pos"].shape[0])
    b = int(num_samples) if num_samples is not None else cache_b
    train_idx, test_idx = get_train_test_indices(b, train_split=train_split, seed=seed)
    mode = "random_split(ratio={:.3f}, seed={})".format(train_split, seed)
    if num_samples is not None and num_samples != cache_b:
        mode += ", after_degenerate_y_filter"
    if split_by not in ("random", "case", "geom"):
        mode += " (unknown split_by={!r}, fallback random)".format(split_by)
    return train_idx, test_idx, mode


def resolve_coord_norm_path(args) -> str:
    """Resolve coord_norm path from cache + frame subsample settings."""
    path = default_coord_norm_path(
        getattr(args, "cache_path", ""),
        frame_stride=int(getattr(args, "frame_stride", 1)),
        frame_offset=int(getattr(args, "frame_offset", 0)),
        max_frames=int(getattr(args, "max_frames", 0)),
        user_path=str(getattr(args, "coord_norm_path", "") or ""),
    )
    setattr(args, "coord_norm_path", path)
    return path


