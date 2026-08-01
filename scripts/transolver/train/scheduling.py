"""Learning-rate schedulers and early-stopping helpers for Transolver training."""

from __future__ import annotations

import torch


def is_meaningful_improvement(
    best: float,
    new: float,
    min_delta_rel: float = 0.001,
) -> bool:
    """True when new improves best by more than min_delta_rel (relative)."""
    if best == float("inf") or best <= 0.0:
        return True
    return new < best * (1.0 - min_delta_rel)


def build_lr_scheduler(optimizer, args):
    """
    Build LR scheduler from args.

    Supported lr_scheduler: none, cosine, step, plateau.
    """
    name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
    lr_min = float(getattr(args, "lr_min", 1e-6))

    if name in ("none", "off", "false", "0"):
        return None, "none"

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.epochs),
            eta_min=lr_min,
        )
        return scheduler, "epoch"

    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(getattr(args, "lr_step_size", 500)),
            gamma=float(getattr(args, "lr_gamma", 0.5)),
        )
        return scheduler, "epoch"

    if name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(args, "lr_plateau_factor", 0.5)),
            patience=int(getattr(args, "lr_plateau_patience", 50)),
            min_lr=lr_min,
            threshold=float(getattr(args, "early_stop_min_delta_rel", 0.001)),
            threshold_mode="rel",
        )
        return scheduler, "plateau"

    raise ValueError(f"Unknown lr_scheduler: {name!r} (use none/cosine/step/plateau)")


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
