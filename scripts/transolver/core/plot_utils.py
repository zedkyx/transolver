from __future__ import annotations

"""Backward-compatible helper surface for legacy imports.

The training path imports this module very early, so we keep imports lazy here:
plotting helpers only pull in heavy visualization dependencies when called.
"""

from typing import Any

from scripts.transolver.train.metrics import relative_l2


def plot_loss_curves(*args: Any, **kwargs: Any):
    from scripts.transolver.viz.loss_curves import plot_loss_curves as _plot_loss_curves

    return _plot_loss_curves(*args, **kwargs)


def plot_loss_curves_channels(*args: Any, **kwargs: Any):
    from scripts.transolver.viz.loss_curves import (
        plot_loss_curves_channels as _plot_loss_curves_channels,
    )

    return _plot_loss_curves_channels(*args, **kwargs)


def save_loss_curves_npy(*args: Any, **kwargs: Any):
    from scripts.transolver.viz.loss_curves import save_loss_curves_npy as _save_loss_curves_npy

    return _save_loss_curves_npy(*args, **kwargs)


def plot_samples(*args: Any, **kwargs: Any):
    from scripts.transolver.viz.samples import plot_samples as _plot_samples

    return _plot_samples(*args, **kwargs)


__all__ = [
    "relative_l2",
    "plot_loss_curves",
    "plot_loss_curves_channels",
    "save_loss_curves_npy",
    "plot_samples",
]
