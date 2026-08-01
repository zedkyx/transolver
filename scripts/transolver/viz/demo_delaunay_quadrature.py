"""Demo: Delaunay lumped nodal area q_i on a 2D irregular point cloud.

Temporary viz script — can delete after the method is confirmed.

  conda activate upt
  python scripts/transolver/viz/demo_delaunay_quadrature.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.quadrature import convex_hull_area, lumped_area_delaunay, triangle_areas
from scripts.transolver.viz.style import apply_paper_rcparams, save_paper_fig, style_axes, style_colorbar

OUT_DIR = ROOT / "runs" / "demo" / "delaunay_quadrature"


def make_nonuniform_cloud(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # sparse background in [0,1]^2
    n_bg = 60
    bg = rng.uniform(0.05, 0.95, size=(n_bg, 2))
    # local refinement cluster near (0.35, 0.35)
    n_dense = 80
    dense = rng.normal(loc=[0.35, 0.35], scale=0.06, size=(n_dense, 2))
    dense = np.clip(dense, 0.02, 0.98)
    return np.vstack([bg, dense]).astype(np.float64)


def print_toy_4point() -> None:
    pos = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.4, 0.4],
        ],
        dtype=np.float64,
    )
    names = ["P0", "P1", "P2", "P3"]
    delaunay = Delaunay(pos)
    areas = triangle_areas(pos, delaunay.simplices)
    q = lumped_area_delaunay(pos)

    print("=== toy 4-point example ===")
    for t, (i, j, k) in enumerate(delaunay.simplices):
        print(
            f"  T{t}: ({names[i]},{names[j]},{names[k]})  "
            f"A={areas[t]:.6f}  share/vert={areas[t]/3:.6f}"
        )
    for i, name in enumerate(names):
        print(f"  {name}{tuple(pos[i])}  q={q[i]:.6f}")
    print(f"  sum(q)={q.sum():.6f}  hull_area={convex_hull_area(pos):.6f}")
    print()


def main() -> None:
    apply_paper_rcparams()
    print_toy_4point()

    pos = make_nonuniform_cloud(seed=0)
    q = lumped_area_delaunay(pos)
    delaunay = Delaunay(pos)
    areas = triangle_areas(pos, delaunay.simplices)
    hull_a = convex_hull_area(pos)

    print("=== nonuniform cloud ===")
    print(f"  N={pos.shape[0]}  n_tri={delaunay.simplices.shape[0]}")
    print(
        f"  q: min={q.min():.6e}  median={np.median(q):.6e}  "
        f"max={q.max():.6e}  mean={q.mean():.6e}"
    )
    print(f"  sum(q)={q.sum():.6f}  sum(A_T)={areas.sum():.6f}  hull_area={hull_a:.6f}")
    # dense vs sparse split by distance to cluster center
    dist = np.linalg.norm(pos - np.array([0.35, 0.35]), axis=1)
    dense_mask = dist < 0.15
    print(
        f"  dense (r<0.15) mean q={q[dense_mask].mean():.6e}  n={dense_mask.sum()}"
    )
    print(
        f"  sparse (r>=0.15) mean q={q[~dense_mask].mean():.6e}  n={(~dense_mask).sum()}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    ax0 = axes[0]
    ax0.triplot(pos[:, 0], pos[:, 1], delaunay.simplices, color="0.55", lw=0.4)
    ax0.scatter(pos[:, 0], pos[:, 1], s=8, c="0.15", zorder=3)
    ax0.set_aspect("equal")
    ax0.set_xlabel(r"$x$")
    ax0.set_ylabel(r"$y$")
    style_axes(ax0)

    ax1 = axes[1]
    ax1.triplot(pos[:, 0], pos[:, 1], delaunay.simplices, color="0.75", lw=0.3, zorder=1)
    sc = ax1.scatter(
        pos[:, 0],
        pos[:, 1],
        c=q,
        s=18,
        cmap="viridis",
        zorder=3,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    style_colorbar(cbar, label=r"$q$")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    style_axes(ax1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "delaunay_lumped_q.png"
    save_paper_fig(fig, out)  # no caption sidecar
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
