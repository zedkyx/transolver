import os
import math
import torch
import matplotlib.pyplot as plt
import numpy as np


def _ensure_plot_dir(base_dir: str):
    plot_dir = os.path.join(base_dir, 'plot')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def _setup_paper_font():
    """论文风格：英文字体 Times New Roman，不可用时回退 DejaVu Serif"""
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']


def _plot_slice_weight_grid(
    pos_np: np.ndarray,
    slice_weights: np.ndarray,
    layer: int,
    head: int,
    tag: str,
    plot_dir: str,
):
    _setup_paper_font()
    num_nodes, num_slices = slice_weights.shape
    ncols = min(8, num_slices)
    nrows = int(math.ceil(num_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows), squeeze=False)
    fig.patch.set_facecolor('white')

    for s in range(num_slices):
        r, c = divmod(s, ncols)
        ax = axes[r, c]
        w = slice_weights[:, s]
        # 每 slice 独立量程（percentile 1,99），凸显空间分布模式，符合论文 Learned Slice Visualization 风格
        vmin = float(np.percentile(w, 1))
        vmax = float(np.percentile(w, 99))
        ax.scatter(
            pos_np[:, 0], pos_np[:, 1],
            c=w,
            cmap='viridis',
            s=1.5,
            alpha=0.9,
            edgecolors='none',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    total_axes = nrows * ncols
    for s in range(num_slices, total_axes):
        r, c = divmod(s, ncols)
        axes[r, c].axis('off')

    # 先排版子图，再在图右侧留空处放置纵向居中的 colorbar
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    cb_w, cb_h = 0.018, 0.65
    cb_left = 0.91
    cb_bottom = (1.0 - cb_h) / 2.0
    cax = fig.add_axes([cb_left, cb_bottom, cb_w, cb_h])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    out_path = os.path.join(plot_dir, f'{tag}_layer{layer}_head{head}_slice_weight_grid.png')
    fig.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_attention_heatmaps(attn_list, tag: str = 'attn', base_dir: str = '.', pos: np.ndarray = None, plot_slice_grid: bool = True,
                           last_layer_only: bool = True):
    """
    attn_list: list of dicts from model.get_last_attention()
        each item: { 'layer': int, 'attn': Tensor[B,H,G,G], 'slice_weights': Tensor[B,H,N,G] }
    pos: optional, spatial coordinates [N, 2] for mapping slice weights back to geometry
    last_layer_only: 仅绘制最后一层的切片可视化
    """
    plot_dir = _ensure_plot_dir(base_dir)
    max_layer = max((item.get('layer', -1) for item in attn_list), default=-1)
    for item in attn_list:
        layer = item.get('layer', -1)
        attn = item.get('attn', None)
        slice_weights = item.get('slice_weights', None)

        # if isinstance(attn, torch.Tensor):
        #     attn_np = attn[0].detach().cpu().numpy()  # [H, G, G]
        #     num_heads = attn_np.shape[0]
        #     for h in range(num_heads):
        #         plt.figure(figsize=(4, 4))
        #         plt.imshow(attn_np[h], cmap='magma', aspect='auto')
        #         plt.colorbar()
        #         plt.title(f'Layer {layer} Head {h} attention')
        #         out_path = os.path.join(plot_dir, f'{tag}_layer{layer}_head{h}.png')
        #         plt.tight_layout()
        #         plt.savefig(out_path, dpi=200)
        #         plt.close()

        if isinstance(slice_weights, torch.Tensor):
            if last_layer_only and layer != max_layer:
                continue
            sw = slice_weights[0].detach().cpu().numpy()  # [H, N, G]
            num_heads, num_nodes, num_slices = sw.shape
            
            # 1. 保留原有的抽象可视化（slice权重均值）
            # sw_mean = sw.mean(axis=1)  # [H, G]
            # plt.figure(figsize=(6, 3))
            # plt.imshow(sw_mean, cmap='viridis', aspect='auto')
            # plt.colorbar()
            # plt.title(f'Layer {layer} slice-weight mean over nodes')
            # plt.xlabel('Slice Index')
            # plt.ylabel('Head Index')
            # out_path = os.path.join(plot_dir, f'{tag}_layer{layer}_sliceweights_mean.png')
            # plt.tight_layout()
            # plt.savefig(out_path, dpi=200)
            # plt.close()
            
            # 2. 如果有空间坐标，映射回几何形状，仅绘制 Learned Slice Visualization 网格图
            if pos is not None:
                pos_np = pos[0].cpu().numpy() if isinstance(pos, torch.Tensor) else pos[0] if pos.ndim == 3 else pos
                # pos_np: [N, 2]
                for h in range(num_heads):
                    if plot_slice_grid:
                        _plot_slice_weight_grid(
                            pos_np=pos_np,
                            slice_weights=sw[h],
                            layer=layer,
                            head=h,
                            tag=tag,
                            plot_dir=plot_dir,
                        )

    return plot_dir


