import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from .Embedding import timestep_embedding
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .Physics_Attention import Physics_Attention_Irregular_Mesh, SpectralLayer, _broadcast_quadrature_weights


def _gumbel_softmax(logits, temperature, tau=1.0):
    """Gumbel-Softmax: 可微的离散采样近似，temperature 为 per-token (B H N 1)。"""
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = logits + gumbel_noise
    y = y / (temperature + 1e-8)
    return F.softmax(y, dim=-1)


class Physics_Attention_Irregular_EideticSlice(nn.Module):
    """
    Physics Attention for irregular mesh，Slice 分配采用 Transolver++ 的
    token-dependent 温度 + Gumbel-Softmax，其余（mask、spectral、record）与原版一致。
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, use_spectral=False,
                 spectral_modes=16, spectral_hi_freq_boost=1.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.record_attention = False
        self.use_spectral = use_spectral
        if self.use_spectral:
            self.spectral_layer = SpectralLayer(dim_head, spectral_modes, heads, hi_freq_boost=spectral_hi_freq_boost)

    def forward(self, x, mask=None, quadrature_weights=None):
        B, N, C = x.shape
        mask_n = None
        if mask is not None:
            if mask.dim() == 3:
                mask_n = mask
            else:
                mask_n = mask.unsqueeze(-1)
            mask_n = mask_n.to(dtype=x.dtype, device=x.device).view(B, 1, N, 1)

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # Eidetic Slice: token-dependent 温度 + Gumbel-Softmax
        logits = self.in_project_slice(x_mid)
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = _gumbel_softmax(logits, temperature)
        if mask_n is not None:
            slice_weights = slice_weights * mask_n
        q = _broadcast_quadrature_weights(
            quadrature_weights, B=B, N=N, dtype=slice_weights.dtype, device=slice_weights.device
        )
        slice_weights_q = slice_weights * q if q is not None else slice_weights
        slice_norm = slice_weights_q.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_q)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)

        if self.use_spectral:
            out_slice_token = out_slice_token + self.spectral_layer(out_slice_token)

        if self.record_attention:
            self.last_attn = attn.detach()
            self.last_slice_weights = slice_weights.detach()
            self.last_slice_weights_q = slice_weights_q.detach()
            self.last_slice_mass = slice_norm.detach()
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class LocalEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, act='gelu'):
        super().__init__()
        if act in ACTIVATION.keys():
            act_fn = ACTIVATION[act]
        else:
            act_fn = nn.GELU
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            act_fn(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, knn_idx, knn_valid=None):
        # x: [B, N, C]
        # knn_idx: [B, N, K]
        # knn_valid: [B, N, K]
        B, N, C = x.shape
        K = knn_idx.shape[-1]
        
        # Get neighbor features using advanced indexing
        # knn_idx: [B, N, K] -> [B, N, K, 1] for gather
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, K)  # [B, N, K]
        x_j = x[batch_idx, knn_idx, :]  # [B, N, K, C]
        
        # x_i: [B, N, K, C] - expand center features
        x_i = x.unsqueeze(2).expand(-1, -1, K, -1)  # [B, N, K, C]
        
        # Edge features: [x_i, x_j - x_i]
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)  # [B, N, K, 2*C]
        
        # MLP
        edge_feat = self.mlp(edge_feat)  # [B, N, K, C_out]
        
        if knn_valid is not None:
            # Mask invalid neighbors (padding)
            edge_feat = edge_feat * knn_valid.unsqueeze(-1).to(edge_feat.dtype)
            # Aggregate (mean)
            out = edge_feat.sum(dim=2) / (knn_valid.sum(dim=2, keepdim=True).to(edge_feat.dtype) + 1e-5)
        else:
            # Aggregate (mean)
            out = edge_feat.mean(dim=2)  # [B, N, C_out]
            
        return out


class Transolver_block(nn.Module):
    """Transformer encoder block.
    use_eidetic=True: Slice 分配用 token-dependent 温度 + Gumbel-Softmax (Transolver++)
    use_eidetic=False: 原始 Physics_Attention_Irregular_Mesh (Softmax + 固定温度)
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            use_spectral=False,
            spectral_modes=16,
            spectral_hi_freq_boost=1.0,
            use_edge_conv=False,
            use_eidetic=False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.use_edge_conv = use_edge_conv
        self.use_eidetic = use_eidetic
        if self.use_edge_conv:
            self.edge_conv = LocalEdgeConv(hidden_dim, hidden_dim, act=act)
            self.ln_edge = nn.LayerNorm(hidden_dim)

        self.ln_1 = nn.LayerNorm(hidden_dim)
        if use_eidetic:
            self.Attn = Physics_Attention_Irregular_EideticSlice(
                hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                dropout=dropout, slice_num=slice_num,
                use_spectral=use_spectral, spectral_modes=spectral_modes,
                spectral_hi_freq_boost=spectral_hi_freq_boost)
        else:
            self.Attn = Physics_Attention_Irregular_Mesh(
                hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                dropout=dropout, slice_num=slice_num,
                use_spectral=use_spectral, spectral_modes=spectral_modes,
                spectral_hi_freq_boost=spectral_hi_freq_boost)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, mask=None, knn_idx=None, knn_valid=None, quadrature_weights=None):
        if self.use_edge_conv:
            if knn_idx is not None:
                fx = self.edge_conv(self.ln_edge(fx), knn_idx, knn_valid) + fx

        ln_fx = self.ln_1(fx)
        fx = self.Attn(ln_fx, mask=mask, quadrature_weights=quadrature_weights) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 # space_dim=1,
                 space_dim=3,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 # fun_dim=1,
                 fun_dim=2,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 use_spectral=False,
                 spectral_modes=16,
                 spectral_hi_freq_boost=1.0,
                 use_edge_conv=False,
                 use_eidetic=False,
                 ):
        super(Model, self).__init__()
        self.__name__ = 'Transolver_plus' if use_eidetic else 'Transolver_1D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.use_edge_conv = use_edge_conv
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      use_spectral=use_spectral,
                                                      spectral_modes=spectral_modes,
                                                      spectral_hi_freq_boost=spectral_hi_freq_boost,
                                                      use_edge_conv=use_edge_conv,
                                                      use_eidetic=use_eidetic,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        # x: B N 2
        # grid_ref
        dev = x.device
        dt = x.dtype
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float32, device=dev)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1]).to(dtype=dt)
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float32, device=dev)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1]).to(dtype=dt)
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref * self.ref, 2)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None, mask=None, knn_idx=None, knn_valid=None, quadrature_weights=None):
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None and self.Time_Input:
            Time_emb = timestep_embedding(T, self.n_hidden).unsqueeze(1).expand(-1, fx.shape[1], -1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(
                fx,
                mask=mask,
                knn_idx=knn_idx,
                knn_valid=knn_valid,
                quadrature_weights=quadrature_weights,
            )

        return fx

    def enable_attention_recording(self, enabled=True):
        """Enable/disable attention recording for visualization."""
        for block in self.blocks:
            if hasattr(block, 'Attn') and hasattr(block.Attn, 'record_attention'):
                block.Attn.record_attention = bool(enabled)
                if enabled:
                    if hasattr(block.Attn, 'last_attn'):
                        delattr(block.Attn, 'last_attn')
                    if hasattr(block.Attn, 'last_slice_weights'):
                        delattr(block.Attn, 'last_slice_weights')

    def get_last_attention(self):
        """Get attention weights and slice weights from the last forward pass.
        
        Returns:
            list: List of dicts, each containing:
                - 'layer': int, layer index
                - 'attn': Tensor[B,H,G,G] or None, attention matrix between slice tokens
                - 'slice_weights': Tensor[B,H,N,G] or None, node-to-slice assignment weights
        """
        results = []
        for layer_index, block in enumerate(self.blocks):
            attn = getattr(block.Attn, 'last_attn', None)
            slice_weights = getattr(block.Attn, 'last_slice_weights', None)
            if (attn is not None) or (slice_weights is not None):
                results.append({'layer': layer_index, 'attn': attn, 'slice_weights': slice_weights})
        return results
