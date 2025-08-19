# vit_s_continuous_deep_embed_experiments.py
# ViT‑S/16 + Continuous Deep‑Embed (CDE) experimental harness
# - per‑config checkpoints (best/last) in dedicated subdirs
# - manifest CSV/LaTeX with checkpoint paths
# - mlflow logging (grid parent + per‑config nested runs)
# - robust training: tf32, channels_last, grad accumulation, warmup‑cosine
# - windows-friendly dataloader defaults; suppresses noisy third‑party logs
# - kaggle cls-loc layout support (--imagenet_layout cls-loc)
# - performance flags: bf16/fp16 autocast, fused adamw, flash sdpa, torch.compile,
#   dataloader prefetch/drop_last
# - upgrades for a stronger baseline:
#   rotary positional embeddings (rope, true 2d by default; 1d fallback if needed)
#   randaugment + random erasing
#   mixup / cutmix + soft-target cross-entropy
#   stochastic depth (drop_path)
#   model ema (on by default; best checkpoint selection prefers ema acc)
#   layer-wise lr decay (llrd)
#   optional grad checkpointing

# --- tame noisy third-party import logs (OpenVINO, HF telemetry) ---
import os, warnings
os.environ.setdefault("OPENVINO_LOG_LEVEL", "error")
os.environ.setdefault("OV_LOG_LEVEL", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", module="openvino")

import sys
import math
import json
import time
import argparse
import random
import csv
import hashlib
import datetime as _dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import xml.etree.ElementTree as ET
from contextlib import contextmanager  # NEW

# Optional deps (degrade gracefully)
try:
    import timm
except Exception as _e:
    timm = None

try:
    import mlflow
    _HAVE_MLFLOW = True
except Exception:
    mlflow = None
    _HAVE_MLFLOW = False

import pandas as pd


# ----------------------- small utilities ---------------------------------

def now_utc() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def sizeof_fmt(num: float, suffix="B") -> str:
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}E{suffix}"


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
    # TF32 for Ampere+/Hopper (faster matmul/conv with minimal accuracy impact)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------- dataset root assertions & helpers (for cls-loc) ----------

def _resolve_ilsvrc_root(root: Path) -> Path:
    """
    Accepts:
      - <base> that contains 'ILSVRC/'
      - 'ILSVRC/' itself
      - '.../ILSVRC/Data/CLS-LOC' (will walk up)
    Returns the 'ILSVRC/' directory.
    """
    p = Path(root)
    # Case 1: base contains ILSVRC
    if (p / "ILSVRC" / "Data" / "CLS-LOC").is_dir():
        return p / "ILSVRC"
    # Case 2: p is ILSVRC
    if (p / "Data" / "CLS-LOC").is_dir() and p.name == "ILSVRC":
        return p
    # Case 3: p ends with .../Data/CLS-LOC
    if p.name == "CLS-LOC" and p.parent.name == "Data" and p.parent.parent.name == "ILSVRC":
        return p.parent.parent
    # Case 4: allow passing .../ILSVRC directly but without Data/CLS-LOC populated yet
    if p.name == "ILSVRC" and (p / "Data" / "CLS-LOC").exists():
        return p
    raise FileNotFoundError(
        f"Could not locate ILSVRC root from '{root}'. "
        "Expected one of: <base>/ILSVRC, <base>/ILSVRC/Data/CLS-LOC, or the ILSVRC directory itself."
    )


def assert_imagenet_root(root: str, layout: str = "folder"):
    if layout == "folder":
        train = Path(root) / "train"
        val = Path(root) / "val"
        if not train.is_dir() or not val.is_dir():
            raise FileNotFoundError(
                f"--imagenet must point to a folder with 'train' and 'val'. Got: {root}"
            )
        # quick sanity: at least one class folder in each
        if not any(train.iterdir()):
            raise FileNotFoundError(f"No class folders found in {train}")
        if not any(val.iterdir()):
            raise FileNotFoundError(f"No class folders found in {val}")
    elif layout == "cls-loc":
        ils = _resolve_ilsvrc_root(Path(root))
        data_train = ils / "Data" / "CLS-LOC" / "train"
        data_val = ils / "Data" / "CLS-LOC" / "val"
        ann_train = ils / "Annotations" / "CLS-LOC" / "train"
        ann_val = ils / "Annotations" / "CLS-LOC" / "val"
        if not data_train.is_dir():
            raise FileNotFoundError(f"Missing train images dir: {data_train}")
        if not ann_train.is_dir():
            warnings.warn(f"Missing train annotations dir (not required for classification): {ann_train}")
        if not data_val.is_dir():
            raise FileNotFoundError(f"Missing val images dir: {data_val}")
        if not ann_val.is_dir():
            raise FileNotFoundError(f"Missing val annotations dir (required to infer labels): {ann_val}")
    else:
        raise ValueError(f"Unknown layout '{layout}' (choices: 'folder', 'cls-loc').")


# ----------------------- model: gates ------------------------------------

class SoftCodebookGate(nn.Module):
    """Soft top-k codebook gate: computes g(z)=Σ α_k E_k and scales target: target*(1+g)."""
    def __init__(
        self,
        K: int,
        d_in: int,
        d_g: int,
        tau: float = 10.0,
        topk: int = 8,
        share_codebook: Optional[nn.Parameter] = None,
        share_E: Optional[nn.Parameter] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.codebook = share_codebook if share_codebook is not None \
            else nn.Parameter(torch.randn(K, d_in) / math.sqrt(d_in))
        if share_E is not None:
            if share_E.shape != (K, d_g):
                raise ValueError(f"shared E has shape {tuple(share_E.shape)}; expected {(K, d_g)}")
            self.E = share_E
        else:
            self.E = nn.Parameter(torch.zeros(K, d_g))
            nn.init.zeros_(self.E)
        self.tau = tau
        self.topk = topk
        self.normalize = normalize

    def forward(self, z, target):
        # z: [B,N,d_in] (pre-MLP normed input), target: [B,N,d_g] (MLP hidden or output)
        zf = F.normalize(z, dim=-1) if self.normalize else z
        Cf = F.normalize(self.codebook, dim=-1) if self.normalize else self.codebook
        logits = self.tau * torch.einsum('bnd,kd->bnk', zf, Cf)  # [B,N,K]
        if self.topk is not None and self.topk < logits.shape[-1]:
            vals, idx = torch.topk(logits, self.topk, dim=-1)       # [B,N,k]
            alpha = F.softmax(vals, dim=-1)                          # [B,N,k]
            Ek = self.E[idx]                                         # [B,N,k,d_g]
            g = (alpha.unsqueeze(-1) * Ek).sum(dim=-2)               # [B,N,d_g]
        else:
            alpha = F.softmax(logits, dim=-1)                        # [B,N,K]
            g = torch.einsum('bnk,kd->bnd', alpha, self.E)           # [B,N,d_g]
        return target * (1.0 + g)


class VQGate(nn.Module):
    """Hard 1-of-K gate with straight-through estimator."""
    def __init__(
        self,
        K: int,
        d_in: int,
        d_g: int,
        tau: float = 20.0,
        share_codebook: Optional[nn.Parameter] = None,
        share_E: Optional[nn.Parameter] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.codebook = share_codebook if share_codebook is not None \
            else nn.Parameter(torch.randn(K, d_in) / math.sqrt(d_in))
        if share_E is not None:
            if share_E.shape != (K, d_g):
                raise ValueError(f"shared E has shape {tuple(share_E.shape)}; expected {(K, d_g)}")
            self.E = share_E
        else:
            self.E = nn.Parameter(torch.zeros(K, d_g))
            nn.init.zeros_(self.E)
        self.tau = tau
        self.normalize = normalize

    def forward(self, z, target):
        zf = F.normalize(z, dim=-1) if self.normalize else z
        Cf = F.normalize(self.codebook, dim=-1) if self.normalize else self.codebook
        logits = self.tau * torch.einsum('bnd,kd->bnk', zf, Cf)      # [B,N,K]
        soft = F.softmax(logits, dim=-1)
        hard_idx = torch.argmax(soft, dim=-1, keepdim=True)          # [B,N,1]
        hard = torch.zeros_like(soft).scatter_(-1, hard_idx, 1.0)
        alpha_st = (hard - soft).detach() + soft                     # STE
        g = torch.einsum('bnk,kd->bnd', alpha_st, self.E)            # [B,N,d_g]
        return target * (1.0 + g)


class MlpWithGateWidth(nn.Module):
    """Gate the MLP OUTPUT (dimension d)."""
    def __init__(self, mlp: nn.Module, gate: nn.Module):
        super().__init__()
        self.mlp = mlp
        self.gate = gate  # expects (z, y_out) -> y_out * (1+g)

    def forward(self, x):
        y = self.mlp(x)                # [B,N,d]
        return self.gate(x, y)


class MlpWithGateExpand(nn.Module):
    """Gate the MLP HIDDEN after fc1/act/drop (dimension d_ff ≈ 4d)."""
    def __init__(self, mlp: nn.Module, gate: nn.Module):
        super().__init__()
        self.mlp = mlp
        # Be robust to timm variants
        self.fc1 = getattr(mlp, 'fc1', getattr(mlp, 'linear1', None))
        self.fc2 = getattr(mlp, 'fc2', getattr(mlp, 'linear2', None))
        if self.fc1 is None or self.fc2 is None:
            raise RuntimeError("Unsupported MLP structure: missing fc1/fc2 (or linear1/linear2).")
        self.act = getattr(mlp, 'act', getattr(mlp, 'act_layer', nn.GELU()))
        self.drop1 = getattr(mlp, 'drop1', getattr(mlp, 'drop', nn.Identity()))
        self.drop2 = getattr(mlp, 'drop2', getattr(mlp, 'drop', nn.Identity()))
        self.gate = gate               # expects (z, h) -> h * (1+g), g ∈ R^{d_ff}
        # Expose hidden dim for budget inference post-wrapping
        self.gate_hidden_dim = self.fc1.out_features

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop1(h)              # [B,N,d_ff]
        h = self.gate(x, h)            # gate at hidden
        y = self.fc2(h)
        y = self.drop2(y)              # [B,N,d]
        return y


def _infer_hidden_dim_from_mlp(mlp: nn.Module, fallback: int) -> int:
    # works both before and after wrapping
    if hasattr(mlp, 'gate_hidden_dim'):
        return int(getattr(mlp, 'gate_hidden_dim'))
    for obj in (mlp, getattr(mlp, 'mlp', None)):
        if obj is None: continue
        for name in ('fc1', 'linear1'):
            if hasattr(obj, name) and hasattr(getattr(obj, name), 'out_features'):
                return int(getattr(obj, name).out_features)
    return int(fallback)


def attach_gates(
    vit: nn.Module,
    kind: str = 'soft',
    K: int = 512,
    d_g_mode: str = 'expand',  # 'expand' (4d) or 'width' (d)
    topk: Optional[int] = 8,
    tau: float = 10.0,
    share_codebook_across_depth: bool = True,
    share_E_across_depth: bool = False,
):
    """Attach a gate to every block's MLP with optional sharing across depth."""
    d_in = vit.embed_dim

    # optional shared codebook across depth
    codebook_shared: Optional[nn.Parameter] = None
    if share_codebook_across_depth:
        codebook_shared = nn.Parameter(torch.randn(K, d_in) / math.sqrt(d_in))
        vit.register_parameter('deepembed_codebook', codebook_shared)

    # optional shared E across depth (shape depends on d_g_mode)
    E_shared: Optional[nn.Parameter] = None
    if share_E_across_depth:
        d_g_example = d_in if d_g_mode == 'width' else _infer_hidden_dim_from_mlp(
            vit.blocks[0].mlp, fallback=4 * d_in
        )
        E_shared = nn.Parameter(torch.zeros(K, d_g_example))
        vit.register_parameter('deepembed_E', E_shared)

    # gentle guardrail: VQ ignores topk; warn once if user set it
    _warned_vq_topk = False

    for li, blk in enumerate(vit.blocks):
        # read hidden dim from original mlp (before wrapping)
        d_ff = _infer_hidden_dim_from_mlp(blk.mlp, fallback=4 * d_in)
        d_g = d_ff if d_g_mode == 'expand' else d_in

        # choose per-block or shared codebook
        cb_param = codebook_shared
        if cb_param is None:  # per-layer codebook
            cb_param = nn.Parameter(torch.randn(K, d_in) / math.sqrt(d_in))
            setattr(blk, f'deepembed_codebook_{li:02d}', cb_param)  # register under block

        if kind == 'soft':
            # Only Soft gate takes 'topk'
            gate = SoftCodebookGate(
                K, d_in, d_g,
                tau=tau,
                topk=topk if (topk is not None) else K,
                share_codebook=cb_param,
                share_E=E_shared
            )
        elif kind == 'vq':
            if (topk is not None) and not _warned_vq_topk:
                print("[CDE] Warning: 'topk' is ignored for VQ; using hard 1-of-K.", flush=True)
                _warned_vq_topk = True
            gate = VQGate(
                K, d_in, d_g,
                tau=max(12.0, tau),
                share_codebook=cb_param,
                share_E=E_shared
            )
        else:
            continue  # no gate

        if d_g_mode == 'expand':
            blk.mlp = MlpWithGateExpand(blk.mlp, gate)
        else:
            blk.mlp = MlpWithGateWidth(blk.mlp, gate)


# ----------------------- attention: RoPE (rotary) ------------------------

def _rope_freqs(head_dim: int, seq_len: int, theta: float, device, dtype):
    """Classic 1D RoPE cache. Returns cos/sin with shape [seq_len, head_dim]."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # [seq_len]
    freqs = torch.einsum('n,d->nd', t, inv_freq)                   # [seq_len, head_dim/2]
    cos = torch.cos(freqs).to(dtype).repeat_interleave(2, dim=-1)  # [seq_len, head_dim]
    sin = torch.sin(freqs).to(dtype).repeat_interleave(2, dim=-1)
    return cos, sin


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., head_dim] with even head_dim
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q, k: [B, H, N, Hd], cos/sin: [N, Hd]
    returns rotated q, k with broadcasting over batch & heads.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,N,Hd]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rope_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rope_rotate_half(k) * sin)
    return q_rot, k_rot


class AttentionWithRoPE(nn.Module):
    """
    wrap timm attention and apply 1d or 2d RoPE to q & k.
    if grid inference fails in 2d mode, we fall back to 1d once.
    """
    def __init__(self, attn: nn.Module, theta: float = 10000.0, kind: str = '1d',
                 grid_size: Optional[Tuple[int,int]] = None):
        super().__init__()
        self.attn = attn
        self.theta = float(theta)
        self.kind = kind if kind in ('1d', '2d') else '1d'
        # (H, W) of patch grid if known; used for 2d mapping
        self.grid_size = tuple(grid_size) if (grid_size is not None) else None
        self._warned_fallback = False
        # simple caches to avoid rebuilding cos/sin every forward
        self._cache_1d = {}   # key: (N, Hd, dtype, device) -> (cos, sin)
        self._cache_2d = {}   # key: (GH, GW, Hd_half, dtype, device) -> (cos_full, sin_full)

    def _get_1d_cos_sin(self, N, Hd, dtype, device):
        key = (int(N), int(Hd), dtype, device)
        if key not in self._cache_1d:
            cos, sin = _rope_freqs(Hd, N, self.theta, device, dtype)
            if N > 0:
                cos, sin = self._get_1d_cos_sin(N, Hd, dtype, device)
            self._cache_1d[key] = (cos, sin)
        return self._cache_1d[key]

    def _get_2d_cos_sin(self, GH, GW, N, Hd, dtype, device):
        Hd_half = Hd // 2
        key = (int(GH), int(GW), int(Hd_half), dtype, device)
        if key not in self._cache_2d:
            cos_w, sin_w = _rope_freqs(Hd_half, GW, self.theta, device, dtype)
            cos_h, sin_h = _rope_freqs(Hd_half, GH, self.theta, device, dtype)
            rr = torch.arange(GH, device=device).unsqueeze(1).expand(GH, GW).reshape(-1)
            cc = torch.arange(GW, device=device).unsqueeze(0).expand(GH, GW).reshape(-1)
            cos_full = torch.ones((N, Hd), device=device, dtype=dtype)
            sin_full = torch.zeros((N, Hd), device=device, dtype=dtype)
            cos_full[1:, :Hd_half] = cos_w[cc]
            cos_full[1:, Hd_half:] = cos_h[rr]
            sin_full[1:, :Hd_half] = sin_w[cc]
            sin_full[1:, Hd_half:] = sin_h[rr]
            self._cache_2d[key] = (cos_full, sin_full)
        return self._cache_2d[key]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        try:
            B, N, C = x.shape
            H = self.attn.num_heads
            Hd = C // H
            qkv = self.attn.qkv(x).reshape(B, N, 3, H, Hd).permute(2, 0, 3, 1, 4)  # 3,B,H,N,Hd
            q, k, v = qkv[0], qkv[1], qkv[2]                                        # B,H,N,Hd
            q = q * getattr(self.attn, 'scale', (Hd ** -0.5))

            device, dtype = q.device, q.dtype
            # build cos/sin per position
            if self.kind == '2d' and N > 1 and (Hd % 2 == 0):
                L = N - 1  # exclude cls
                GH = GW = None
                if self.grid_size is not None:
                    GH, GW = int(self.grid_size[0]), int(self.grid_size[1])
                if (GH is None) or (GW is None) or (GH * GW != L):
                    s = int(math.sqrt(L))
                    if s * s == L:
                        GH, GW = s, s
                if (GH is not None) and (GW is not None) and (GH * GW == L):
                    cos_full, sin_full = self._get_2d_cos_sin(GH, GW, N, Hd, dtype, device)
                    q, k = _apply_rope(q, k, cos_full, sin_full)
                else:
                    if not self._warned_fallback:
                        print("[rope] 2d grid inference failed; falling back to 1d.", flush=True)
                        self._warned_fallback = True
                    cos, sin = self._get_1d_cos_sin(N, Hd, dtype, device)
                    q, k = _apply_rope(q, k, cos, sin)
            else:
                cos, sin = _rope_freqs(Hd, N, self.theta, device, dtype)            # [N,Hd]
                if N > 0:
                    cos = cos.clone(); sin = sin.clone()
                    cos[0].fill_(1.0); sin[0].zero_()
                q, k = _apply_rope(q, k, cos, sin)

            # ---- SDPA / FlashAttention path (fast) ----
            drop_p = float(getattr(getattr(self.attn, 'attn_drop', None), 'p', 0.0)) if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                               dropout_p=drop_p, is_causal=False)
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.attn.proj(x)
            if hasattr(self.attn, 'proj_drop') and self.attn.proj_drop is not None:
                x = self.attn.proj_drop(x)
            return x
        except Exception:
            # fall back to the original module, forwarding any provided mask
            return self.attn(x, attn_mask=attn_mask)


def attach_rope(vit: nn.Module, theta: float = 10000.0, kind: str = '1d'):
    """replace each block.attn by an AttentionWithRoPE wrapper (passes patch grid)."""
    grid = None
    try:
        pe = getattr(vit, 'patch_embed', None)
        gs = getattr(pe, 'grid_size', None)
        if gs is not None:
            grid = (int(gs[0]), int(gs[1]))
    except Exception:
        grid = None
    for li, blk in enumerate(getattr(vit, 'blocks', [])):
        if hasattr(blk, 'attn') and isinstance(blk.attn, nn.Module):
            blk.attn = AttentionWithRoPE(blk.attn, theta=theta, kind=kind, grid_size=grid)


def disable_abs_pos_embed(vit: nn.Module, zero_weights: bool = True):
    """Disable learned absolute pos embed (recommended with RoPE)."""
    pe = getattr(vit, 'pos_embed', None)
    if isinstance(pe, torch.nn.Parameter):
        if zero_weights:
            with torch.no_grad():
                pe.data.zero_()
        pe.requires_grad_(False)


# ----------------------- accounting --------------------------------------

BASELINE_GFLOPS_DEIT_S_224 = 4.6  # reported (MAC-based)

@dataclass
class Budget:
    params_m: float
    est_gflops_total: float
    est_gflops_overhead: float


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_gate_overhead_gflops(
    K: int,
    d_in: int,
    d_g: int,
    N: int,
    L: int,
    k: int,
    assign_once: bool = True,
) -> float:
    # assignment cost: N*K*d_in (once or per-layer)
    assign = (N * K * d_in) * (1 if assign_once else L)
    # mixing cost (soft top-k): N*L*k*d_g; vq has k=1 but we still do one gather
    mix = N * L * k * d_g
    # apply gate (mul): N*L*d_g
    apply = N * L * d_g
    return (assign + mix + apply) / 1e9


def _infer_d_g(model: nn.Module, d_g_mode: str) -> int:
    if d_g_mode != 'expand':
        return int(model.embed_dim)
    try:
        mlp0 = model.blocks[0].mlp
        return _infer_hidden_dim_from_mlp(mlp0, fallback=4 * model.embed_dim)
    except Exception:
        return int(4 * model.embed_dim)


def compute_budget(
    model: nn.Module,
    gate: str,
    K: int,
    topk: Optional[int],
    d_g_mode: str,
    assign_once: bool,
    tokens: int = 197,
    share_codebook: bool = True,
) -> Budget:
    d_in = model.embed_dim
    L = len(model.blocks)
    d_g = _infer_d_g(model, d_g_mode)
    k = 1 if gate == 'vq' else (topk or K)
    eff_assign_once = assign_once and share_codebook
    overhead = 0.0 if gate == 'none' else estimate_gate_overhead_gflops(
        K, d_in, d_g, tokens, L, k, assign_once=eff_assign_once
    )
    total = BASELINE_GFLOPS_DEIT_S_224 + overhead
    return Budget(count_params_m(model), total, overhead)


# ----------------------- data --------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ImageNetClsLoc(Dataset):
    """
    ImageNet classification dataset over the Kaggle ILSVRC CLS-LOC layout.

    Train:
      <root>/ILSVRC/Data/CLS-LOC/train/<wnid>/*.JPEG   (folders define classes)

    Val:
      <root>/ILSVRC/Data/CLS-LOC/val/*.JPEG            (flat)
      <root>/ILSVRC/Annotations/CLS-LOC/val/<stem>.xml (labels: first object/name -> wnid)
    """
    IMG_EXTS = {".jpeg", ".jpg", ".JPEG", ".JPG"}

    def __init__(self, root: str, split: str, transform=None):
        super().__init__()
        self.transform = transform
        self.split = split.lower()
        ils = _resolve_ilsvrc_root(Path(root))

        self.data_dir = ils / "Data" / "CLS-LOC" / self.split
        self.ann_dir  = ils / "Annotations" / "CLS-LOC" / self.split

        if self.split == "train":
            # classes from folders
            class_dirs = [p for p in self.data_dir.iterdir() if p.is_dir()]
            self.classes = sorted([p.name for p in class_dirs])
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            samples: List[Tuple[str, int]] = []
            for cdir in class_dirs:
                idx = self.class_to_idx[cdir.name]
                for imgp in cdir.iterdir():
                    if imgp.suffix in self.IMG_EXTS:
                        samples.append((str(imgp), idx))
            self.samples = samples
        elif self.split == "val":
            # classes derived from TRAIN (stable ordering)
            train_dir = ils / "Data" / "CLS-LOC" / "train"
            class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
            self.classes = sorted([p.name for p in class_dirs])
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            imgs = sorted([p for p in self.data_dir.iterdir() if p.suffix in self.IMG_EXTS])
            samples = []
            miss_xml = 0
            miss_wnid = 0
            for i, imgp in enumerate(imgs, 1):
                stem = imgp.stem  # e.g., ILSVRC2012_val_00000001
                xmlp = self.ann_dir / f"{stem}.xml"
                if not xmlp.is_file():
                    miss_xml += 1
                    continue
                wnid = None
                try:
                    root_xml = ET.parse(str(xmlp)).getroot()
                    obj = root_xml.find("object")
                    if obj is not None:
                        wnid = obj.findtext("name")
                except Exception:
                    wnid = None
                if wnid is None or wnid not in self.class_to_idx:
                    miss_wnid += 1
                    continue
                samples.append((str(imgp), self.class_to_idx[wnid]))
            if (miss_xml + miss_wnid) > 0:
                print(f"[cls-loc:val] skipped images: missing_xml={miss_xml}, missing_or_unknown_wnid={miss_wnid}", flush=True)
            self.samples = samples
        else:
            raise ValueError("split must be 'train' or 'val'")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split='{self.split}' in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, target


def make_imagenet_loaders(
    root: str,
    bs: int,
    workers: Optional[int] = None,
    size: int = 224,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    layout: str = "folder",   # existing
    prefetch: int = 6,        # NEW
    drop_last: bool = False,  # NEW
    randaug_n: int = 2,
    randaug_m: int = 9,
    random_erase: float = 0.25,
):
    """Create loaders; on Windows, default to workers=0 to avoid spawn spam unless user overrides."""
    if workers is None:
        workers = 0 if os.name == "nt" else min(8, os.cpu_count() or 8)

    ra = transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m) if randaug_n > 0 else None
    re = transforms.RandomErasing(p=random_erase, value='random') if random_erase and random_erase > 0 else None
    train_ops = [
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if ra is not None:
        train_ops.append(ra)
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if re is not None:
        train_ops.append(re)
    train_tf = transforms.Compose(train_ops)

    val_tf = transforms.Compose([
        transforms.Resize(int(size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if layout == "folder":
        train_ds = datasets.ImageFolder(os.path.join(root, 'train'), train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(root, 'val'),   val_tf)
    elif layout == "cls-loc":
        train_ds = ImageNetClsLoc(root, split="train", transform=train_tf)
        val_ds   = ImageNetClsLoc(root, split="val",   transform=val_tf)
    else:
        raise ValueError(f"Unknown layout '{layout}'")

    if rank == 0:
        n_classes = len(getattr(train_ds, 'classes', [])) or (len(train_ds.classes) if hasattr(train_ds, 'classes') else 0)
        print(f"[data:{layout}] root={root} -> train={len(train_ds)}  val={len(val_ds)}  classes={n_classes}", flush=True)

    train_sampler = None
    val_sampler = None
    if ddp and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    pin = torch.cuda.is_available()
    persistent = workers > 0

    # Note: prefetch_factor is used only when workers > 0.
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=workers, pin_memory=pin,
        persistent_workers=persistent, prefetch_factor=prefetch,
        pin_memory_device=("cuda" if pin else ""), drop_last=drop_last
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        sampler=val_sampler, num_workers=workers, pin_memory=pin,
        persistent_workers=persistent, prefetch_factor=prefetch,
        pin_memory_device=("cuda" if pin else "")
    )

    # number of classes
    n_classes = len(train_ds.classes) if hasattr(train_ds, "classes") else 1000
    return train_loader, val_loader, n_classes


# ----------------------- training utils ----------------------------------

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch + 1
        if ep <= self.warmup_epochs:
            return [base_lr * ep / max(1, self.warmup_epochs) for base_lr in self.base_lrs]
        # cosine decay
        t = (ep - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return [base_lr * cos for base_lr in self.base_lrs]


def build_param_groups(model: nn.Module, weight_decay: float,
                       base_lr: Optional[float] = None, layer_decay: float = 1.0) -> List[Dict[str, Any]]:
    """AdamW param groups with (optional) LLRD; no weight decay for biases/LayerNorms."""
    if (base_lr is None) or (layer_decay >= 0.9999):
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    # layer-wise lr decay
    num_layers = (len(getattr(model, "blocks", [])) if hasattr(model, "blocks") else 0) + 2  # embed..blocks..head
    def _layer_id(name: str) -> int:
        if name.startswith("patch_embed") or ("pos_embed" in name) or ("cls_token" in name):
            return 0
        if name.startswith("blocks."):
            try:
                lid = int(name.split(".")[1])
            except Exception:
                lid = 0
            return lid + 1
        return num_layers - 1
    groups: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        lid = _layer_id(n)
        scale = layer_decay ** (num_layers - 1 - lid)
        lr = base_lr * scale
        wd = 0.0 if (n.endswith(".bias") or "norm" in n.lower()) else weight_decay
        key = (lr, wd)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": wd, "lr": lr}
        groups[key]["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def evaluate_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    tot = correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        tot += y.numel()
    return 100.0 * correct / max(1, tot)


# ----------------------- losses, aug, ema --------------------------------

def one_hot(y: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    assert 0.0 <= smoothing < 1.0
    with torch.no_grad():
        y = y.view(-1)
        off = smoothing / float(num_classes)
        on = 1.0 - smoothing + off
        out = torch.full((y.size(0), num_classes), off, device=y.device, dtype=torch.float32)
        out.scatter_(1, y.unsqueeze(1), on)
    return out


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [B,C], targets: [B,C] (probabilities)
    return (-targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _rand_bbox(w: int, h: int, lam: float):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    return x1, y1, x2, y2


def apply_mixup(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float):
    if alpha <= 0.0:  # no-op
        return x, one_hot(y, num_classes)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    y1 = one_hot(y, num_classes)
    y2 = y1[index]
    x_m = lam * x + (1.0 - lam) * x[index]
    t_m = lam * y1 + (1.0 - lam) * y2
    return x_m, t_m


def apply_cutmix(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float):
    if alpha <= 0.0:
        return x, one_hot(y, num_classes)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    y1 = one_hot(y, num_classes)
    y2 = y1[index]
    B, C, H, W = x.shape
    x1, y1b, x2, y2b = _rand_bbox(W, H, lam)
    x_m = x.clone()
    x_m[:, :, y1b:y2b, x1:x2] = x[index, :, y1b:y2b, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2b - y1b) / float(W * H))
    t_m = lam_adj * one_hot(y, num_classes) + (1.0 - lam_adj) * y2
    return x_m, t_m


class ModelEMA(nn.Module):
    """Exponential Moving Average of model parameters for evaluation stability."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=(1.0 - d))


# ----------------------- MLflow & checkpoints ----------------------------

class MLflowLogger:
    def __init__(self, enabled: bool, exp_name: str, run_name: str, tags: Dict[str, Any]):
        self.enabled = enabled and _HAVE_MLFLOW
        self.run = None
        if not self.enabled:
            return
        try:
            if exp_name:
                mlflow.set_experiment(exp_name)
            self.run = mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)
        except Exception:
            self.enabled = False
            self.run = None

    def child(self, run_name: str):
        if not self.enabled: return self
        try:
            mlflow.start_run(run_name=run_name, nested=True)
        except Exception:
            pass
        return self

    def end_child(self):
        if not self.enabled: return
        try: mlflow.end_run()
        except Exception: pass

    def log_params(self, params: Dict[str, Any]):
        if not self.enabled: return
        try: mlflow.log_params(params)
        except Exception: pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if not self.enabled: return
        try: mlflow.log_metrics(metrics, step=step)
        except Exception: pass

    def log_artifact(self, path: str):
        if not self.enabled: return
        try: mlflow.log_artifact(path)
        except Exception: pass

    def log_artifacts(self, dir_path: str):
        if not self.enabled: return
        try: mlflow.log_artifacts(dir_path)
        except Exception: pass

    def end(self):
        if not self.enabled or self.run is None: return
        try: mlflow.end_run()
        except Exception: pass


def save_checkpoint_pair(cfg_out: Path, state: Dict[str, Any], is_best: bool):
    ensure_dir(cfg_out)
    last_path = cfg_out / "ckpt_last.pt"
    torch.save(state, last_path)
    if is_best:
        best_path = cfg_out / "ckpt_best.pt"
        torch.save(state, best_path)
    return str(last_path), (str(cfg_out / "ckpt_best.pt") if is_best else None)


def load_checkpoint(path: str, model: nn.Module, opt=None, sched=None, scaler=None) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if opt and "opt" in ckpt: opt.load_state_dict(ckpt["opt"])
    if sched and "sched" in ckpt: sched.load_state_dict(ckpt["sched"])
    if scaler and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0)
    best = ckpt.get("best_acc1", float("-inf"))
    print(f"[resume] loaded {path} (epoch={start_epoch}, best={best:.2f})")
    return start_epoch, best


# ----------------------- experiment grid ---------------------------------

@dataclass
class ExpCfg:
    gate: str            # 'none' | 'vq' | 'soft'
    K: int               # codebook size
    topk: Optional[int]  # for soft; None for vq/none
    d_g_mode: str        # 'width' | 'expand'
    tau: float           # temperature
    assign_once: bool    # for est flops only
    # depth sharing toggles
    share_codebook: bool = True  # True: one codebook across depth; False: per-layer codebook
    share_E: bool = False        # True: one E across depth; False: per-layer E (default)


def build_model(cfg: ExpCfg, num_classes: int = 1000, model_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    if timm is None:
        raise RuntimeError("timm is required (pip install timm).")
    mk = model_kwargs or {}
    model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=num_classes, **mk)
    if cfg.gate != 'none':
        attach_gates(
            model, kind=cfg.gate, K=cfg.K, d_g_mode=cfg.d_g_mode,
            topk=cfg.topk, tau=cfg.tau,
            share_codebook_across_depth=cfg.share_codebook,
            share_E_across_depth=cfg.share_E,
        )
    return model


def cfg_hash(cfg: "ExpCfg") -> str:
    s = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def cfg_slug(cfg: "ExpCfg") -> str:
    topk = cfg.topk if cfg.topk is not None else "n"
    cb = "cbS" if getattr(cfg, "share_codebook", True) else "cbL"
    e  = "eS" if getattr(cfg, "share_E", False) else "eL"
    return f"{cfg.gate}-K{cfg.K}-k{topk}-{cfg.d_g_mode}-t{cfg.tau:g}-{cb}-{e}"


def cfg_dir(base: Path, idx: int, cfg: "ExpCfg") -> Path:
    return base / f"{idx:02d}_{cfg_slug(cfg)}_{cfg_hash(cfg)}"


def exp_row_dict(cfg: ExpCfg, budget: Budget, acc1: Optional[float], epochs: int,
                 cfg_rel_dir: str, last_rel: Optional[str], best_rel: Optional[str]) -> dict:
    return {
        'model': 'ViT-S/16',
        'gate': cfg.gate,
        'K': cfg.K if cfg.gate != 'none' else '-',
        'topk': cfg.topk if cfg.gate == 'soft' else '-',
        'd_g': cfg.d_g_mode,
        'assign_once': cfg.assign_once,
        'cb_shared': cfg.share_codebook,
        'E_shared': cfg.share_E,
        'params_M': round(budget.params_m, 3),
        'GFLOPs_total': round(budget.est_gflops_total, 3),
        'GFLOPs_overhead': round(budget.est_gflops_overhead, 3),
        'epochs': epochs,
        'top1_acc': None if acc1 is None else round(acc1, 2),
        'ckpt_dir': cfg_rel_dir,
        'ckpt_last': last_rel or '-',
        'ckpt_best': best_rel or '-',
    }


def default_grid() -> List[ExpCfg]:
    return [
        ExpCfg('none', 0, None, 'width', 0.0, True),
        ExpCfg('vq', 512, None, 'width', 20.0, True),
        ExpCfg('vq', 512, None, 'expand', 20.0, True),
        ExpCfg('soft', 512, 8, 'width', 10.0, True),
        ExpCfg('soft', 512, 8, 'expand', 10.0, True),
    ]


def load_grid(arg: str) -> List[ExpCfg]:
    """Accept 'default' | 'smoke' | path to JSON (list of dicts)."""
    PREDEF = {
        "smoke": [
            {"gate":"none","K":0,"topk":None,"d_g_mode":"width","tau":10.0,"assign_once":True},
            {"gate":"vq","K":512,"topk":None,"d_g_mode":"width","tau":20.0,"assign_once":True},
            {"gate":"soft","K":512,"topk":8,"d_g_mode":"width","tau":10.0,"assign_once":True},
        ]
    }
    if arg == 'default':
        data = [asdict(x) for x in default_grid()]
    elif os.path.isfile(arg):
        with open(arg, 'r') as f:
            data = json.load(f)
    elif arg in PREDEF:
        data = PREDEF[arg]
    else:
        raise FileNotFoundError(f"grid '{arg}' not found (file or predefined 'default'/'smoke').")

    norm: List[ExpCfg] = []
    for g in data:
        g = dict(g)
        # legacy alias support
        if "dg" in g and "d_g_mode" not in g:
            g["d_g_mode"] = g.pop("dg")
        if g.get("topk") in (0, "0"):
            g["topk"] = None
        g.setdefault("assign_once", True)
        # sharing defaults + aliases
        # aliases: codebook_per_layer / cb_per_layer -> share_codebook = not(...)
        if "share_codebook" not in g:
            per_layer_alias = g.pop("codebook_per_layer", None)
            per_layer_alias = g.pop("cb_per_layer", per_layer_alias)
            if per_layer_alias is not None:
                g["share_codebook"] = (not bool(per_layer_alias))
            else:
                g["share_codebook"] = True
        g.setdefault("share_E", False)
        norm.append(ExpCfg(**g))
    return norm


def write_csv_and_latex(rows: List[dict], out_dir: Path, fname: str = 'results'):
    if not rows: return
    ensure_dir(out_dir)
    csv_path = out_dir / f'{fname}.csv'
    tex_path = out_dir / f'{fname}.tex'

    # CSV
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # LaTeX
    cols = ['model','gate','K','topk','d_g','assign_once','cb_shared','E_shared',
            'params_M','GFLOPs_total','GFLOPs_overhead','epochs','top1_acc',
            'ckpt_dir','ckpt_last','ckpt_best']
    header = ' & '.join(cols) + ' \\\\'
    lines = [
        '% auto-generated table',
        '\\begin{tabular}{l l r r l l l l r r r r r l l l}',
        '\\toprule',
        header,
        '\\midrule',
    ]
    for r in rows:
        vals = [str(r[c]) for c in cols]
        lines.append(' & '.join(vals) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {csv_path}')
    print(f'wrote {tex_path}')


# ----------------------- SDPA backend compat shim ------------------------

# Prefer new torch.nn.attention.sdpa_kernel; fallback to legacy torch.backends.cuda.sdp_kernel
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_new
    from torch.nn.attention import SDPBackend as _SDPBackend

    def set_sdpa_backend(pref: str):
        # Map CLI choices to backend preference tuples
        if pref == 'flash':
            backends = [_SDPBackend.FLASH_ATTENTION, _SDPBackend.EFFICIENT_ATTENTION]
        elif pref == 'mem_efficient':
            backends = [_SDPBackend.EFFICIENT_ATTENTION,]
        elif pref == 'math':
            backends = [_SDPBackend.MATH,]
        else:  # auto
            backends = [_SDPBackend.FLASH_ATTENTION, _SDPBackend.EFFICIENT_ATTENTION, _SDPBackend.MATH]

        @contextmanager
        def _ctx():
            # New API accepts an iterable of backends
            with _sdpa_kernel_new(backends):
                yield
        return _ctx()

except Exception:
    # Legacy API
    try:
        from torch.backends.cuda import sdp_kernel as _sdpa_kernel_legacy

        def set_sdpa_backend(pref: str):
            if pref == 'flash':
                kw = dict(enable_flash=True,  enable_mem_efficient=True,  enable_math=False)
            elif pref == 'mem_efficient':
                kw = dict(enable_flash=False, enable_mem_efficient=True,  enable_math=False)
            elif pref == 'math':
                kw = dict(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            else:  # auto
                kw = dict(enable_flash=True,  enable_mem_efficient=True,  enable_math=True)

            @contextmanager
            def _ctx():
                with _sdpa_kernel_legacy(**kw):
                    yield
            return _ctx()
    except Exception:
        # No-op shim
        def set_sdpa_backend(pref: str):
            @contextmanager
            def _ctx():
                yield
            return _ctx()


# ----------------------- dtype helper ------------------------------------

def _dtype_from_str(name: str):
    name = name.lower()
    if name == 'bf16': return torch.bfloat16
    if name == 'fp16': return torch.float16
    return torch.float32


# ----------------------- main --------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagenet', type=str, required=True,
                    help='root for ImageNet. '
                         "If --imagenet_layout=folder, it must contain 'train/' and 'val/'. "
                         "If --imagenet_layout=cls-loc, it may be the parent containing 'ILSVRC/', the 'ILSVRC/' folder, or even '.../ILSVRC/Data/CLS-LOC'.")
    ap.add_argument('--imagenet_layout', type=str, default='folder', choices=['folder','cls-loc'],
                    help="Directory layout of the dataset root (default: folder).")
    ap.add_argument('--grid', type=str, default='default',
                    help="grid: 'default' | 'smoke' | path to JSON")
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')
    ap.add_argument('--workers', type=int, default=None,
                    help='num dataloader workers (default: 0 on Windows, 8 otherwise)')
    ap.add_argument('--prefetch', type=int, default=6,
                    help='DataLoader prefetch_factor')
    ap.add_argument('--drop_last', action='store_true',
                    help='drop last batch in train loader')
    ap.add_argument('--val_every', type=int, default=1,
                    help='validate every N epochs (1 = every epoch)')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--clip', type=float, default=1.0, help='grad-norm clip (0 disables)')
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--amp_dtype', type=str, default='bf16',
                    choices=['bf16', 'fp16', 'fp32'],
                    help='autocast dtype (Hopper: bf16 is recommended)')
    # data aug
    ap.add_argument('--randaug_n', type=int, default=2, help='RandAugment ops')
    ap.add_argument('--randaug_m', type=int, default=9, help='RandAugment magnitude')
    ap.add_argument('--random_erase', type=float, default=0.25, help='RandomErasing prob (0 to disable)')
    # mixup / cutmix
    ap.add_argument('--mixup', type=float, default=0.2, help='mixup alpha (0 to disable)')
    ap.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha (0 to disable)')
    ap.add_argument('--mix_prob', type=float, default=1.0, help='probability to apply mixup/cutmix to a batch')
    # model tweaks
    ap.add_argument('--drop_path', type=float, default=0.1, help='stochastic depth rate')
    ap.add_argument('--grad_ckpt', action='store_true', help='enable gradient checkpointing if supported')
    ap.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise LR decay (e.g., 0.75)')
    # rope / ema
    ap.add_argument('--rope', type=str, default='2d', choices=['none', '1d', '2d'], help='rotary pos. embedding')
    ap.add_argument('--rope_theta', type=float, default=10000.0, help='theta for RoPE frequency base')
    group_ema = ap.add_mutually_exclusive_group()
    group_ema.add_argument('--ema', dest='ema', action='store_true', help='use EMA of weights')
    group_ema.add_argument('--no_ema', dest='ema', action='store_false', help='disable EMA')
    ap.set_defaults(ema=True)
    ap.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay')
    ap.add_argument('--assign_once', action='store_true',
                    help='compute est flops with assignment once (budgets only)')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--out_dir', type=str, default=f'runs/exp_{now_utc()}')
    ap.add_argument('--skip_train', action='store_true',
                    help='only compute budgets, skip training')
    ap.add_argument('--save_every', type=int, default=1, help='save checkpoint every N epochs')
    ap.add_argument('--keep_snapshots', action='store_true',
                    help='also save ckpt_eXXX.pt per epoch (disk heavy)')
    ap.add_argument('--resume', type=str, default=None,
                    help='path/to/ckpt_last.pt to resume (manual)')
    ap.add_argument('--resume_auto', action='store_true',
                    help='auto resume per-config if its ckpt_last.pt exists')
    # DDP (optional; use torchrun)
    ap.add_argument('--ddp', action='store_true', help='enable DDP if launched with torchrun')
    ap.add_argument('--ddp_backend', type=str, default='nccl', help='nccl|gloo')
    # MLflow
    ap.add_argument('--mlflow', action='store_true', help='enable MLflow logging if installed')
    ap.add_argument('--mlflow_exp', type=str, default='CDE-ViT')
    ap.add_argument('--mlflow_run', type=str, default=None)
    # Performance toggles
    ap.add_argument('--fused_adamw', action='store_true',
                    help='use fused AdamW (PyTorch 2.x, CUDA only)')
    ap.add_argument('--compile_mode', type=str, default='none',
                    choices=['none', 'reduce-overhead', 'max-autotune'],
                    help='torch.compile mode')
    ap.add_argument('--sdp', type=str, default='flash',
                    choices=['auto', 'flash', 'mem_efficient', 'math'],
                    help='Scaled-Dot-Product-Attention backend preference')
    args = ap.parse_args()

    # rank/world detection
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_active = args.ddp and world_size > 1

    if ddp_active:
        torch.distributed.init_process_group(backend=args.ddp_backend)
        torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))

    # seed & device
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # i/o
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    (out_dir / "meta").mkdir(exist_ok=True, parents=True)

    # persist config
    if rank == 0:
        with open(out_dir / "meta" / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # dataset sanity
    assert_imagenet_root(args.imagenet, layout=args.imagenet_layout)

    # grid
    grid = load_grid(args.grid)

    # mlflow parent run
    run_name = args.mlflow_run or f"{Path(__file__).stem}-{now_utc()}"
    mlf = MLflowLogger(args.mlflow and (rank == 0), args.mlflow_exp, run_name,
                       tags={"host": os.uname().nodename if hasattr(os, "uname") else "win",
                             "grid": args.grid,
                             "imagenet_layout": args.imagenet_layout})

    rows: List[dict] = []

    # ---- SDPA backend context (new API or legacy fallback) ----
    backend_ctx = set_sdpa_backend(args.sdp)

    try:
        with backend_ctx:
            # data
            train_loader, val_loader, num_classes = make_imagenet_loaders(
                args.imagenet, args.bs, workers=args.workers, size=224,
                ddp=ddp_active, rank=rank, world_size=world_size,
                layout=args.imagenet_layout, prefetch=args.prefetch,
                drop_last=args.drop_last,
                randaug_n=args.randaug_n, randaug_m=args.randaug_m,
                random_erase=args.random_erase
            )

            if rank == 0:
                mlf.log_params({
                    "epochs": args.epochs, "warmup": args.warmup,
                    "bs": args.bs, "accum": args.accum, "workers": (0 if args.workers is None and os.name=="nt" else (args.workers or 8)),
                    "lr": args.lr, "wd": args.wd, "clip": args.clip, "label_smoothing": args.label_smoothing,
                    "amp": (not args.no_amp), "amp_dtype": args.amp_dtype,
                    "assign_once": args.assign_once, "num_classes": num_classes,
                    "ddp": ddp_active, "world_size": world_size,
                    "prefetch": args.prefetch, "drop_last": args.drop_last,
                    "compile_mode": args.compile_mode, "sdp": args.sdp, "fused_adamw": args.fused_adamw,
                    "val_every": args.val_every, "drop_path": args.drop_path,
                    "randaug_n": args.randaug_n, "randaug_m": args.randaug_m, "random_erase": args.random_erase,
                    "mixup": args.mixup, "cutmix": args.cutmix, "mix_prob": args.mix_prob,
                    "rope": args.rope, "rope_theta": args.rope_theta, "ema": args.ema,
                    "layer_decay": args.layer_decay
                })

            # loss
            ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)  # kept for non-soft-target path

            for j, cfg in enumerate(grid):
                cfg_out = cfg_dir(out_dir, j, cfg)
                if rank == 0:
                    ensure_dir(cfg_out)
                    # persist cfg/budget placeholders early
                    with open(cfg_out / "cfg.json", "w") as f: json.dump(asdict(cfg), f, indent=2)

                if rank == 0:
                    print(f'\n== experiment {j+1}/{len(grid)}: {cfg} => {cfg_out.name}\n', flush=True)

                # nested MLflow run per-config
                if rank == 0:
                    mlf.child(run_name=f"{j:02d}_{cfg_slug(cfg)}")

                # model
                model = build_model(cfg, num_classes=num_classes,
                                    model_kwargs={"drop_path_rate": args.drop_path})
                if args.rope != 'none':
                    attach_rope(model, theta=args.rope_theta, kind=args.rope)
                    # prefer RoPE-only: disable learned absolute pos embed unless user keeps it
                    disable_abs_pos_embed(model, zero_weights=True)
                model.to(device)
                if torch.cuda.is_available():
                    model.to(memory_format=torch.channels_last)

                # torch.compile (compile BEFORE DDP wrap)
                if args.compile_mode != 'none':
                    mode = 'reduce-overhead' if args.compile_mode == 'reduce-overhead' else 'max-autotune'
                    try:
                        model = torch.compile(model, mode=mode, fullgraph=False)
                    except Exception as _e:
                        if rank == 0:
                            print(f"[warn] torch.compile failed: {_e}. Continuing without compile.", flush=True)

                if args.grad_ckpt and hasattr(model, 'set_grad_checkpointing'):
                    try: model.set_grad_checkpointing()
                    except Exception: pass

                if ddp_active:
                    from torch.nn.parallel import DistributedDataParallel as DDP
                    model = DDP(model, device_ids=[rank % max(1, torch.cuda.device_count())],
                                output_device=rank % max(1, torch.cuda.device_count()),
                                broadcast_buffers=False)

                # budget (note: assign_once only valid when codebook is shared)
                budget = compute_budget(model.module if ddp_active else model,
                                        cfg.gate, cfg.K, cfg.topk, cfg.d_g_mode,
                                        cfg.assign_once, share_codebook=cfg.share_codebook)
                if rank == 0:
                    print(f'params(M)={budget.params_m:.3f}  '
                          f'est_total_GFLOPs={budget.est_gflops_total:.3f}  '
                          f'overhead={budget.est_gflops_overhead:.3f}', flush=True)
                    with open(cfg_out / "budget.json", "w") as f: json.dump(asdict(budget), f, indent=2)
                    mlf.log_params({
                        "cfg_gate": cfg.gate, "cfg_K": cfg.K, "cfg_topk": cfg.topk,
                        "cfg_d_g_mode": cfg.d_g_mode, "cfg_tau": cfg.tau, "cfg_assign_once": cfg.assign_once,
                        "cfg_cb_shared": cfg.share_codebook, "cfg_E_shared": cfg.share_E,
                        "params_M": round(budget.params_m, 3),
                        "GFLOPs_total": round(budget.est_gflops_total, 3),
                        "GFLOPs_overhead": round(budget.est_gflops_overhead, 3),
                    })

                # optim/sched/amp
                param_groups = build_param_groups(model if not ddp_active else model.module,
                                                  args.wd, base_lr=args.lr, layer_decay=args.layer_decay)
                opt = torch.optim.AdamW(
                    param_groups, lr=args.lr, betas=(0.9, 0.999),
                    fused=getattr(torch.optim.AdamW, 'fused', False) and args.fused_adamw
                )
                sched = WarmupCosine(opt, warmup_epochs=args.warmup, max_epochs=args.epochs)

                amp_dtype = _dtype_from_str(args.amp_dtype)
                use_fp16_scaler = (amp_dtype == torch.float16) and (not args.no_amp)
                try:
                    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16_scaler)
                except TypeError:
                    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
                # ema
                ema = None
                base_model_ref = (model.module if hasattr(model, "module") else model)
                if args.ema and not ddp_active:
                    ema = ModelEMA(base_model_ref, decay=args.ema_decay)

                # resume?
                start_epoch = 0
                best_acc1 = float("-inf")
                if args.resume:
                    start_epoch, best_acc1 = load_checkpoint(args.resume, model, opt, sched, scaler)
                elif args.resume_auto:
                    auto = cfg_out / "ckpt_last.pt"
                    if auto.is_file():
                        start_epoch, best_acc1 = load_checkpoint(str(auto), model, opt, sched, scaler)

                # training
                acc1 = None
                last_rel = None
                best_rel = None
                acc1_ema = None

                if not args.skip_train:
                    for ep in range(start_epoch, args.epochs):
                        if ddp_active and hasattr(train_loader.sampler, "set_epoch"):
                            train_loader.sampler.set_epoch(ep)

                        model.train()
                        t0 = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats(device)

                        tot_loss = 0.0
                        seen = 0
                        steps = len(train_loader)
                        accum = max(1, args.accum)
                        opt.zero_grad(set_to_none=True)

                        for i, (x, y) in enumerate(train_loader):
                            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                            y = y.to(device, non_blocking=True)
                            # soft targets path if any aug/smoothing enabled
                            use_soft = (args.mixup > 0.0) or (args.cutmix > 0.0) or (args.label_smoothing > 0.0)
                            # apply mixup/cutmix with probability
                            if use_soft and (random.random() < max(0.0, min(1.0, args.mix_prob))):
                                do_cutmix = (args.cutmix > 0.0) and (random.random() < 0.5)
                                if do_cutmix:
                                    x_aug, t = apply_cutmix(x, y, num_classes, args.cutmix)
                                else:
                                    x_aug, t = apply_mixup(x, y, num_classes, args.mixup if args.mixup > 0.0 else 0.2)
                            else:
                                x_aug, t = x, (one_hot(y, num_classes, smoothing=args.label_smoothing)
                                               if use_soft else None)
                            with torch.amp.autocast(device_type='cuda', enabled=(not args.no_amp), dtype=amp_dtype):
                                logits = model(x_aug)
                                if t is None:
                                    loss = ce(logits, y) / accum
                                else:
                                    loss = soft_cross_entropy(logits, t) / accum
                            if use_fp16_scaler:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()

                            if (i + 1) % accum == 0 or (i + 1) == steps:
                                if args.clip and args.clip > 0:
                                    if use_fp16_scaler:
                                        scaler.unscale_(opt)
                                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                                if use_fp16_scaler:
                                    scaler.step(opt)
                                    scaler.update()
                                else:
                                    opt.step()
                                if ema is not None:
                                    ema.update(model.module if hasattr(model, "module") else model)
                                opt.zero_grad(set_to_none=True)

                            tot_loss += loss.item() * accum * x.size(0)
                            seen += x.size(0)

                            # progress (rank 0)
                            if rank == 0 and (i + 1) % max(1, steps // 10) == 0:
                                print(f'ep {ep+1}/{args.epochs} it {i+1}/{steps} '
                                      f'loss {tot_loss/max(1,seen):.4f}', flush=True)

                        sched.step()
                        dt = time.time() - t0
                        imgs_per_s = seen / max(1e-6, dt)

                        # validate every N epochs (and on last)
                        do_val = ((ep + 1) % max(1, args.val_every) == 0) or (ep + 1 == args.epochs)
                        if do_val:
                            acc1 = evaluate_top1(model, val_loader, device)
                            acc1_ema = None
                            if ema is not None:
                                acc1_ema = evaluate_top1(ema.module, val_loader, device)
                        else:
                            acc1 = None
                            acc1_ema = None

                        if rank == 0:
                            max_mem = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
                            acc_str = f"{acc1:.2f}%" if acc1 is not None else "NA"
                            extra = f' ema_acc1={acc1_ema:.2f}%' if acc1_ema is not None else ""
                            print(f'epoch {ep+1}/{args.epochs} acc1={acc_str}{extra} '
                                  f'loss={tot_loss/max(1,seen):.4f} '
                                  f'time={dt:.1f}s ({imgs_per_s:.1f} img/s) '
                                  f'gpu_mem={sizeof_fmt(max_mem)}', flush=True)
                            log_dict = {
                                "loss": float(tot_loss/max(1,seen)),
                                "imgs_per_s": float(imgs_per_s)
                            }
                            if acc1 is not None:
                                log_dict["acc1"] = float(acc1)
                            if acc1_ema is not None:
                                log_dict["acc1_ema"] = float(acc1_ema)
                            mlf.log_metrics(log_dict, step=ep+1)

                            # checkpoints (per-config)
                            is_save_epoch = ((ep + 1) % max(1, args.save_every) == 0) or (ep + 1 == args.epochs)
                            if is_save_epoch:
                                # prefer ema for model selection if available
                                score = acc1_ema if (acc1_ema is not None) else acc1
                                is_best = False
                                if (score is not None) and (score >= best_acc1):
                                    best_acc1 = score
                                    is_best = True

                                base_state = {
                                    "epoch": ep + 1,
                                    "best_acc1": best_acc1 if best_acc1 != float("-inf") else None,
                                    "model": (model.module if hasattr(model, "module") else model).state_dict(),
                                    "opt": opt.state_dict(),
                                    "sched": sched.state_dict(),
                                    "scaler": scaler.state_dict(),
                                    "model_ema": (ema.module.state_dict() if ema is not None else None),
                                    "cfg": asdict(cfg),
                                    "args": vars(args)
                                }
                                last_path, new_best = save_checkpoint_pair(cfg_out, base_state, is_best=is_best)
                                last_rel = str(Path(last_path).relative_to(out_dir))
                                if new_best and is_best:
                                    best_rel = str(Path(new_best).relative_to(out_dir))
                                if args.keep_snapshots:
                                    snap = cfg_out / f"ckpt_e{ep+1:03d}_a{(acc1 if acc1 is not None else float('nan')):.2f}.pt"
                                    torch.save(base_state, snap)

                    # metrics.json per-config
                    if rank == 0:
                        with open(cfg_out / "metrics.json", "w") as f:
                            json.dump({
                                "acc1_best": best_acc1 if best_acc1 != float("-inf") else None
                            }, f, indent=2)

                # end of one config
                if rank == 0:
                    # in case of skip_train, derive rel paths if any existing
                    if last_rel is None and (cfg_out / "ckpt_last.pt").is_file():
                        last_rel = str((cfg_out / "ckpt_last.pt").relative_to(out_dir))
                    if best_rel is None and (cfg_out / "ckpt_best.pt").is_file():
                        best_rel = str((cfg_out / "ckpt_best.pt").relative_to(out_dir))

                    row = exp_row_dict(cfg, budget, best_acc1 if best_acc1 != float("-inf") else None,
                                       args.epochs if not args.skip_train else 0,
                                       str(cfg_out.relative_to(out_dir)), last_rel, best_rel)
                    rows.append(row)
                    write_csv_and_latex(rows, out_dir, fname='results')
                    # log artifacts (per-config dir)
                    mlf.log_artifacts(str(cfg_out))
                    mlf.end_child()

                # --- FREE GPU MEMORY BETWEEN CONFIGS ---
                try:
                    del model, opt, sched, scaler, ema
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    except KeyboardInterrupt:
        if rank == 0:
            print("\n[interrupt] saving partial results...", flush=True)
            write_csv_and_latex(rows, out_dir, fname='results')
    finally:
        if rank == 0:
            # final artifacts
            try:
                mlf.log_artifacts(str(out_dir))
            except Exception:
                pass
            mlf.end()
        if args.ddp and world_size > 1:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
