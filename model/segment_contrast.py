from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F



def _clamp_segment(start: int, end: int, T: int) -> Tuple[int, int]:
  
    s = max(0, min(int(start), T))
    e = max(0, min(int(end), T))
    if e < s:
        e = s
    return s, e


def segment_mean_pool_2d(x: torch.Tensor, start: int, end: int) -> torch.Tensor:

    if x.dim() != 2:
        raise ValueError(f"segment_mean_pool_2d expects [T,D], got {tuple(x.shape)}")
    T, D = x.shape
    s, e = _clamp_segment(start, end, T)
    if e == s:
        return x.new_zeros((D,))
    return x[s:e].mean(dim=0)


def segment_mean_pool_3d(
    x: torch.Tensor,
    b: int,
    start: int,
    end: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if x.dim() != 3:
        raise ValueError(f"segment_mean_pool_3d expects [B,T,D], got {tuple(x.shape)}")
    B, T, D = x.shape
    if not (0 <= b < B):
        raise ValueError(f"b out of range: {b} (B={B})")

    s, e = _clamp_segment(start, end, T)
    if e == s:
        return x.new_zeros((D,))

    seg = x[b, s:e]  
    if mask is None:
        return seg.mean(dim=0)

    if mask.dim() != 2 or mask.shape[:2] != (B, T):
        raise ValueError(f"mask must be [B,T], got {tuple(mask.shape)}")

    seg_mask = mask[b, s:e] 
    if seg_mask.any():
        seg = seg[seg_mask]
        return seg.mean(dim=0)
    return x.new_zeros((D,))



def build_pair_vectors(
    seq_emb: Union[List[torch.Tensor], torch.Tensor],
    pairs: List[Dict[str, Any]],
    seq_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError("pairs must be a non-empty list")

    is_list = isinstance(seq_emb, list)
    if is_list:
        if len(seq_emb) == 0:
            raise ValueError("seq_emb list is empty")
        B = len(seq_emb)
        D = int(seq_emb[0].shape[-1])
        device = seq_emb[0].device
        dtype = seq_emb[0].dtype
    else:
        if not torch.is_tensor(seq_emb) or seq_emb.dim() != 3:
            raise ValueError(f"seq_emb must be list[tensor] or tensor[B,T,D], got {type(seq_emb)}")
        B, _, D = seq_emb.shape
        device = seq_emb.device
        dtype = seq_emb.dtype
        if seq_mask is not None and (seq_mask.shape[:2] != seq_emb.shape[:2]):
            raise ValueError("seq_mask shape must match seq_emb [B,T]")

    g_out: List[torch.Tensor] = []
    l_out: List[torch.Tensor] = []

    for p in pairs:
        bi = int(p["video_pos"])
        if not (0 <= bi < B):
            raise ValueError(f"pair.video_pos={bi} out of range (B={B})")

        gs, ge = p["global"]
        ls, le = p["local"]

        if is_list:
            x = seq_emb[bi]  
            g_vec = segment_mean_pool_2d(x, gs, ge)
            l_vec = segment_mean_pool_2d(x, ls, le)
        else:
            g_vec = segment_mean_pool_3d(seq_emb, bi, gs, ge, mask=seq_mask)
            l_vec = segment_mean_pool_3d(seq_emb, bi, ls, le, mask=seq_mask)

        g_out.append(g_vec)
        l_out.append(l_vec)

    g_vecs = torch.stack(g_out, dim=0).to(device=device, dtype=dtype)  
    l_vecs = torch.stack(l_out, dim=0).to(device=device, dtype=dtype) 
    return g_vecs, l_vecs




def info_nce_bidirectional(
    g_vecs: torch.Tensor,
    l_vecs: torch.Tensor,
    *,
    tau: float,
    normalize: bool,
    reduction: str,
) -> Dict[str, Any]:

    if g_vecs.dim() != 2 or l_vecs.dim() != 2:
        raise ValueError(f"g_vecs/l_vecs must be 2D, got {g_vecs.dim()} and {l_vecs.dim()}")
    if g_vecs.shape != l_vecs.shape:
        raise ValueError(f"shape mismatch: {g_vecs.shape} vs {l_vecs.shape}")
    P, _ = g_vecs.shape
    if P <= 0:
        raise ValueError("P must be > 0")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be mean/sum/ got {reduction}")

    if normalize:
        g = F.normalize(g_vecs, p=2, dim=-1)
        l = F.normalize(l_vecs, p=2, dim=-1)
    else:
        g, l = g_vecs, l_vecs

    logits = (g @ l.t()) / float(tau)  # [P, P]
    labels = torch.arange(P, device=logits.device, dtype=torch.long)

    loss_g2l = F.cross_entropy(logits, labels, reduction=reduction)
    loss_l2g = F.cross_entropy(logits.t(), labels, reduction=reduction)


    loss = 0.5 * (loss_g2l + loss_l2g)

    with torch.no_grad():
        pred_g2l = logits.argmax(dim=1)
        pred_l2g = logits.t().argmax(dim=1)
        acc_g2l = (pred_g2l == labels).float().mean().item()
        acc_l2g = (pred_l2g == labels).float().mean().item()

    return {
        "loss": loss,
        "loss_g2l": loss_g2l,
        "loss_l2g": loss_l2g,
        "logits": logits,
        "labels": labels,
        "acc_g2l": acc_g2l,
        "acc_l2g": acc_l2g,
    }




@dataclass
class ContrastiveConfig:
    tau: float
    normalize: bool
    reduction: str


class SegmentContrastive(nn.Module):


    def __init__(self, *, args: Optional[Any] = None, cfg: Optional[ContrastiveConfig] = None):
        super().__init__()
        if cfg is None:
            if args is None:
                raise ValueError(
                    "SegmentContrastive requires either cfg or args "
                    "(an object with contrast_tau/contrast_normalize/contrast_reduction)."
                )
            cfg = ContrastiveConfig.from_args(args)
        self.cfg = cfg

    def forward(self, g_vecs: torch.Tensor, l_vecs: torch.Tensor) -> Dict[str, Any]:
        
        return info_nce_bidirectional(
            g_vecs=g_vecs,
            l_vecs=l_vecs,
            tau=self.cfg.tau,
            normalize=self.cfg.normalize,
            reduction=self.cfg.reduction,
        )
        


def compute_contrastive_from_sequences(
    *,
    seq_emb: Union[List[torch.Tensor], torch.Tensor],
    pairs: List[Dict[str, Any]],
    seq_mask: Optional[torch.Tensor] = None,
    args: Optional[Any] = None,
    cfg: Optional[ContrastiveConfig] = None,
) -> Dict[str, Any]:

    g_vecs, l_vecs = build_pair_vectors(seq_emb=seq_emb, pairs=pairs, seq_mask=seq_mask)
    module = SegmentContrastive(args=args, cfg=cfg)
    out = module(g_vecs, l_vecs)
    out["g_vecs"] = g_vecs
    out["l_vecs"] = l_vecs
    return out