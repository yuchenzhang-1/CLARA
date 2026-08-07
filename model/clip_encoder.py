from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def masked_mean_2d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    if mask.dtype != torch.bool:
        mask = mask != 0
    m = mask.to(dtype=x.dtype, device=x.device)  
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0) 
    return (x * m.unsqueeze(-1)).sum(dim=1) / denom



class TwoLayerMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))





class FeatureProjector(nn.Module):

    def __init__(
        self,
        proj_out_dim: int,
        proj_hidden: int,
        raw_textual_emb_dim: int,
        proj_dropout: float,
        proj_pool_type: str, 
    ):
        super().__init__()
        self.proj_out_dim = int(proj_out_dim)

   
        self.audio_proj = TwoLayerMLP(
            d_in=1280,
            d_out=self.proj_out_dim,
            hidden=proj_hidden,
            dropout=proj_dropout,
        )

        self.visual_proj = TwoLayerMLP(
            d_in=768,
            d_out=self.proj_out_dim,
            hidden=proj_hidden,
            dropout=proj_dropout,
        )

   
        self.text_proj_transcript = TwoLayerMLP(
            d_in=raw_textual_emb_dim,
            d_out=self.proj_out_dim,
            hidden=proj_hidden,
            dropout=proj_dropout,
        )
        self.text_proj_ocr = TwoLayerMLP(
            d_in=raw_textual_emb_dim,
            d_out=self.proj_out_dim,
            hidden=proj_hidden,
            dropout=proj_dropout,
        )
        self.text_proj_both = TwoLayerMLP(
            d_in=raw_textual_emb_dim + raw_textual_emb_dim,
            d_out=self.proj_out_dim,
            hidden=proj_hidden,
            dropout=proj_dropout,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        whisper = batch["whisper"].to(device=device)  

        vit = batch["vit"].to(device=device)        
        vit_mask = batch.get("vit_mask", None)
        if vit_mask is None:
            vit_mask = torch.ones(vit.size(0), device=device, dtype=torch.bool)
        else:
            vit_mask = vit_mask.to(device=device).bool()

        text = batch["text"].to(device=device)     
        text_mask = batch.get("text_mask", None)
        if text_mask is None:
            text_mask = torch.ones(text.size(0), device=device, dtype=torch.bool)
        else:
            text_mask = text_mask.to(device=device).bool()

        ocr = batch["ocr"].to(device=device)       
        ocr_mask = batch.get("ocr_mask", None)
        if ocr_mask is None:
            ocr_mask = torch.ones(ocr.size(0), device=device, dtype=torch.bool)
        else:
            ocr_mask = ocr_mask.to(device=device).bool()

        
        audio_emb_proj = self.audio_proj(whisper)

        audio_mask = torch.ones(whisper.size(0), device=device, dtype=torch.bool)

        visual_emb_proj = torch.zeros(vit.size(0), self.proj_out_dim, device=device, dtype=audio_emb_proj.dtype)
        if vit_mask.any():
            visual_emb_proj[vit_mask] = self.visual_proj(vit[vit_mask])
        visual_mask = vit_mask

        text_emb_proj = torch.zeros(text.size(0), self.proj_out_dim, device=device, dtype=audio_emb_proj.dtype)

        both_ph = (~text_mask) & (~ocr_mask)
        only_text = text_mask & (~ocr_mask)
        only_ocr = (~text_mask) & ocr_mask
        both_real = text_mask & ocr_mask

        out_text_mask = (~both_ph).bool()

        if only_text.any():
            text_emb_proj[only_text] = self.text_proj_transcript(text[only_text])

        if only_ocr.any():
            text_emb_proj[only_ocr] = self.text_proj_ocr(ocr[only_ocr])

        if both_real.any():
            text_emb_proj[both_real] = self.text_proj_both(torch.cat([text[both_real], ocr[both_real]], dim=-1))

        return {
            "audio_emb_proj": audio_emb_proj,
            "text_emb_proj": text_emb_proj,
            "visual_emb_proj": visual_emb_proj,
            "audio_mask": audio_mask,
            "text_mask": out_text_mask,
            "visual_mask": visual_mask,
        }



@dataclass
class MoEAux:
    gate_probs: torch.Tensor
    topk_idx: torch.Tensor
    topk_probs: torch.Tensor
    alpha_plain: Optional[torch.Tensor] = None
    alpha_used: Optional[torch.Tensor] = None



class ExpertMLP(nn.Module):
    def __init__(self, fused_out_dim: int, moe_expert_ffn_mult: int, moe_dropout: float):
        super().__init__()
        ffn = fused_out_dim * moe_expert_ffn_mult
        self.fc1 = nn.Linear(fused_out_dim, ffn)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(moe_dropout)
        self.fc2 = nn.Linear(ffn, fused_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))



class MoEClipEncoder(nn.Module):
    def __init__(
        self,
        proj_out_dim: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_dropout: float,
        moe_use_mask_in_gating: bool,
        moe_fusion_type: str,          
        moe_prior_lambda: float,
        moe_expert_ffn_mult: int,
        moe_fast_dispatch: bool = True,
    ):
        super().__init__()
        self.proj_out_dim = int(proj_out_dim)
        self.fused_out_dim = 3 * self.proj_out_dim

        self.moe_num_experts = int(moe_num_experts)
        self.moe_top_k = int(moe_top_k)
        self.moe_use_mask_in_gating = bool(moe_use_mask_in_gating)
        self.moe_fusion_type = str(moe_fusion_type)
        self.moe_prior_lambda = float(moe_prior_lambda)
        self.moe_fast_dispatch = bool(moe_fast_dispatch)

        gate_in_dim = self.fused_out_dim + (3 if self.moe_use_mask_in_gating else 0)
        self.gate = nn.Linear(gate_in_dim, self.moe_num_experts)

        self.experts = nn.ModuleList([
            ExpertMLP(self.fused_out_dim, moe_expert_ffn_mult=moe_expert_ffn_mult, moe_dropout=moe_dropout)
            for _ in range(self.moe_num_experts)
        ])

        self.attn_query = nn.Parameter(torch.zeros(self.fused_out_dim))
        nn.init.normal_(self.attn_query, mean=0.0, std=0.02)

    def _build_fused(
        self,
        audio_emb_proj: torch.Tensor,
        text_emb_proj: torch.Tensor,
        visual_emb_proj: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        a_m = audio_mask.float().unsqueeze(-1)
        t_m = text_mask.float().unsqueeze(-1)
        v_m = visual_mask.float().unsqueeze(-1)

        a = audio_emb_proj * a_m
        t = text_emb_proj * t_m
        v = visual_emb_proj * v_m

        fused = torch.cat([v, t, a], dim=-1)

        if self.moe_use_mask_in_gating:
            masks = torch.stack([v_m.squeeze(-1), t_m.squeeze(-1), a_m.squeeze(-1)], dim=-1)
            gate_in = torch.cat([fused, masks], dim=-1)
        else:
            gate_in = fused
        return fused, gate_in

    def _dispatch_experts(
        self,
        fused_rep: torch.Tensor,
        flat_idx: torch.Tensor,
    ) -> torch.Tensor:
        BK, D = fused_rep.shape
        out_flat = torch.zeros((BK, D), device=fused_rep.device, dtype=fused_rep.dtype)

        if not self.moe_fast_dispatch:
            for e in range(self.moe_num_experts):
                sel = (flat_idx == e)
                if sel.any():
                    out_flat[sel] = self.experts[e](fused_rep[sel])
            return out_flat

        order = torch.argsort(flat_idx)
        flat_idx_s = flat_idx.index_select(0, order)
        fused_s = fused_rep.index_select(0, order)

        counts = torch.bincount(flat_idx_s, minlength=self.moe_num_experts).tolist()

        start = 0
        out_s = fused_s.new_zeros((BK, D))
        for e, c in enumerate(counts):
            if c <= 0:
                continue
            end = start + c
            out_s[start:end] = self.experts[e](fused_s[start:end])
            start = end

        inv = torch.empty_like(order)
        inv[order] = torch.arange(BK, device=order.device)
        return out_s.index_select(0, inv)

    def forward(
        self,
        audio_emb_proj: torch.Tensor,
        text_emb_proj: torch.Tensor,
        visual_emb_proj: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
        rationale: Optional[torch.Tensor] = None,
        rationale_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, MoEAux]:

        B = audio_emb_proj.size(0)

        fused, gate_in = self._build_fused(
            audio_emb_proj, text_emb_proj, visual_emb_proj,
            audio_mask, text_mask, visual_mask
        )

        gate_logits = self.gate(gate_in)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(gate_probs, k=self.moe_top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        BK = B * self.moe_top_k
        flat_idx = topk_idx.reshape(-1)
        fused_rep = fused.unsqueeze(1).expand(B, self.moe_top_k, self.fused_out_dim).reshape(BK, self.fused_out_dim)

        out_flat = self._dispatch_experts(fused_rep=fused_rep, flat_idx=flat_idx)
        topk_out = out_flat.view(B, self.moe_top_k, self.fused_out_dim)

        importance = gate_probs.mean(dim=0)
        load = torch.bincount(flat_idx, minlength=self.moe_num_experts).float() / float(BK)
        lb_loss = self.moe_num_experts * torch.sum(importance * load)

        alpha_plain = None
        alpha_used = None

        if self.moe_fusion_type == "weighted_sum":
            clip_emb_final = (topk_probs.unsqueeze(-1) * topk_out).sum(dim=1)
        else:
            s_base = torch.einsum("bkd,d->bk", topk_out, self.attn_query) / math.sqrt(self.fused_out_dim)
            alpha_plain = F.softmax(s_base, dim=-1)
            s_used = s_base + self.moe_prior_lambda * torch.log(topk_probs.clamp(min=1e-12))
            alpha_used = F.softmax(s_used, dim=-1)
            clip_emb_final = (alpha_used.unsqueeze(-1) * topk_out).sum(dim=1)

        aux = MoEAux(
            gate_probs=gate_probs,
            topk_idx=topk_idx,
            topk_probs=topk_probs,
            alpha_plain=alpha_plain,
            alpha_used=alpha_used,
        )
        return clip_emb_final, lb_loss, aux




class MLPEncoder(nn.Module):

    def __init__(
        self,
        proj_out_dim: int,
        clipencoder_mlp_ffn_mult: int,
        clipencoder_mlp_dropout: float,
    ):
        super().__init__()
        self.proj_out_dim = int(proj_out_dim)
        self.fused_out_dim = 3 * self.proj_out_dim

        hidden = int(self.fused_out_dim * clipencoder_mlp_ffn_mult)
        self.ln = nn.LayerNorm(self.fused_out_dim)
        self.fc1 = nn.Linear(self.fused_out_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(float(clipencoder_mlp_dropout))
        self.fc2 = nn.Linear(hidden, self.fused_out_dim)

    def forward(
        self,
        audio_emb_proj: torch.Tensor,
        text_emb_proj: torch.Tensor,
        visual_emb_proj: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
        rationale: Optional[torch.Tensor] = None,
        rationale_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:

        fused = torch.cat(
            [
                visual_emb_proj * visual_mask.float().unsqueeze(-1),
                text_emb_proj * text_mask.float().unsqueeze(-1),
                audio_emb_proj * audio_mask.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        y = self.fc2(self.drop(self.act(self.fc1(self.ln(fused)))))
        clip_emb_final = fused + self.drop(y)

        lb_loss = clip_emb_final.new_zeros(())
        aux = MoEAux(
            gate_probs=clip_emb_final.new_zeros((clip_emb_final.size(0), 1)),
            topk_idx=torch.zeros((clip_emb_final.size(0), 1), device=clip_emb_final.device, dtype=torch.long),
            topk_probs=clip_emb_final.new_zeros((clip_emb_final.size(0), 1)),
        )
        return clip_emb_final, lb_loss, aux




class ClipEncoder(nn.Module):

    def __init__(self, projector: FeatureProjector, backend: nn.Module):
        super().__init__()
        self.projector = projector
        self.backend = backend

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        proj = self.projector(batch)

        clip_emb_final, lb_loss, aux = self.backend(
            audio_emb_proj=proj["audio_emb_proj"],
            text_emb_proj=proj["text_emb_proj"],
            visual_emb_proj=proj["visual_emb_proj"],
            audio_mask=proj["audio_mask"].bool(),
            text_mask=proj["text_mask"].bool(),
            visual_mask=proj["visual_mask"].bool(),
            rationale=batch.get("rationale", None),
            rationale_mask=batch.get("rationale_mask", None),
        )

        return {
            "clip_emb_final": clip_emb_final,
            "lb_loss": lb_loss,
            "aux": aux,
            "mb_size": torch.tensor(int(clip_emb_final.size(0)), device=clip_emb_final.device),
            "modality_masks": {
                "audio_mask": proj["audio_mask"].bool(),
                "text_mask": proj["text_mask"].bool(),
                "visual_mask": proj["visual_mask"].bool(),
            },
        }