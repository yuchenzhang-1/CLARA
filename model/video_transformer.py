from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math

import torch
import torch.nn as nn


RAT_FIELDS = [
    "objective_summary",               
    "visual_description",              
    "textual_description",             
    "cross_modal_explanation",         
    "contextually_important_elements", 
    "overall_decision",                
    "reasons",                         
    "notes",                           
]
OBJ_IDXS = [0, 1, 2, 3, 4]
DEC_IDXS = [5, 6, 7]


# ---------------------------
# helpers
# ---------------------------
def _logit(p: float) -> float:

    eps = 1e-6
    p = float(p)
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    if x.dim() != 3 or mask.dim() != 2:
        raise ValueError(f"masked_mean_pool expects x [B,S,D], mask [B,S]; got {x.shape}, {mask.shape}")
    if x.size(0) != mask.size(0) or x.size(1) != mask.size(1):
        raise ValueError(f"masked_mean_pool shape mismatch: x {x.shape}, mask {mask.shape}")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be bool")

    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)  
    denom = mask_f.sum(dim=1).clamp_min(1.0)       
    return (x * mask_f).sum(dim=1) / denom


def select_rationale_tokens(
    rat: torch.Tensor, 
    rat_mask: torch.Tensor,
    mode: str
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:

    if mode == "none":
        return rat[:, :0, :], rat_mask[:, :0], []

    if mode == "obj":
        idxs = OBJ_IDXS
    elif mode == "dec":
        idxs = DEC_IDXS
    elif mode == "both":
        idxs = list(range(8))
    else:
        raise ValueError(f"Unknown rationale_mode: {mode}")

    rat_sel = rat[:, idxs, :]
    rat_sel_mask = rat_mask[:, idxs]
    return rat_sel, rat_sel_mask, idxs


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")

        pe = torch.zeros(max_len, d_model)  # [L,D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1,L,D]

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len, :]


class LearnablePositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pos[:, :seq_len, :]



@dataclass
class GatedVideoTransformerConfig:
    proj_out_dim: int
    raw_textual_emb_dim: int
    num_layers: int
    num_heads: int
    gvt_ffn_mult: int
    dropout: float

    max_seq_len_clips: int
    max_seq_len_rationale: int

 
    use_cls: bool
    use_mean_pool: bool
    mean_pool_include_rationale: bool

    rationale_mode_default: str  

  
    pos_encoding: str            

  
    gate_init_p: float          

    use_source_gate: bool
    use_rationale_token_gate: bool



class GatedVideoTransformer(nn.Module):
    def __init__(self, cfg: GatedVideoTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_out_dim = cfg.proj_out_dim
        self.raw_textual_emb_dim = cfg.raw_textual_emb_dim
        self.gvt_hidden_dim = 3 * cfg.proj_out_dim          
        self.gvt_ffn_dim = self.gvt_hidden_dim * cfg.gvt_ffn_mult

        D = self.gvt_hidden_dim

   
        if cfg.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        
        self.max_tokens = (1 if cfg.use_cls else 0) + cfg.max_seq_len_rationale + cfg.max_seq_len_clips

   
        pe_type = str(cfg.pos_encoding).lower()
        if pe_type == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(D, max_len=self.max_tokens)
        elif pe_type == "learnable":
            self.pos_enc = LearnablePositionalEmbedding(D, max_len=self.max_tokens)
        else:
            raise ValueError(f"Unknown pos_encoding: {cfg.pos_encoding}")

        self.pos_dropout = nn.Dropout(cfg.dropout)


        self.rationale_proj = nn.Linear(self.raw_textual_emb_dim,self.gvt_hidden_dim)

        self.rationale_gate = nn.Linear(D, 1, bias=True)         
        self.clip_gate = nn.Linear(D, 1, bias=True)              
        self.rationale_token_gate = nn.Linear(D, 1, bias=True)  


        init_b = _logit(cfg.gate_init_p)
        nn.init.zeros_(self.rationale_gate.weight)
        nn.init.zeros_(self.clip_gate.weight)
        nn.init.zeros_(self.rationale_token_gate.weight)
        nn.init.constant_(self.rationale_gate.bias, init_b)
        nn.init.constant_(self.clip_gate.bias, init_b)
        nn.init.constant_(self.rationale_token_gate.bias, init_b)


        if self.gvt_hidden_dim % cfg.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.gvt_hidden_dim}) must be divisible by num_heads ({cfg.num_heads}).")

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.num_heads,
            dim_feedforward=self.gvt_ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.final_norm = nn.LayerNorm(D)

    @staticmethod
    def _masked_mean_2d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean_pool(x, mask)

    def forward(
        self,
        clip_embs: torch.Tensor,                 
        clip_mask: torch.Tensor,                 
        rationale_embs: Optional[torch.Tensor],  
        rationale_mask: Optional[torch.Tensor] = None,  
        rationale_mode: Optional[str] = None,
        return_token_outputs: bool = False,
        return_gates: bool = True,
    ) -> Dict[str, Any]:
        if clip_embs.dim() != 3:
            raise ValueError(f"clip_embs must be [B,T,D], got {clip_embs.shape}")
        B, T, D = clip_embs.shape
        if D != self.gvt_hidden_dim:
            raise ValueError(f"clip_embs last dim {D} != hidden_dim {self.gvt_hidden_dim}")
        if clip_mask.shape != (B, T) or clip_mask.dtype != torch.bool:
            raise ValueError(f"clip_mask must be bool [B,T], got {clip_mask.shape}, {clip_mask.dtype}")

        device = clip_embs.device


        if T > self.cfg.max_seq_len_clips:
            clip_embs = clip_embs[:, : self.cfg.max_seq_len_clips, :]
            clip_mask = clip_mask[:, : self.cfg.max_seq_len_clips]
            T = self.cfg.max_seq_len_clips


        mode = rationale_mode if rationale_mode is not None else self.cfg.rationale_mode_default

        if rationale_embs is None:
            rat_sel = clip_embs[:, :0, :]  
            rat_sel_mask = clip_mask[:, :0]  
            sel_idxs: List[int] = []
        else:
            if rationale_embs.dim() != 3:
                raise ValueError(f"rationale_embs must be 3D [B,8,Dr], got {rationale_embs.shape}")
            if rationale_embs.size(0) != B or rationale_embs.size(1) != 8:
                raise ValueError(f"rationale_embs must be [B,8,Dr], got {rationale_embs.shape}")

            if rationale_mask is None:
                rationale_mask = torch.ones((B, 8), device=device, dtype=torch.bool)
            else:
                if rationale_mask.shape != (B, 8) or rationale_mask.dtype != torch.bool:
                    raise ValueError(f"rationale_mask must be bool [B,8], got {rationale_mask.shape}, {rationale_mask.dtype}")
                rationale_mask = rationale_mask.to(device=device)

  
            if rationale_embs.size(-1) != D:
                rationale_embs = self.rationale_proj(rationale_embs.to(device=device))
            else:
                rationale_embs = rationale_embs.to(device=device)

            rat_sel, rat_sel_mask, sel_idxs = select_rationale_tokens(rationale_embs, rationale_mask, mode)

            if rat_sel.size(1) > 0 and (rat_sel_mask.sum(dim=1) == 0).all():
                rat_sel = rat_sel[:, :0, :]
                rat_sel_mask = rat_sel_mask[:, :0]
                sel_idxs = []

        R = rat_sel.size(1) 


        source_gate_enabled = bool(self.cfg.use_source_gate)
        token_gate_enabled = bool(self.cfg.use_source_gate and self.cfg.use_rationale_token_gate)


        if not source_gate_enabled:
            gate_clips = torch.ones((B,), device=device, dtype=clip_embs.dtype)
            gate_rationale = torch.ones((B,), device=device, dtype=clip_embs.dtype)
            gate_rat_tokens = None
        else:
            clips_mean = self._masked_mean_2d(clip_embs, clip_mask)
            gate_clips = torch.sigmoid(self.clip_gate(clips_mean)).squeeze(-1)

            if R > 0:
                
                rat_mean = self._masked_mean_2d(rat_sel, rat_sel_mask)  
                gate_rationale = torch.sigmoid(self.rationale_gate(rat_mean)).squeeze(-1)

                if token_gate_enabled:
                    
                    gate_rat_tokens = torch.sigmoid(self.rationale_token_gate(rat_sel)).squeeze(-1)
                   
                    gate_rat_tokens = gate_rat_tokens * rat_sel_mask.to(dtype=gate_rat_tokens.dtype)
                else:
                    gate_rat_tokens = None
            else:
                gate_clips = torch.ones((B,), device=device, dtype=clip_embs.dtype)
                gate_rationale = torch.zeros((B,), device=device, dtype=clip_embs.dtype)
                gate_rat_tokens = None


        if source_gate_enabled:
            clip_tok = clip_embs * gate_clips.view(B, 1, 1)
        else:
            clip_tok = clip_embs

        if R > 0:
            rat_tok = rat_sel
            if source_gate_enabled:
                rat_tok = rat_tok * gate_rationale.view(B, 1, 1)
                if gate_rat_tokens is not None:
                    rat_tok = rat_tok * gate_rat_tokens.view(B, R, 1)

    
            rat_tok = rat_tok * rat_sel_mask.to(dtype=rat_tok.dtype).unsqueeze(-1)
        else:
            rat_tok = rat_sel  


        parts: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        if self.cfg.use_cls:
            cls = self.cls_token.expand(B, 1, D) 
            parts.append(cls)
            masks.append(torch.ones((B, 1), dtype=torch.bool, device=device))

        if R > 0:
            parts.append(rat_tok)
           
            masks.append(rat_sel_mask)

        parts.append(clip_tok)
        masks.append(clip_mask)

        x = torch.cat(parts, dim=1)          
        token_mask = torch.cat(masks, dim=1) 
        S = x.size(1)
        if S > self.max_tokens:
            raise ValueError(f"S={S} exceeds max_tokens={self.max_tokens} (check max lens / truncation).")


        x = x + self.pos_enc(seq_len=S)
        x = self.pos_dropout(x)

 
        key_padding_mask = ~token_mask
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  
        h = self.final_norm(h)


        if self.cfg.use_mean_pool:
            if self.cfg.use_cls:
                h_wo_cls = h[:, 1:, :]
                m_wo_cls = token_mask[:, 1:]
            else:
                h_wo_cls = h
                m_wo_cls = token_mask

            if (not self.cfg.mean_pool_include_rationale) and R > 0:
                h_pool = h_wo_cls[:, R:, :]
                m_pool = m_wo_cls[:, R:]
            else:
                h_pool = h_wo_cls
                m_pool = m_wo_cls

            video_emb = masked_mean_pool(h_pool, m_pool)
        else:
            if not self.cfg.use_cls:
                raise ValueError("use_mean_pool=False requires use_cls=True (CLS-based representation).")
            video_emb = h[:, 0, :]

        out: Dict[str, Any] = {
            "video_emb": video_emb,
            "token_mask": token_mask,
        }

        if return_token_outputs:
            out["token_outputs"] = h

        if return_gates:
            out["gates"] = {
                "source_gate_enabled": source_gate_enabled,
                "token_gate_enabled": token_gate_enabled,
                "gate_rationale": gate_rationale.detach(),
                "gate_clips": gate_clips.detach(),
                "gate_rat_tokens": None if gate_rat_tokens is None else gate_rat_tokens.detach(),
                "rationale_mode": mode,
                "selected_field_idxs": sel_idxs,
                "pos_encoding": self.cfg.pos_encoding,
                "rationale_tokens_valid": int(rat_sel_mask.sum().item()) if R > 0 else 0,
            }

        return out