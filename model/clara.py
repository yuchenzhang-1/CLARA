from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.video_transformer import (
    GatedVideoTransformer,
    select_rationale_tokens,
    masked_mean_pool,
)



@dataclass
class CLARAConfig:
    proj_out_dim: int
    raw_textual_emb_dim: int
    use_transformer: bool
    use_contrastive: bool

    rationale_mode: str                 
    mean_pool_include_rationale: bool

    num_classes: int
    clf_dropout: float



def build_padded_clip_sequence(per_video_clip_embs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    per_video_clip_embs: list of [Ti,D], Ti>0
    return:
      clip_embs_padded: [B,Tmax,D]
      clip_mask: [B,Tmax] bool
    """
    B = len(per_video_clip_embs)
    device = per_video_clip_embs[0].device
    dtype = per_video_clip_embs[0].dtype
    D = int(per_video_clip_embs[0].size(-1))

    lens = [int(x.size(0)) for x in per_video_clip_embs]
    Tmax = max(lens)

    out = torch.zeros((B, Tmax, D), device=device, dtype=dtype)
    mask = torch.zeros((B, Tmax), device=device, dtype=torch.bool)

    for i, x in enumerate(per_video_clip_embs):
        L = int(x.size(0))
        out[i, :L] = x
        mask[i, :L] = True

    return out, mask


def segment_mean_pool(seq: torch.Tensor, start: int, end: int) -> torch.Tensor:
    start = int(start)
    end = int(end)
    if end <= start:
        return torch.zeros((seq.size(-1),), device=seq.device, dtype=seq.dtype)
    return seq[start:end].mean(dim=0)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class CLARA(nn.Module):
    def __init__(
        self,
        cfg: CLARAConfig,
        clip_encoder: nn.Module,
        transformer: Optional[GatedVideoTransformer],
    ):
        super().__init__()
        self.cfg = cfg
        self.clip_encoder = clip_encoder
        self.transformer = transformer

        self.clf_in_dim = 3 * int(cfg.proj_out_dim)
        self.clf_hidden = int(cfg.proj_out_dim)

        self.raw_textual_emb_dim = int(cfg.raw_textual_emb_dim)
        self.rationale_proj: Optional[nn.Linear] = None

        self.classifier = MLPClassifier(
            in_dim=self.clf_in_dim,
            hidden=self.clf_hidden,
            out_dim=int(cfg.num_classes),
            dropout=float(cfg.clf_dropout),
        )



    def _build_microbatch_input(
        self,
        batch: Dict[str, Any],
        *,
        bi: int,
        t0: int,
        t1: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        whisper = batch["whisper"][bi][t0:t1].to(device, non_blocking=True) 
        vit = batch["vit"][bi][t0:t1].to(device, non_blocking=True)         
        vit_mask = batch["vit_mask"][bi][t0:t1].to(device, non_blocking=True)

        ocr = batch["ocr"][bi][t0:t1].to(device, non_blocking=True)         
        ocr_mask = batch["ocr_mask"][bi][t0:t1].to(device, non_blocking=True)

        text = batch["text"][bi][t0:t1].to(device, non_blocking=True)      
        text_mask = batch["text_mask"][bi][t0:t1].to(device, non_blocking=True)

        out: Dict[str, Any] = {
            "whisper": whisper,
            "vit": vit,
            "vit_mask": vit_mask,
            "ocr": ocr,
            "ocr_mask": ocr_mask,
            "text": text,
            "text_mask": text_mask,
        }

  
        rationale = batch.get("rationale", None)
        if rationale is not None:
            rat = rationale[bi].to(device, non_blocking=True)  
            n = int(whisper.size(0))
            out["rationale"] = rat.unsqueeze(0).expand(n, -1, -1).contiguous() 

        rationale_mask = batch.get("rationale_mask", None)
        if rationale_mask is not None:
            rm = rationale_mask[bi].to(device, non_blocking=True)  
            n = int(whisper.size(0))
            out["rationale_mask"] = rm.unsqueeze(0).expand(n, -1).contiguous() 

        return out



    def encode_clips_by_video_microbatches(
        self,
        batch: Dict[str, Any],
        microbatches: List[Tuple[int, int, int]],
        T_list: List[int],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any], Optional[torch.Tensor]]:

        B = len(T_list)


        microbatches = sorted(microbatches, key=lambda x: (int(x[0]), int(x[1])))

        per_video_chunks: List[List[torch.Tensor]] = [[] for _ in range(B)]

        aux_list: List[Any] = []
        modality_masks_list: List[Any] = []

        lb_losses: List[torch.Tensor] = []
        lb_sizes: List[torch.Tensor] = []

        for (video_pos, t0, t1) in microbatches:
            bi = int(video_pos)
            t0 = int(t0)
            t1 = int(t1)

            mb_in = self._build_microbatch_input(batch, bi=bi, t0=t0, t1=t1, device=device)

            out = self.clip_encoder(mb_in)
            emb = out["clip_emb_final"]  
            per_video_chunks[bi].append(emb)

            aux_list.append(out.get("aux", None))
            modality_masks_list.append(out.get("modality_masks", None))

            lb = out.get("lb_loss", None)
            if torch.is_tensor(lb):
                mb_n = emb.new_tensor(int(emb.size(0)))
                lb_losses.append(lb.reshape(()))
                lb_sizes.append(mb_n.reshape(()).to(dtype=lb.dtype))

        per_video_seq: List[torch.Tensor] = []
        for bi in range(B):
            seq = torch.cat(per_video_chunks[bi], dim=0)
            if int(seq.size(0)) != int(T_list[bi]):
                raise ValueError(
                    f"microbatches do not cover full video: bi={bi}, got {int(seq.size(0))}, expect {int(T_list[bi])}"
                )
            per_video_seq.append(seq)

        lb_loss = None
        if len(lb_losses) > 0:
            lbs = torch.stack(lb_losses, dim=0)
            ns = torch.stack(lb_sizes, dim=0).clamp(min=1.0)
            lb_loss = (lbs * ns).sum() / ns.sum()

        aux_info = {"aux_list": aux_list, "modality_masks_list": modality_masks_list}
        return per_video_seq, aux_info, lb_loss



    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.parameters()).device

        labels = batch.get("labels", None)
        microbatches: List[Tuple[int, int, int]] = batch["clip_microbatches"]
        T_list: List[int] = batch["T_list"]

        per_video_seq, clip_aux_info, clip_lb_loss = self.encode_clips_by_video_microbatches(
            batch=batch,
            microbatches=microbatches,
            T_list=T_list,
            device=device,
        )

        clip_seq_padded, clip_mask = build_padded_clip_sequence(per_video_seq)  

   
        rationale_embs = batch.get("rationale", None)
        rationale_mask = batch.get("rationale_mask", None)

        if rationale_embs is not None:
            rationale_embs = rationale_embs.to(device, non_blocking=True)
            if rationale_mask is None:
                rationale_mask = torch.ones((rationale_embs.size(0), 8), device=device, dtype=torch.bool)
            else:
                rationale_mask = rationale_mask.to(device, non_blocking=True)

            rat_sel, rat_sel_mask, _ = select_rationale_tokens(rationale_embs, rationale_mask, self.cfg.rationale_mode)
        else:
            rat_sel, rat_sel_mask = None, None

        if self.cfg.use_transformer:
            if self.transformer is None:
                raise ValueError("cfg.use_transformer=True but transformer is None.")

            tf_out = self.transformer(
                clip_embs=clip_seq_padded,
                clip_mask=clip_mask,
                rationale_embs=rationale_embs,
                rationale_mask=rationale_mask,          
                rationale_mode=self.cfg.rationale_mode,
                return_token_outputs=False,
                return_gates=True,
            )
            video_emb = tf_out["video_emb"]
            tf_gate = tf_out.get("gates", None)
        else:
            tokens: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []

            clip_dim = int(clip_seq_padded.size(-1))

            if rat_sel is not None and rat_sel.size(1) > 0 and self.cfg.mean_pool_include_rationale:
                if int(rat_sel.size(-1)) != clip_dim:
                    if self.rationale_proj is None:
                        self.rationale_proj = nn.Linear(self.raw_textual_emb_dim, clip_dim, bias=True).to(device)
                    rat_sel = self.rationale_proj(rat_sel)

                tokens.append(rat_sel)
                masks.append(rat_sel_mask)   

            tokens.append(clip_seq_padded)
            masks.append(clip_mask)

            x = torch.cat(tokens, dim=1)
            m = torch.cat(masks, dim=1)
            video_emb = masked_mean_pool(x, m)
            tf_gate = None

        logits = self.classifier(video_emb)


        contrastive_out = None
        if self.cfg.use_contrastive:
            pairs = batch["contrastive"]["pairs"]
            g_list: List[torch.Tensor] = []
            l_list: List[torch.Tensor] = []

            for p in pairs:
                bi = int(p["video_pos"])
                gs, ge = p["global"]
                ls, le = p["local"]
                seq_i = per_video_seq[bi] 
                g_list.append(segment_mean_pool(seq_i, gs, ge))
                l_list.append(segment_mean_pool(seq_i, ls, le))

            if len(g_list) > 0:
                g_vecs = torch.stack(g_list, dim=0)
                l_vecs = torch.stack(l_list, dim=0)
            else:
                g_vecs = torch.empty((0, self.clf_in_dim), device=device, dtype=video_emb.dtype)
                l_vecs = torch.empty((0, self.clf_in_dim), device=device, dtype=video_emb.dtype)

            contrastive_out = {"g_vecs": g_vecs, "l_vecs": l_vecs}

        return {
            "logits": logits,
            "labels": labels,
            "transformer_gates": tf_gate,
            "contrastive_out": contrastive_out,
            "clip_aux_info": clip_aux_info,
            "lb_loss": clip_lb_loss,
        }