from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch

from utils.segment_sampling import (
    SegmentSamplingConfig,
    build_video_rng,
    sample_pairs_for_video,
)


def make_microbatch_slices(T: int, microbatch_size: int) -> List[Tuple[int, int]]:

    mb = int(microbatch_size)
    if mb <= 0:
        return [(0, int(T))]
    return [(s, min(int(T), s + mb)) for s in range(0, int(T), mb)]


class CLARACollator:

    def __init__(self, data_args, *, seg_cfg: SegmentSamplingConfig, rank: int = 0):
        self.clip_microbatch_size = int(data_args.clip_microbatch_size)
        self.seg_cfg = seg_cfg
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(samples)
        if B == 0:
            raise ValueError("Empty batch: samples list is empty.")

  
        required = [
            "video_id", "label", "T",
            "whisper", "vit", "vit_mask",
            "ocr", "ocr_mask",
            "text", "text_mask",
            "rationale", "rationale_mask",
        ]
        for i, s in enumerate(samples):
            miss = [k for k in required if k not in s]
            if miss:
                raise KeyError(f"Sample[{i}] missing keys: {miss}. video_id={s.get('video_id','N/A')}")

        video_ids = [str(s["video_id"]) for s in samples]
        labels = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long)

    
        whisper_list = [s["whisper"] for s in samples]       
        vit_list = [s["vit"] for s in samples]              
        vit_mask_list = [s["vit_mask"].to(torch.bool) for s in samples]     
        ocr_list = [s["ocr"] for s in samples]             
        ocr_mask_list = [s["ocr_mask"].to(torch.bool) for s in samples]    
        text_list = [s["text"] for s in samples]            
        text_mask_list = [s["text_mask"].to(torch.bool) for s in samples]  
        rationale = torch.stack([s["rationale"] for s in samples], dim=0)
        rationale_mask = torch.stack([s["rationale_mask"].to(torch.bool) for s in samples], dim=0)

  
        if rationale.dim() != 3 or rationale.size(1) != 8:
            raise ValueError(f"rationale must be [B,8,D], got shape={tuple(rationale.shape)}")
        if rationale_mask.dim() != 2 or rationale_mask.size(1) != 8:
            raise ValueError(f"rationale_mask must be [B,8], got shape={tuple(rationale_mask.shape)}")


        T_list: List[int] = []
        video_offsets: List[Tuple[int, int]] = []
        cursor = 0

        for si, s in enumerate(samples):
            T = int(s["T"])
            if T <= 0:
                raise ValueError(f"T must be > 0, got {T} for video_id={s['video_id']}")

            def _len(x: torch.Tensor) -> int:
                return int(x.size(0))

            if _len(s["whisper"]) != T or _len(s["vit"]) != T or _len(s["vit_mask"]) != T \
               or _len(s["ocr"]) != T or _len(s["ocr_mask"]) != T \
               or _len(s["text"]) != T or _len(s["text_mask"]) != T:
                raise ValueError(
                    f"Length mismatch in sample[{si}] video_id={s['video_id']}: "
                    f"T={T}, lens={{"
                    f"whisper:{_len(s['whisper'])}, vit:{_len(s['vit'])}, vit_mask:{_len(s['vit_mask'])}, "
                    f"ocr:{_len(s['ocr'])}, ocr_mask:{_len(s['ocr_mask'])}, "
                    f"text:{_len(s['text'])}, text_mask:{_len(s['text_mask'])}"
                    f"}}"
                )

            T_list.append(T)
            start = cursor
            cursor += T
            end = cursor
            video_offsets.append((start, end))


        clip_microbatches: List[Tuple[int, int, int]] = []
        for bi, T in enumerate(T_list):
            for (t0, t1) in make_microbatch_slices(T, self.clip_microbatch_size):
                clip_microbatches.append((int(bi), int(t0), int(t1)))

     
        pairs_out: List[Dict[str, Any]] = []
        for bi, s in enumerate(samples):
            T = T_list[bi]
            vid = s["video_id"]

            video_rng = build_video_rng(
                base_seed=self.seg_cfg.base_seed,
                epoch=self.epoch,
                rank=self.rank,
                video_id=str(vid),
            )
            pairs = sample_pairs_for_video(
                T=T,
                num_pairs_per_video=self.seg_cfg.num_pairs_per_video,
                mode=self.seg_cfg.mode,
                l_ratio=self.seg_cfg.l_ratio,
                g_ratio=self.seg_cfg.g_ratio,
                min_l=self.seg_cfg.min_l,
                min_g=self.seg_cfg.min_g,
                max_l=self.seg_cfg.max_l,
                max_g=self.seg_cfg.max_g,
                rng=video_rng,
            )

            for pi, p in enumerate(pairs):
                pairs_out.append({
                    "video_id": str(vid),
                    "video_pos": bi,
                    "pair_id": pi,
                    "global": [int(p.global_seg.start), int(p.global_seg.end)],
                    "local":  [int(p.local_seg.start),  int(p.local_seg.end)],
                })

        batch: Dict[str, Any] = {
            "labels": labels,
            "video_id": video_ids,

            "whisper": whisper_list,
            "vit": vit_list,
            "vit_mask": vit_mask_list,
            "ocr": ocr_list,
            "ocr_mask": ocr_mask_list,
            "text": text_list,
            "text_mask": text_mask_list,

            "T_list": T_list,
            "clip_microbatches": clip_microbatches,
            "video_offsets": video_offsets,

            "contrastive": {
                "cfg": asdict(self.seg_cfg),
                "pairs": pairs_out,
            },

            "rationale": rationale,
            "rationale_mask": rationale_mask,  
        }

        return batch