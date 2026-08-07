#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


def _stable_int_from_str(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _select_clip_indices(
    T: int,
    K: int,
    mode: str,
    *,
    video_id: Optional[str] = None,
    seed: int = 42,
) -> List[int]:
    if K <= 0 or T <= 0:
        return []
    if T <= K:
        return list(range(T))

    mode = str(mode).lower()
    if mode == "truncate":
        return list(range(K))

    if mode == "uniform":
        if K == 1:
            return [0]
        idxs = [round(i * (T - 1) / (K - 1)) for i in range(K)]
        idxs = sorted(set(idxs))
        if len(idxs) < K:
            used = set(idxs)
            for j in range(T):
                if j not in used:
                    idxs.append(j)
                    used.add(j)
                    if len(idxs) == K:
                        break
        return sorted(idxs[:K])

    if mode == "random":
        s = int(seed)
        if video_id is not None:
            s = s + _stable_int_from_str(video_id)
        rng = random.Random(s)
        return sorted(rng.sample(range(T), K))

    raise ValueError(f"Unknown clip_select mode: {mode}")


def _as_bool_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bool:
            return x
        return x != 0
    return torch.tensor(x, dtype=torch.bool)


class VideoDataset(Dataset):
    def __init__(self, data_args, split: str):
        super().__init__()
        assert split in {"train", "valid", "test"}
        self.split = split

        
        self.dataset_root = data_args.dataset_root
        self.dataset_name = data_args.dataset_name
        self.split_json = data_args.split_json
        self.fold = data_args.fold
        self.map_location: str = data_args.map_location

       
        self.max_clips_per_video = data_args.max_clips_per_video
        self.clip_select = data_args.clip_select
        self.clip_select_seed = data_args.clip_select_seed

        
        self.rationale_source = data_args.rationale_source.strip()   
        self.text_emb_model = data_args.text_emb_model.strip()       

        
        self.frame_budget = data_args.budget
    
        self.root = Path(self.dataset_root)
        self.video_emb_root = self.root / self.dataset_name / "video_embeddings"

        raw = json.loads(Path(self.split_json).read_text(encoding="utf-8"))
        self.samples: List[Dict[str, Any]] = raw["folds"][self.fold][split]

        if self.text_emb_model not in {"bert", "qwen_0.6", "qwen_8"}:
            raise ValueError(f"Unknown text_emb_model: {self.text_emb_model}")
        if self.rationale_source not in {"Qwen", "LLaVA"}:
            raise ValueError(f"Unknown rationale_source: {self.rationale_source} (expected Qwen/LLaVA)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        video_id = str(item["video_id"])
        label = int(item["label"])

        pack_path = self.video_emb_root / f"{video_id}.pt"
        pack = torch.load(pack_path, map_location=self.map_location, weights_only=True)


        whisper = pack["whisper"]  
        whisper_mask = _as_bool_tensor(pack["whisper_mask"])  


        vit_key = f"vit_{self.frame_budget}"
        vit_mask_key = f"vit_mask_{self.frame_budget}"
        if vit_key not in pack or vit_mask_key not in pack:
            raise KeyError(f"Missing {vit_key} / {vit_mask_key} in {pack_path}")

        vit = pack[vit_key]  
        vit_mask = _as_bool_tensor(pack[vit_mask_key])  

        text_key = f"text_{self.text_emb_model}"
        if text_key not in pack:
            raise KeyError(f"Missing {text_key} in {pack_path}")
        text = pack[text_key]  
        text_mask = _as_bool_tensor(pack["text_mask"]) 


        ocr_key = f"ocr_{self.text_emb_model}_{self.frame_budget}"
        ocr_mask_key = f"ocr_mask_{self.frame_budget}"
        if ocr_key not in pack or ocr_mask_key not in pack:
            raise KeyError(f"Missing {ocr_key} / {ocr_mask_key} in {pack_path}")

        ocr = pack[ocr_key]  
        ocr_mask = _as_bool_tensor(pack[ocr_mask_key]) 

  
        rat_key = f"rationale_{self.rationale_source}_{self.text_emb_model}"
        rat_mask_key = f"rationale_mask_{self.rationale_source}"
        if rat_key not in pack or rat_mask_key not in pack:
            raise KeyError(f"Missing {rat_key} / {rat_mask_key} in {pack_path}")

        rationale = pack[rat_key]  
        rationale_mask = _as_bool_tensor(pack[rat_mask_key])  


        T_all = int(vit.size(0))
        keep = _select_clip_indices(
            T=T_all,
            K=int(self.max_clips_per_video),
            mode=self.clip_select,
            video_id=video_id,
            seed=int(self.clip_select_seed),
        )

        if len(keep) == 0:
            whisper = whisper[:0]
            whisper_mask = whisper_mask[:0]
            vit = vit[:0]
            vit_mask = vit_mask[:0]
            ocr = ocr[:0]
            ocr_mask = ocr_mask[:0]
            text = text[:0]
            text_mask = text_mask[:0]
        else:
            whisper = whisper[keep]
            whisper_mask = whisper_mask[keep]
            vit = vit[keep]
            vit_mask = vit_mask[keep]
            ocr = ocr[keep]
            ocr_mask = ocr_mask[keep]
            text = text[keep]
            text_mask = text_mask[keep]

        out: Dict[str, Any] = {
            "video_id": video_id,
            "label": label,
            "T": int(vit.size(0)),

            "whisper": whisper,
            "whisper_mask": whisper_mask,

            "vit": vit,
            "vit_mask": vit_mask,

            "ocr": ocr,
            "ocr_mask": ocr_mask,

            "text": text,
            "text_mask": text_mask,

            "rationale": rationale,
            "rationale_mask": rationale_mask,  
        }
        return out