from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import is_dataclass, asdict
import torch


def get_global_rank(fallback: int = 0) -> int:
    r = os.environ.get("RANK", None)
    if r is None:
        return fallback
    try:
        return int(r)
    except Exception:
        return fallback

class GVTJSONLLogger:


    def __init__(self, out_dir: Union[str, Path], rank: Optional[int] = None, filename_prefix: str = "gvt_gates"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.rank = int(rank) if rank is not None else get_global_rank(0)
        self.path = out_dir / f"{filename_prefix}.rank{self.rank}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")

    @torch.no_grad()
    def log_batch(
        self,
        *,
        split: str,               
        epoch: int,
        step: int,
        global_step: Optional[int],
        video_ids: List[str],      
        gates: Dict[str, Any],     
    ) -> None:
        assert split in ("train", "valid", "test")

        gr = gates["gate_rationale"].detach().float().cpu()  
        gc = gates["gate_clips"].detach().float().cpu()      

        gt = gates.get("gate_rat_tokens", None)              
        if torch.is_tensor(gt):
            gt = gt.detach().float().cpu()
       
        has_token_gate = (gt is not None) and (gt.dim() == 2) and (gt.size(1) > 0)

        B = gr.numel()
        if len(video_ids) != B:
            raise ValueError(f"video_ids len {len(video_ids)} != B {B}")

        for i in range(B):
            gate_rat_tokens = (gt[i].tolist() if has_token_gate else None)  
            rec = {
                "split": split,
                "epoch": int(epoch),
                "step": int(step),
                "global_step": (int(global_step) if global_step is not None else None),
                "video_id": str(video_ids[i]),
                "gate_rationale": float(gr[i].item()),
                "gate_clips": float(gc[i].item()),
                "gate_rat_tokens": gate_rat_tokens,
            }
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def file_path(self) -> str:
        return str(self.path)


def _to_jsonable(x: Any) -> Any:
    """Convert tensors / scalars into JSON-serializable Python types."""
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.dim() == 0:
            return x.item()
        return x.tolist()
    if isinstance(x, (int, float, str, bool)):
        return x
    return x  


def _aux_to_dict(aux: Any) -> Dict[str, Any]:
    """
    Accept:
      - dataclass (MoEAux)
      - dict
    Return:
      - dict of fields -> values (usually tensors shaped [N,...] or scalar tensors)
    """
    if aux is None:
        return {}
    if is_dataclass(aux):
        return asdict(aux)
    if hasattr(aux, "__dict__") and not isinstance(aux, dict):
        return dict(aux.__dict__)
    if isinstance(aux, dict):
        return aux
    raise TypeError(f"Unsupported aux type: {type(aux)}")


def _take_i(v: Any, i: int) -> Any:
    if v is None:
        return None
    if torch.is_tensor(v):
        if v.dim() == 0:
            return _to_jsonable(v)
       
        return _to_jsonable(v[i])
    return _to_jsonable(v)



class MoEJSONLLogger:
   

    def __init__(self, out_dir: Union[str, Path], rank: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.path = self.out_dir / f"moe_aux_rank{self.rank}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @torch.no_grad()
    def log_microbatch(
        self,
        *,
        split: str,                       
        epoch: int,
        step: int,
        global_step: Optional[int],
        meta: Dict[str, Any],            
        aux: Any,                         
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,  
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert split in ("train", "valid", "test"), f"split must be train/valid/test, got {split}"

        video_ids = meta.get("video_id", None)
        clip_idxs = meta.get("clip_idx", None)
        if video_ids is None or clip_idxs is None:
            raise KeyError("meta must contain keys: 'video_id' and 'clip_idx' (lists aligned with N clips)")

        if len(video_ids) != len(clip_idxs):
            raise ValueError(f"meta['video_id'] and meta['clip_idx'] length mismatch: {len(video_ids)} vs {len(clip_idxs)}")

        N = len(video_ids)

        aux_dict = _aux_to_dict(aux) 

       
        mm = modality_masks or {}
    
        mm_keys = ["audio_mask", "text_mask", "visual_mask"]
        mm = {k: mm[k] for k in mm_keys if k in mm}

        for i in range(N):
            rec = {
                "rank": self.rank,
                "split": split,
                "epoch": int(epoch),
                "step": int(step),
                "global_step": int(global_step) if global_step is not None else int(step),
                "video_id": str(video_ids[i]),
                "clip_idx": int(clip_idxs[i]),
                "aux": {},
                "modality_masks": {},
            }

       
            for k, v in aux_dict.items():
                rec["aux"][k] = _take_i(v, i)

            for k, v in mm.items():
                rec["modality_masks"][k] = _take_i(v, i)

            if extra:
                rec["extra"] = {ek: _to_jsonable(ev) for ek, ev in extra.items()}

            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self._fh.flush()

    def file_path(self) -> str:
        return str(self.path)