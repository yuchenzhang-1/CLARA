from __future__ import annotations
from datetime import datetime
import sys
import json
import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import HfArgumentParser, TrainerCallback
from transformers import TrainingArguments as HFTrainingArguments


from utils.training_arguments import TrainingArguments, DataArguments
from utils.dataset import VideoDataset
from utils.collator import CLARACollator
from utils.evaluation import compute_metrics
from utils.segment_sampling import SegmentSamplingConfig
from model.clara import CLARA, CLARAConfig
from model.video_transformer import GatedVideoTransformer, GatedVideoTransformerConfig
from model.trainer import build_trainer
from model.clip_encoder import (
    FeatureProjector,
    ClipEncoder,
    MoEClipEncoder,
    MLPEncoder,
)


def _parse_explicit_cli(argv):

    out = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k = a[2:]
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                out[k] = True
                i += 1
            else:
                out[k] = argv[i + 1]
                i += 2
        else:
            i += 1
    return out

def saving_training_args(*, training_args_obj, data_args_obj, explicit_cli_args: dict) -> dict:

    hf_keys = {f.name for f in fields(HFTrainingArguments)}
    data_keys = {f.name for f in fields(data_args_obj.__class__)}

    train_all = asdict(training_args_obj)


    training_args_custom = {k: v for k, v in train_all.items() if k not in hf_keys}

    cli_hf_dedup = {}
    for k, v in explicit_cli_args.items():
        if k not in hf_keys:           
            continue
        if k in data_keys:             
            continue
        if k in training_args_custom:  
            continue
        cli_hf_dedup[k] = v


    merged = {**cli_hf_dedup, **training_args_custom}
    return merged

def _get_global_rank(args: TrainingArguments) -> int:
    r = os.environ.get("RANK", None)
    if r is not None:
        try:
            return int(r)
        except Exception:
            pass
    try:
        return int(args.local_rank)
    except Exception:
        return 0

def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

class SetEpochCallback(TrainerCallback):
    def __init__(self, collator: Any):
        self.collator = collator

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.collator, "set_epoch"):
            try:
                ep = int(state.epoch) if state.epoch is not None else 0
            except Exception:
                ep = 0
            self.collator.set_epoch(ep)
        return control

def infer_text_dim(text_emb_model: str) -> int:
    m = text_emb_model.lower()
    if m == "bert":
        return 768
    if m == "qwen_0.6":
        return 1024
    if m == "qwen_8":
        return 4096  
    raise ValueError(f"Unknown text_emb_model: {text_emb_model}")



def build_seg_cfg(training_args: TrainingArguments) -> SegmentSamplingConfig:
    
    return SegmentSamplingConfig(
        mode=training_args.seg_mode,
        num_pairs_per_video=training_args.seg_num_pairs_per_video,

        l_ratio=training_args.seg_l_ratio,
        g_ratio=training_args.seg_g_ratio,

        min_l=training_args.seg_min_l,
        min_g=training_args.seg_min_g,
        max_l=training_args.seg_max_l,
        max_g=training_args.seg_max_g,

        base_seed=training_args.seg_base_seed,
    )

def build_clip_encoder(training_args: TrainingArguments) -> ClipEncoder:
  
    projector = FeatureProjector(
        proj_out_dim=training_args.proj_out_dim,
        raw_textual_emb_dim=training_args.raw_textual_emb_dim,    
        proj_hidden=training_args.proj_hidden,             
        proj_dropout=training_args.proj_dropout,
        proj_pool_type = training_args.proj_pool_type,
    )

    backend_name = training_args.clip_encoder_backend

    if backend_name == "moe":
        backend = MoEClipEncoder(
            proj_out_dim=training_args.proj_out_dim, 
            moe_num_experts=training_args.moe_num_experts,
            moe_top_k=training_args.moe_top_k,
            moe_dropout=training_args.moe_dropout,
            moe_use_mask_in_gating=training_args.moe_use_mask_in_gating,
            moe_fusion_type=training_args.moe_fusion_type,
            moe_prior_lambda=training_args.moe_prior_lambda,
            moe_expert_ffn_mult=training_args.moe_expert_ffn_mult,
        )


    elif backend_name == "mlp":
        backend = MLPEncoder(
            proj_out_dim=training_args.proj_out_dim,    
            clipencoder_mlp_ffn_mult=training_args.clipencoder_mlp_ffn_mult,
            clipencoder_mlp_dropout=training_args.clipencoder_mlp_dropout,
        )

    else:
        raise ValueError(f"Unknown fusion_backend: {backend_name}")


    return ClipEncoder(projector=projector, backend=backend)

def build_gvt(training_args: TrainingArguments) -> GatedVideoTransformer:
    cfg = GatedVideoTransformerConfig(
        proj_out_dim=training_args.proj_out_dim,
        raw_textual_emb_dim=training_args.raw_textual_emb_dim, 
        num_layers=training_args.gvt_num_layers,
        num_heads=training_args.gvt_num_heads,
        gvt_ffn_mult=training_args.gvt_ffn_mult,
        dropout=training_args.gvt_dropout,
        max_seq_len_clips=training_args.gvt_max_seq_len_clips,
        max_seq_len_rationale=training_args.gvt_max_seq_len_rationale,
        use_cls=training_args.gvt_use_cls,
        use_mean_pool=training_args.gvt_use_mean_pool,
        mean_pool_include_rationale=training_args.gvt_mean_pool_include_rationale,
        rationale_mode_default=training_args.gvt_rationale_mode,
        pos_encoding=training_args.gvt_pos_encoding,
        gate_init_p=training_args.gvt_gate_init_p,
        use_source_gate=training_args.gvt_use_source_gate,
        use_rationale_token_gate=training_args.gvt_use_rationale_token_gate,
    )
    return GatedVideoTransformer(cfg)

def build_clara_cfg(training_args: TrainingArguments) -> CLARAConfig:
    return CLARAConfig(
        proj_out_dim=training_args.proj_out_dim,
        raw_textual_emb_dim=training_args.raw_textual_emb_dim, 
        use_transformer=training_args.use_transformer,
        use_contrastive=training_args.use_contrastive,
        rationale_mode=training_args.gvt_rationale_mode,
        mean_pool_include_rationale=training_args.gvt_mean_pool_include_rationale,
        num_classes=training_args.num_classes,
        clf_dropout=training_args.clf_dropout,
    )


def save_predictions_for_roc(
    *,
    out_path: str | Path,
    video_ids: List[str],
    labels: np.ndarray,
    logits: np.ndarray,
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    probs = _softmax_np(logits, axis=-1)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        y_score = probs[:, 1]
    else:
        y_score = probs.reshape(-1)

    with out_path.open("w", encoding="utf-8") as f:
        for vid, y, score, logit in zip(video_ids, labels.tolist(), y_score.tolist(), logits.tolist()):
            f.write(json.dumps({
                "video_id": vid,
                "label": int(y),
                "score_pos": float(score),
                "logits": logit,
            }, ensure_ascii=False) + "\n")



def main():
    parser = HfArgumentParser((TrainingArguments, DataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    training_args.raw_textual_emb_dim = infer_text_dim(data_args.text_emb_model)
    

    if training_args.output_dir is None or str(training_args.output_dir).strip() == "":
        raise ValueError("TrainingArguments.output_dir must be set (HF Trainer requires it).")

    _ensure_dir(training_args.output_dir)

    if training_args.moe_log_dir is not None:
        _ensure_dir(training_args.moe_log_dir)
    if training_args.gvt_log_dir is not None:
        _ensure_dir(training_args.gvt_log_dir)


    rank = _get_global_rank(training_args)


    train_dataset = None
    eval_dataset = None
    test_dataset = None

    if training_args.do_train:
        train_dataset = VideoDataset(data_args=data_args, split="train")
    if training_args.do_eval:
        eval_dataset = VideoDataset(data_args=data_args, split="valid")
    if training_args.do_predict:
        test_dataset = VideoDataset(data_args=data_args, split="test")



    seg_cfg = build_seg_cfg(training_args)
    collator = CLARACollator(data_args, seg_cfg=seg_cfg, rank=rank)

  
    clip_encoder = build_clip_encoder(training_args)
    transformer = build_gvt(training_args) if training_args.use_transformer else None
    clara_cfg = build_clara_cfg(training_args)

    model = CLARA(
        cfg=clara_cfg,
        clip_encoder=clip_encoder,
        transformer=transformer,
    )
    if rank == 0:
        print(model)
    
  
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

  
    trainer.add_callback(SetEpochCallback(collator))


    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
       

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


    if training_args.do_predict:
        pred = trainer.predict(test_dataset)
        test_metrics = pred.metrics
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

        test_video_ids = [test_dataset.samples[i]["video_id"] for i in range(len(test_dataset))]
        logits = np.asarray(pred.predictions)
        labels = np.asarray(pred.label_ids)

        save_predictions_for_roc(
            out_path=Path(training_args.output_dir) / "predictions_test.jsonl",
            video_ids=test_video_ids,
            labels=labels,
            logits=logits,
        )


    data_dict = asdict(data_args)
    explicit_cli = _parse_explicit_cli(sys.argv[1:])
    training_args_merged = saving_training_args(
        training_args_obj=training_args,
        data_args_obj=data_args,
        explicit_cli_args=explicit_cli,   
    )

    summary = {
        "output_dir": training_args.output_dir,
        "data_args": data_dict,
        "training_args": training_args_merged,
    }

    (Path(training_args.output_dir) / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()