from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback, EarlyStoppingCallback, set_seed
import torch.nn as nn
from torch.nn.modules.lazy import LazyModuleMixin

from model.segment_contrast import SegmentContrastive, ContrastiveConfig
from utils.logger import MoEJSONLLogger, GVTJSONLLogger


class SaveLastCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.last_dir = os.path.join(output_dir, "last")

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return control
        if os.path.isdir(self.last_dir):
            shutil.rmtree(self.last_dir, ignore_errors=True)
        shutil.copytree(ckpt_dir, self.last_dir)
        return control


def _iter_microbatches(mb_plan: Any) -> List[Tuple[int, int, int]]:

    if mb_plan is None or len(mb_plan) == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    for i, x in enumerate(mb_plan):
        if not (isinstance(x, (list, tuple)) and len(x) == 3):
            raise ValueError(
                f"clip_microbatches must be List[(video_pos,t0,t1)], "
                f"got element {i}: {type(x)} {x}"
            )
        bi, t0, t1 = int(x[0]), int(x[1]), int(x[2])
        if t1 <= t0:
            raise ValueError(f"Invalid microbatch at {i}: (t0,t1)=({t0},{t1})")
        if bi < 0:
            raise ValueError(f"Invalid microbatch at {i}: video_pos={bi}")
        out.append((bi, t0, t1))
    return out


def _mb_meta_from_plan(
    *,
    video_ids: List[str],
    video_pos: int,
    t0: int,
    t1: int,
) -> Dict[str, Any]:

    vid = video_ids[video_pos]
    n = int(t1 - t0)
    return {
        "video_id": [vid] * n,
        "clip_idx": list(range(int(t0), int(t1))),
    }


def _get_global_rank(training_args) -> int:
    r = os.environ.get("RANK", None)
    if r is not None:
        try:
            return int(r)
        except Exception:
            pass
    if hasattr(training_args, "local_rank"):
        try:
            return int(training_args.local_rank)
        except Exception:
            pass
    return 0


def _require_fields(obj: Any, fields: List[str], where: str) -> None:
    missing = [k for k in fields if not hasattr(obj, k)]
    if missing:
        raise AttributeError(f"{where} missing required fields: {missing}")


def _move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_move_to_device(v, device) for v in x]
        return type(x)(t)
    return x



class CLARATrainer(Trainer):

    def __init__(
        self,
        *,
        model,
        args,
        train_dataset,
        eval_dataset,
        data_collator,
        compute_metrics,
        callbacks,
        contrastive_module: Optional[SegmentContrastive],
        moe_logger: Optional[MoEJSONLLogger],
        gvt_logger: Optional[GVTJSONLLogger],
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        self.contrastive_module = contrastive_module
        self._moe_logger = moe_logger
        self._gvt_logger = gvt_logger

        
        self._predict_split = "test"

        
        self._rank = _get_global_rank(args)

    def train(self, *args, **kwargs):
        
        return super().train(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        
        self._predict_split = "valid"
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix: str = "test"):
        
        self._predict_split = "test"
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


    def _log_moe_from_outputs(self, *, split: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        if self._moe_logger is None:
            return

        clip_aux_info = outputs.get("clip_aux_info", None)
        if not isinstance(clip_aux_info, dict):
            return

        aux_list = clip_aux_info.get("aux_list", None)
        mm_list = clip_aux_info.get("modality_masks_list", None)
        if aux_list is None or mm_list is None:
            return

        try:
            mb_plan = inputs["clip_microbatches"]          
            video_ids = inputs["video_id"]                 
        except Exception:
            return

        slices = _iter_microbatches(mb_plan)
        if len(slices) != len(aux_list) or len(slices) != len(mm_list):
            return

        try:
            epoch = int(self.state.epoch) if self.state.epoch is not None else -1
        except Exception:
            epoch = -1
        step = int(self.state.global_step)

        for (video_pos, t0, t1), aux, mm in zip(slices, aux_list, mm_list):
            try:
                mb_meta = _mb_meta_from_plan(
                    video_ids=video_ids,
                    video_pos=int(video_pos),
                    t0=int(t0),
                    t1=int(t1),
                )
                self._moe_logger.log_microbatch(
                    split=split,
                    epoch=epoch,
                    step=step,
                    global_step=step,
                    meta=mb_meta,
                    aux=aux,
                    modality_masks=mm,
                )
            except Exception:
                pass


    def _log_gvt_from_outputs(self, *, split: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        if self._gvt_logger is None:
            return

        gates = outputs.get("transformer_gates", None)
        if not isinstance(gates, dict):
            return

        
        video_ids = inputs.get("video_id", None)
        if not isinstance(video_ids, list):
            try:
                video_ids = list(video_ids)
            except Exception:
                return

        try:
            epoch = int(self.state.epoch) if self.state.epoch is not None else -1
        except Exception:
            epoch = -1
        step = int(self.state.global_step)

        try:
            self._gvt_logger.log_batch(
                split=split,
                epoch=epoch,
                step=step,
                global_step=step,
                video_ids=video_ids,
                gates=gates,
            )
        except Exception:
            pass

    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs):
        outputs = model(inputs)
        logits = outputs["logits"]
        labels = inputs["labels"].to(logits.device)

        cls_loss = F.cross_entropy(logits, labels)
        total_loss = cls_loss

        split = "train" if model.training else "valid"


        con_loss = None
        acc_g2l = None
        acc_l2g = None

        if self.args.use_contrastive:
            if self.contrastive_module is None:
                raise ValueError("args.use_contrastive=True but contrastive_module is None.")

            c_out = outputs.get("contrastive_out", None)
            if not isinstance(c_out, dict):
                raise ValueError("contrastive_out must be a dict when use_contrastive=True.")

            g_vecs = c_out.get("g_vecs", None)
            l_vecs = c_out.get("l_vecs", None)
            if g_vecs is None or l_vecs is None:
                raise ValueError("contrastive_out must contain 'g_vecs' and 'l_vecs' when use_contrastive=True.")
            if g_vecs.numel() == 0:
                raise ValueError("g_vecs is empty but use_contrastive=True. Check your pair sampling / clip truncation.")

            con_out = self.contrastive_module(g_vecs, l_vecs)
            con_loss = con_out["loss"]

       
            w = self.args.contrastive_weight
            total_loss = (1 - w) * cls_loss + w * con_loss

            if "acc_g2l" not in con_out or "acc_l2g" not in con_out:
                raise ValueError("contrastive_module output must contain 'acc_g2l' and 'acc_l2g' when enabled.")
            acc_g2l = con_out["acc_g2l"]
            acc_l2g = con_out["acc_l2g"]

 
        lb_loss = outputs.get("lb_loss", None)
        lb_active = (lb_loss is not None) and (self.args.lb_weight > 0.0)
        if lb_active:
            total_loss = total_loss + self.args.lb_weight * lb_loss

  
        log_dict = {
            f"{split}/loss_total": float(total_loss.detach().cpu()),
            f"{split}/loss_cls": float(cls_loss.detach().cpu()),
        }

        if self.args.use_contrastive:
            log_dict[f"{split}/loss_contrast"] = float(con_loss.detach().cpu())
            log_dict[f"{split}/acc_g2l"] = float(acc_g2l)
            log_dict[f"{split}/acc_l2g"] = float(acc_l2g)
            log_dict[f"{split}/contrastive_weight"] = float(self.args.contrastive_weight)

        if lb_loss is not None:
            log_dict[f"{split}/loss_lb"] = float(lb_loss.detach().cpu())
            log_dict[f"{split}/lb_weight"] = float(self.args.lb_weight)

        self.log(log_dict)


        if self._rank == 0 and self.state.global_step % 50 == 0:
            if self.args.use_contrastive:
                w = float(self.args.contrastive_weight)
                eq = f"total=(1-{w})*cls+{w}*contrast"
            else:
                eq = "total=cls"
            if lb_active:
                eq += f"+{float(self.args.lb_weight)}*lb"
            print(f"[step {int(self.state.global_step)}] {eq}")

    
        self._log_moe_from_outputs(split=split, inputs=inputs, outputs=outputs)
        self._log_gvt_from_outputs(split=split, inputs=inputs, outputs=outputs)

        return (total_loss, outputs) if return_outputs else total_loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels", None)

        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs["logits"]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                loss = F.cross_entropy(logits, labels)


        self._log_moe_from_outputs(split=self._predict_split, inputs=inputs, outputs=outputs)
        self._log_gvt_from_outputs(split=self._predict_split, inputs=inputs, outputs=outputs)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def build_trainer(
    *,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    compute_metrics,
) -> CLARATrainer:

    _require_fields(
        training_args,
        [
            "seed",
            "output_dir",
            "use_contrastive",
            "contrastive_weight",
            "contrast_tau",
            "contrast_normalize",
            "contrast_reduction",
            "lb_weight",
            "moe_log_dir",
            "gvt_log_dir",
            "early_stopping_patience",
            "early_stopping_threshold",
            "save_last",
        ],
        "TrainingArguments",
    )

    if training_args.seed is not None:
        set_seed(training_args.seed)

    contrastive_module = None
    if training_args.use_contrastive:
        c_cfg = ContrastiveConfig(
            tau=float(training_args.contrast_tau),
            normalize=bool(training_args.contrast_normalize),
            reduction=str(training_args.contrast_reduction),
        )
        contrastive_module = SegmentContrastive(cfg=c_cfg)

    rank = _get_global_rank(training_args)

    moe_logger = None
    if training_args.moe_log_dir is not None:
        moe_logger = MoEJSONLLogger(out_dir=training_args.moe_log_dir, rank=rank)

    gvt_logger = None
    if training_args.gvt_log_dir is not None:
        gvt_logger = GVTJSONLLogger(out_dir=training_args.gvt_log_dir, rank=rank)

    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold,
        ))

    if training_args.save_last:
        callbacks.append(SaveLastCallback(training_args.output_dir))

    return CLARATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        contrastive_module=contrastive_module,
        moe_logger=moe_logger,
        gvt_logger=gvt_logger,
    )