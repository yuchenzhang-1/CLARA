from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class DataArguments:
    _argument_group_name = "Data arguments"


    dataset_root: str = field(
        default="...",
        metadata={"help": "Preprocessed root dir. dataset.py uses: <root>/<dataset_name>/embeddings etc."},
    )
    dataset_name: str = field(
        default="...",
        metadata={"help": "Dataset name under preprocessed_root, e.g. DeHate / HateMM / MHC_CN ..."},
    )
    split_json: str = field(
        default="",
        metadata={"help": "5-fold split json path. dataset reads folds[fold][split]."},
    )
    fold: int = field(default=0, metadata={"help": "Fold id (0..4)."})
    budget: int = field(
        default=40,
        metadata={"help": "Embedding budget: choose among {40,60,80,100,120}. Controls vit_{b}.pt and bert_ocr_{b}.pt."},
    )

    max_clips_per_video: int = field(
        default=100,
        metadata={"help": "Max clips kept per video in Dataset (before collator). Large number disables truncation."},
    )

    clip_select: Literal["truncate", "uniform", "random"] = field(
        default="truncate",
        metadata={"help": "How to choose K clips if a video has more than max_clips_per_video."},
    )

    clip_select_seed: int = field(
        default=42,
        metadata={"help": "Base seed for deterministic clip_select=random (mixed with video_id hash)."},
    )

    text_emb_model: Literal["bert", "qwen_0.6", "qwen_8"] = field(
        default="bert",
        metadata={"help": "text embedding model used to embed transcript/ocr/rationle"},
    )

    # rationale
    rationale_source: str = field(
        default="Qwen",
        metadata={"help": "Rationale source suffix, e.g. LLaVA/Qwen. If None -> do not load rationale."},
    )
    map_location: str = field(
        default="cpu",
        metadata={"help": "torch.load map_location used by dataset (cpu/cuda)."},
    )



@dataclass
class TrainingArguments(HFTrainingArguments):
    _argument_group_name = "Training+Model arguments (extends HF TrainingArguments)"


    proj_out_dim: int = field(default=256, metadata={"help": "Projector output dim for each modality."})
    proj_hidden: int = field(default=768, metadata={"help": "TwoLayerMLP hidden dim in projector."})
    proj_dropout: float = field(default=0.1, metadata={"help": "Projector dropout."})
    proj_pool_type: Literal["attentive", "mean",] = field(default="mean", metadata={"help": "how to pool rwa afeatures of vit/whisper/ocr."},)
    raw_textual_emb_dim: int = field(default=-1,metadata={"help": "Inferred raw text embedding dim for transcript/ocr/rationale (set in main)."},)
    clip_encoder_backend: Literal["moe", "mlp",] = field(default="moe", metadata={"help": "Clip-level fusion backend."},)
    
    moe_expert_ffn_mult: int = field(default=1, metadata={"help": "Expert MLP FFN expansion factor."})
    moe_num_experts: int = field(default=8, metadata={"help": "Number of MoE experts."})
    moe_top_k: int = field(default=3, metadata={"help": "Top-k experts activated per clip."})
    moe_dropout: float = field(default=0.1, metadata={"help": "Expert dropout."})
    moe_use_mask_in_gating: bool = field(default=True, metadata={"help": "Concat modality masks into gating input."})
    moe_fusion_type: Literal["weighted_sum", "weighted_attention"] = field(default="weighted_sum",metadata={"help": "MoE fusion type."},)
    moe_prior_lambda: float = field(default=1.0, metadata={"help": "Prior lambda for weighted_attention fusion."})

    clipencoder_mlp_ffn_mult: int = field(default=1, metadata={"help": "FFN expansion factor for MLPEncoder."})
    clipencoder_mlp_dropout: float = field(default=0.1, metadata={"help": "Dropout for MLPEncoder."})


    seg_mode: Literal["within", "independent"] = field(default="independent", metadata={"help": "Segment sampling mode: within or independent."})
    seg_num_pairs_per_video: int = field(default=1, metadata={"help": "Number of contrastive pairs per video."})
    seg_l_ratio: float = field(default=0.3, metadata={"help": "Local segment ratio."})
    seg_g_ratio: float = field(default=0.8, metadata={"help": "Global segment ratio."})
    seg_min_l: int = field(default=2, metadata={"help": "Min local segment length."})
    seg_min_g: int = field(default=4, metadata={"help": "Min global segment length."})
    seg_max_l: int = field(default=16, metadata={"help": "Max local segment length."})
    seg_max_g: int = field(default=64, metadata={"help": "Max global segment length."})
    seg_base_seed: int = field(default=42, metadata={"help": "Base seed for segment sampling."})

    contrast_tau: float = field(default=0.07, metadata={"help": "InfoNCE temperature."})
    contrast_normalize: bool = field(default=True, metadata={"help": "L2-normalize g/l vectors."})
    contrast_reduction: Literal["mean", "sum"] = field(default="mean", metadata={"help": "CE reduction."})


    gvt_num_layers: int = field(default=4, metadata={"help": "Transformer layers."})
    gvt_num_heads: int = field(default=8, metadata={"help": "Transformer heads."})
    gvt_ffn_mult: int = field(default=4, metadata={"help": "Transformer FFN dim = gvt_ffn_mult * gvt_hidden_dim."})
    gvt_dropout: float = field(default=0.1, metadata={"help": "Transformer dropout."})
    gvt_max_seq_len_clips: int = field(default=100, metadata={"help": "Transformer max length for clips."})
    gvt_max_seq_len_rationale: int = field(default=8, metadata={"help": "Transformer max length for rationales."})
    gvt_pos_encoding: Literal["sinusoidal", "learnable"] = field(default="sinusoidal",metadata={"help": "Positional encoding type."},)
    
    gvt_use_cls: bool = field(default=True, metadata={"help": "Use CLS token output."})
    gvt_use_mean_pool: bool = field(default=False, metadata={"help": "Use masked mean pooling output."})
    gvt_mean_pool_include_rationale: bool = field(default=True,metadata={"help": "Include rationale embedding while mean pooling"},)
    gvt_rationale_mode: Literal["none", "obj", "dec", "both"] = field(default="both",metadata={"help": "Which rationale tokens to include: none|obj|dec|both."},)
   
    gvt_gate_init_p: float = field(default=0.9, metadata={"help": "Init sigmoid(gate)=p."})
    gvt_use_source_gate: bool = field(default=False, metadata={"help": "Use source-level  gates."})
    gvt_use_rationale_token_gate: bool = field(default=False, metadata={"help": "Use ratioanle token gate."})
    

 
    use_transformer: bool = field(default=True, metadata={"help": "Use video transformer. If False, mean-pool [rationale+clips] as video embedding."},)
    use_contrastive: bool = field(default=True, metadata={"help": "Enable segment-level contrastive learning (loss computed in trainer)."},)
    num_classes: int = field(default=2, metadata={"help": "Classification classes (binary=2)."})
    clf_dropout: float = field(default=0.1, metadata={"help": "Classifier dropout."})

    lb_weight: float = field(default=0.01,metadata={"help": "Total loss += lb_weight * lb_loss (MoE load-balancing). 0 disables."},)

    contrastive_weight: float = field(
        default=0.3,
        metadata={"help": "Total loss = (1-contrastive_weight) * cls_loss + contrastive_weight * contrastive_loss."},
    )
    
 
    early_stopping_patience: int = field(default=10, metadata={"help": "EarlyStopping patience in eval steps."})
    early_stopping_threshold: float = field(default=0.0, metadata={"help": "Min improvement delta."})
    save_last: bool = field(default=False, metadata={"help": "Maintain output_dir/last checkpoint."})
    save_best: bool = field(default=True, metadata={"help": "Save best checkpoint using metric_for_best_model."})

    moe_log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "If set, write MoE aux JSONL logs to this directory (per-rank file)."},
    )
    gvt_log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "If set, write GVT logs to this directory (per-rank file)."},
    )
