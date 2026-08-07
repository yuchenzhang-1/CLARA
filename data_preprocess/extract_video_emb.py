from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
from PIL import Image
from tqdm import tqdm

from transformers import ViTImageProcessor, ViTModel
from transformers import BertTokenizer, BertModel
from transformers import WhisperProcessor, WhisperModel
from sentence_transformers import SentenceTransformer



def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_done_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def list_video_dirs(raw_root: Path) -> List[Path]:
    vids: List[Path] = []
    for p in raw_root.iterdir():
        if p.is_dir() and (p / "clipinfo.json").exists():
            vids.append(p)
    return sorted(vids)

def list_frames(frames_dir: Path) -> List[Path]:
    def key(p: Path) -> int:
        m = re.findall(r"\d+", p.stem)
        return int(m[0]) if m else 0
    return sorted(frames_dir.glob("frame_*.jpg"), key=key)

def has_no_frames_placeholder(frames_dir: Path) -> bool:
    return (frames_dir / "_NO_FRAMES.json").exists()

def wav_load_mono_16k(wav_path: Path, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(wav_path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.contiguous(), sr

def slice_audio(wav: torch.Tensor, sr: int, start_s: float, end_s: float) -> torch.Tensor:
    st = max(0, int(round(start_s * sr)))
    ed = max(st + 1, int(round(end_s * sr)))
    ed = min(ed, wav.numel())
    return wav[st:ed].contiguous()



def write_failed_tsv(path: Path, rows: List[Tuple[str, str, str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("video_id\tllm\tstage\terror\n")
        for vid, llm, stage, err in rows:
            f.write(f"{vid}\t{llm}\t{stage}\t{err}\n")



OUT_DTYPE = torch.float16

def cast_fp16(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError("cast_fp16 expects torch.Tensor")
    if t.dtype.is_floating_point:
        return t.to(dtype=OUT_DTYPE)
    return t



@torch.inference_mode()
def whisper_clip_meanpool(
    processor: WhisperProcessor,
    model: WhisperModel,
    audio_16k: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x = audio_16k.detach().cpu().float().numpy()
    feats = processor(
        x,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=30 * 16000,
    )
    input_features = feats.input_features.to(device)
    enc = model.encoder(input_features=input_features).last_hidden_state  
    enc = enc.squeeze(0).detach().cpu().float()                           
    return enc.mean(dim=0)                                               


@torch.inference_mode()
def vit_clip_meanpool(
    vit: ViTModel,
    processor: ViTImageProcessor,
    frame_paths: List[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if len(frame_paths) == 0:
        return torch.zeros((768,), dtype=torch.float32)

    all_cls: List[torch.Tensor] = []
    for i in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[i:i + batch_size]
        images = [Image.open(str(p)).convert("RGB") for p in chunk]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = vit(pixel_values=pixel_values).last_hidden_state[:, 0, :] 
        all_cls.append(out.detach().cpu().float())

    cls = torch.cat(all_cls, dim=0) 
    return cls.mean(dim=0)         



@torch.inference_mode()
def bert_encode_one_cls(
    tokenizer: BertTokenizer,
    bert: BertModel,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    t = str(text).strip()
    if not t:
        return torch.zeros((768,), dtype=torch.float32)
    enc = tokenizer(t, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    h = bert(**enc).last_hidden_state[0, 0, :].detach().cpu().float()
    return h

@torch.inference_mode()
def bert_encode_cls_batch(
    tokenizer: BertTokenizer,
    bert: BertModel,
    texts: List[str],
    device: torch.device,
) -> torch.Tensor:
    xs = [str(x).strip() for x in texts]
    if len(xs) == 0:
        return torch.zeros((0, 768), dtype=torch.float32)
    enc = tokenizer(xs, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    h = bert(**enc).last_hidden_state 
    cls = h[:, 0, :].detach().cpu().float()
    return cls



def qwen_encode_one(model: SentenceTransformer, text: str, dim: int) -> torch.Tensor:
    t = str(text).strip()
    if not t:
        return torch.zeros((dim,), dtype=torch.float32)
    v = model.encode([t], normalize_embeddings=True)[0]
    return torch.from_numpy(v).to(torch.float32)

def qwen_encode_items(model: SentenceTransformer, items: List[str], batch_size: int, dim: int) -> torch.Tensor:
    xs = [str(x).strip() for x in items if str(x).strip()]
    if len(xs) == 0:
        return torch.zeros((0, dim), dtype=torch.float32)
    arr = model.encode(xs, batch_size=batch_size, normalize_embeddings=True)
    return torch.from_numpy(arr).to(torch.float32)




def load_ocr_words_keep(raw_clip_dir: Path, budget: int) -> List[str]:
    ocr_json = raw_clip_dir / "ocr_text" / "ocr_clip.json"
    if not ocr_json.exists():
        return []
    ocr_data = load_json(ocr_json)
    bd = (ocr_data.get("budgets", {}) or {}).get(str(budget), {}) or {}
    words_keep = bd.get("clip_words_keep", []) or []
    return [str(x) for x in words_keep]



def is_blank_text(s: Any) -> bool:
    return (s is None) or (str(s).strip() == "")

def is_na_text(s: Any) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if t == "":
        return True
    return t.upper() == "N/A"

def transcript_exists_from_clip(c: Dict[str, Any]) -> bool:
    is_silent = bool(c.get("is_silent", False))
    raw_tr = c.get("transcript", "")
    if is_silent:
        return False
    if is_na_text(raw_tr):
        return False
    return True  

def apply_blank_mask_and_zero_rows(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mat: [N,D] float
    mask: [N] bool
    For any mask[i]==False: set mat[i] = 0
    """
    if mat.dim() != 2 or mask.dim() != 1:
        raise ValueError("apply_blank_mask_and_zero_rows expects mat [N,D] and mask [N]")
    if mat.size(0) != mask.size(0):
        raise ValueError(f"row mismatch: mat {mat.size(0)} vs mask {mask.size(0)}")
    out = mat.clone()
    if (~mask).any():
        out[~mask] = 0
    return out


def safe_video_id(rationale: Dict[str, Any], fallback: str) -> str:
    vid = rationale.get("video_id")
    if isinstance(vid, str) and vid.strip():
        return vid.strip()
    return fallback

def build_header_text(final_decision: Dict[str, Any]) -> str:
    label = (final_decision.get("label") or "").strip().lower()
    confidence = (final_decision.get("confidence") or "unknown").strip().lower()
    explicitness = (final_decision.get("explicitness") or "").strip().lower()

    if label == "non-hate":
        return f"The video is considered non-hateful with {confidence} confidence."
    if label == "hate":
        if explicitness:
            return f"The video is considered hateful ({explicitness}) with {confidence} confidence."
        return f"The video is considered hateful with {confidence} confidence."
    return ""

def extract_text_fields_8(rationale: Dict[str, Any]) -> List[str]:
    od = rationale.get("objective_description", {})
    fd = rationale.get("final_decision", {})

    texts: List[str] = []
    texts.append((od.get("objective_summary", "") or ""))
    texts.append((od.get("visual_description", "") or ""))
    texts.append((od.get("textual_description", "") or ""))

    rel = (od.get("cross_modal_relation", "") or "").strip()
    expl = (od.get("cross_modal_explanation", "") or "").strip()
    if rel and expl:
        texts.append(f"The relation between textual and visual content is {rel} and {expl}")
    else:
        texts.append((rel or expl) or "")

    ctx = od.get("contextually_important_elements", [])
    if isinstance(ctx, list) and ctx:
        items = [str(x).strip() for x in ctx if str(x).strip()]
        texts.append("contextually important elements include: " + ", ".join(items) if items else "")
    else:
        texts.append("")

    texts.append(build_header_text(fd))
    texts.append(fd.get("reasons", "") or "")
    texts.append(fd.get("notes", "") or "")

    if len(texts) != 8:
        raise ValueError(f"Expected 8 rationale fields, got {len(texts)}")
    return texts


# =========================
# Main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract embeddings and write flat pt files. Shared masks depend only on content existence."
    )

    ap.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="Root folder containing video subfolders. Each video folder must contain clipinfo.json and clip_XXX subfolders.",
    )
    ap.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Folder containing wav files named {video_id}.wav.",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output folder to write {video_id}.pt. Failure log is written to out_root/_logs/failed.tsv.",
    )

    ap.add_argument(
        "--video_id",
        type=str,
        default="",
        help="If non-empty, only process this one video_id under raw_root.",
    )

    ap.add_argument(
        "--frame_budgets",
        type=str,
        default="40,60,80,100,120",
        help="Comma-separated budgets. Frames read from clip_XXX/frames/frame_{budget}/frame_*.jpg.",
    )
    ap.add_argument(
        "--rationale_dir",
        type=str,
        required=True,
        help="Root folder containing rationale subfolders: rationale_dir/{LLM}/{video_id}_rationale.json.",
    )
    ap.add_argument(
        "--rationale_models",
        type=str,
        default="Qwen,LLaVA",
        help="Comma-separated rationale LLM folder names (LLM keys). Rationale is REQUIRED: missing/failed => skip video.",
    )

    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Whisper/BERT/ViT (cuda or cpu).",
    )
    ap.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU index when device is cuda.",
    )

    ap.add_argument(
        "--vit_batch",
        type=int,
        default=32,
        help="Batch size for ViT frame encoding (frames per forward).",
    )
    ap.add_argument(
        "--ocr_batch_qwen",
        type=int,
        default=128,
        help="Batch size for Qwen OCR item encoding.",
    )

    ap.add_argument(
        "--skip_existing",
        type=str,
        choices=["true", "false"],
        default="true",
        help="If true, skip a video when out_root/{video_id}.pt already exists and is non-empty.",
    )
    ap.add_argument(
        "--max_clips_per_video",
        type=int,
        default=-1,
        help="If > 0, only keep the first N clips from clipinfo.json (debugging).",
    )

    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of shards for multi-GPU processing. Use 1 to disable.",
    )
    ap.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard index in [0, num_shards-1].",
    )

    ap.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-large-v3",
        help="HuggingFace model name for Whisper encoder.",
    )
    ap.add_argument(
        "--vit_ckpt",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="HuggingFace checkpoint for ViT.",
    )
    ap.add_argument(
        "--bert_ckpt",
        type=str,
        default="google-bert/bert-base-uncased",
        help="HuggingFace checkpoint for BERT embeddings. use google-bert/bert-base-chinese when handling Chinese text.",
    )
    ap.add_argument(
        "--qwen06_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer model name for Qwen 0.6B embeddings.",
    )
    ap.add_argument(
        "--qwen8_name",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="SentenceTransformer model name for Qwen 8B embeddings.",
    )

    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    wav_dir = Path(args.wav_dir)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    rat_root = Path(args.rationale_dir)
    budgets = [int(x) for x in args.frame_budgets.split(",") if x.strip()]
    rat_models = [x.strip() for x in args.rationale_models.split(",") if x.strip()]
    skip_existing = (args.skip_existing == "true")

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard_id must be in [0, {args.num_shards - 1}]")


    if args.device.startswith("cuda") and torch.cuda.is_available():
        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"--gpu_id {args.gpu_id} invalid; available 0..{torch.cuda.device_count()-1}")
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    log_dir = out_root / "_logs"
    ensure_dir(log_dir)
    failed_tsv = log_dir / "failed.tsv"
    failed_rows: List[Tuple[str, str, str, str]] = []

    print("[INFO] Loading Whisper...")
    whisper_proc = WhisperProcessor.from_pretrained(args.whisper_model)
    whisper = WhisperModel.from_pretrained(args.whisper_model).to(device).eval()

    print("[INFO] Loading ViT...")
    vit_proc = ViTImageProcessor.from_pretrained(args.vit_ckpt)
    vit = ViTModel.from_pretrained(args.vit_ckpt).to(device).eval()

    print("[INFO] Loading BERT...")
    bert_tok = BertTokenizer.from_pretrained(args.bert_ckpt)
    bert = BertModel.from_pretrained(args.bert_ckpt).to(device).eval()

    RA_BERT_CKPT = "google-bert/bert-base-uncased"

    if args.bert_ckpt == RA_BERT_CKPT:
        ra_bert_tok = bert_tok
        ra_bert = bert
    else:
        ra_bert_tok = BertTokenizer.from_pretrained(RA_BERT_CKPT)
        ra_bert = BertModel.from_pretrained(RA_BERT_CKPT).to(device).eval()

    print("[INFO] Loading Qwen embedding models...")
    qwen_dev = str(device)
    qwen06 = SentenceTransformer(args.qwen06_name, device="cpu")
    qwen8 = SentenceTransformer(args.qwen8_name, device=qwen_dev)
    dim06 = int(qwen06.encode(["dim_probe"], normalize_embeddings=True).shape[1])
    dim8 = int(qwen8.encode(["dim_probe"], normalize_embeddings=True).shape[1])
    print(f"[INFO] Qwen dims: qwen_0.6={dim06}, qwen_8={dim8}")

    text_models = ["bert", "qwen_0.6", "qwen_8"]
    D_of = {"bert": 768, "qwen_0.6": dim06, "qwen_8": dim8}


    if args.video_id.strip():
        vdir = raw_root / args.video_id.strip()
        if not vdir.exists():
            raise FileNotFoundError(f"--video_id {args.video_id} not found under {raw_root}")
        video_dirs = [vdir]
    else:
        video_dirs_all = list_video_dirs(raw_root)
        if not video_dirs_all:
            print(f"[WARN] No video dirs found under {raw_root}")
            return
        video_dirs = [vd for idx, vd in enumerate(video_dirs_all) if (idx % args.num_shards) == args.shard_id]

    t0 = time.time()
    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    pbar = tqdm(video_dirs, desc="extract_to_flat_all", unit="video", dynamic_ncols=True)

    for vdir in pbar:
        video_id = vdir.name
        out_path = out_root / f"{video_id}.pt"

        if skip_existing and is_done_file(out_path):
            skip_cnt += 1
            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)
            continue

        clipinfo_path = vdir / "clipinfo.json"
        wav_path = wav_dir / f"{video_id}.wav"

        if not clipinfo_path.exists():
            failed_rows.append((video_id, "N/A", "clipinfo_missing", f"missing {clipinfo_path}"))
            fail_cnt += 1
            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)
            continue

        if not wav_path.exists():
            failed_rows.append((video_id, "N/A", "wav_missing", f"missing {wav_path}"))
            fail_cnt += 1
            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)
            continue

        try:
            wav, sr = wav_load_mono_16k(wav_path, target_sr=16000)
        except Exception as e:
            failed_rows.append((video_id, "N/A", "wav_load", repr(e)))
            fail_cnt += 1
            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)
            continue

        try:
            clipinfo = load_json(clipinfo_path)
            clips = clipinfo["clips"]
            if not isinstance(clips, list):
                raise ValueError("clipinfo['clips'] must be a list")
        except Exception as e:
            failed_rows.append((video_id, "N/A", "clipinfo_load", repr(e)))
            fail_cnt += 1
            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)
            continue

        if args.max_clips_per_video > 0:
            clips = clips[: int(args.max_clips_per_video)]

        T = len(clips)
        if T == 0:
            failed_rows.append((video_id, "N/A", "no_clips", "empty clips"))
            fail_cnt += 1
            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)
            continue


        rationale_out: Dict[str, torch.Tensor] = {}
        rationale_mask_out: Dict[str, torch.Tensor] = {}

        rationale_failed = False
        for llm in rat_models:
            rjson = rat_root / llm / f"{video_id}_rationale.json"
            if not rjson.exists():
                failed_rows.append((video_id, llm, "rationale_json_missing", f"missing {rjson}"))
                rationale_failed = True
                break

            try:
                rationale = json.loads(rjson.read_text(encoding="utf-8"))
                if not isinstance(rationale, dict):
                    raise ValueError("rationale json must be a dict")

                _ = safe_video_id(rationale, fallback=video_id)

                texts8 = extract_text_fields_8(rationale)  
                mask8 = torch.tensor([not is_blank_text(s) for s in texts8], dtype=torch.bool)  

        
                rationale_mask_out[f"rationale_mask_{llm}"] = mask8.clone()


                r_bert = bert_encode_cls_batch(ra_bert_tok, ra_bert, texts8, device) 
                if r_bert.shape != (8, 768):
                    raise ValueError(f"bert rationale shape {tuple(r_bert.shape)} expected (8,768)")
                r_bert = apply_blank_mask_and_zero_rows(r_bert, mask8)
                rationale_out[f"rationale_{llm}_bert"] = cast_fp16(r_bert)

     
                r06 = torch.from_numpy(qwen06.encode(texts8, batch_size=8, normalize_embeddings=True)).to(torch.float32)
                r8 = torch.from_numpy(qwen8.encode(texts8, batch_size=8, normalize_embeddings=True)).to(torch.float32)

                if r06.shape != (8, dim06):
                    raise ValueError(f"qwen_0.6 rationale shape {tuple(r06.shape)} expected (8,{dim06})")
                if r8.shape != (8, dim8):
                    raise ValueError(f"qwen_8 rationale shape {tuple(r8.shape)} expected (8,{dim8})")

                r06 = apply_blank_mask_and_zero_rows(r06, mask8)
                r8 = apply_blank_mask_and_zero_rows(r8, mask8)

                rationale_out[f"rationale_{llm}_qwen_0.6"] = cast_fp16(r06)
                rationale_out[f"rationale_{llm}_qwen_8"] = cast_fp16(r8)

            except Exception as e:
                failed_rows.append((video_id, llm, "rationale_encode", repr(e)))
                rationale_failed = True
                break

        if rationale_failed:
            fail_cnt += 1
            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)
            continue


        whisper_out = torch.zeros((T, 1280), dtype=OUT_DTYPE)
        whisper_mask = torch.zeros((T,), dtype=torch.bool)

        vit_out: Dict[int, torch.Tensor] = {b: torch.zeros((T, 768), dtype=OUT_DTYPE) for b in budgets}
        vit_mask: Dict[int, torch.Tensor] = {b: torch.zeros((T,), dtype=torch.bool) for b in budgets}


        text_out: Dict[str, torch.Tensor] = {m: torch.zeros((T, D_of[m]), dtype=OUT_DTYPE) for m in text_models}
        text_mask = torch.zeros((T,), dtype=torch.bool)

        ocr_out: Dict[Tuple[str, int], torch.Tensor] = {(m, b): torch.zeros((T, D_of[m]), dtype=OUT_DTYPE)
                                                        for m in text_models for b in budgets}
        ocr_mask: Dict[int, torch.Tensor] = {b: torch.zeros((T,), dtype=torch.bool) for b in budgets}


        for i, c in enumerate(clips):
            clip_idx = int(c["clip_idx"])
            clip_name = f"clip_{clip_idx:03d}"
            raw_cdir = vdir / clip_name
            start_s = float(c["start"])
            end_s = float(c["end"])

            try:
                audio_clip = slice_audio(wav, sr, start_s, end_s)
                w = whisper_clip_meanpool(whisper_proc, whisper, audio_clip, device)
                whisper_out[i] = cast_fp16(w)
                whisper_mask[i] = True
            except Exception as e:
                whisper_mask[i] = False
                failed_rows.append((video_id, "N/A", "whisper_encode", f"clip={clip_idx} {repr(e)}"))


            for b in budgets:
                frames_dir = raw_cdir / "frames" / f"frame_{b}"
                if (not frames_dir.exists()) or has_no_frames_placeholder(frames_dir):
                    vit_mask[b][i] = False
                    continue
                frame_paths = list_frames(frames_dir)
                if len(frame_paths) == 0:
                    vit_mask[b][i] = False
                    continue
                try:
                    v = vit_clip_meanpool(vit, vit_proc, frame_paths, device, batch_size=int(args.vit_batch))
                    vit_out[b][i] = cast_fp16(v)
                    vit_mask[b][i] = True
                except Exception as e:
                    vit_mask[b][i] = False
                    failed_rows.append((video_id, "N/A", "vit_encode", f"clip={clip_idx} budget={b} {repr(e)}"))

            has_text = transcript_exists_from_clip(c)
            text_mask[i] = bool(has_text)

            transcript = ""
            if has_text:
                transcript = str(c.get("transcript", "")).strip()


            if has_text:
                try:
                    tbert = bert_encode_one_cls(bert_tok, bert, transcript, device)
                    text_out["bert"][i] = cast_fp16(tbert)
                except Exception as e:
                    failed_rows.append((video_id, "N/A", "text_encode_bert", f"clip={clip_idx} {repr(e)}"))

                try:
                    t06 = qwen_encode_one(qwen06, transcript, dim06)
                    t8 = qwen_encode_one(qwen8, transcript, dim8)
                    text_out["qwen_0.6"][i] = cast_fp16(t06)
                    text_out["qwen_8"][i] = cast_fp16(t8)
                except Exception as e:
                    failed_rows.append((video_id, "N/A", "text_encode_qwen", f"clip={clip_idx} {repr(e)}"))


            for b in budgets:
                words_keep = load_ocr_words_keep(raw_cdir, b)
                items = [w.strip() for w in words_keep if w.strip()]
                has_ocr = (len(items) > 0)
                ocr_mask[b][i] = bool(has_ocr)

                if not has_ocr:
                    continue

     
                try:
                    mat = bert_encode_cls_batch(bert_tok, bert, items, device)  
                    if mat.shape[0] > 0:
                        ocr_out[("bert", b)][i] = cast_fp16(mat.mean(dim=0))
                    else:
                        failed_rows.append((video_id, "N/A", "ocr_empty_after_filter", f"clip={clip_idx} budget={b}"))
                except Exception as e:
                    failed_rows.append((video_id, "N/A", "ocr_encode_bert", f"clip={clip_idx} budget={b} {repr(e)}"))

                try:
                    m06 = qwen_encode_items(qwen06, items, batch_size=int(args.ocr_batch_qwen), dim=dim06)
                    m8 = qwen_encode_items(qwen8, items, batch_size=int(args.ocr_batch_qwen), dim=dim8)
                    if m06.shape[0] > 0:
                        ocr_out[("qwen_0.6", b)][i] = cast_fp16(m06.mean(dim=0))
                    if m8.shape[0] > 0:
                        ocr_out[("qwen_8", b)][i] = cast_fp16(m8.mean(dim=0))
                except Exception as e:
                    failed_rows.append((video_id, "N/A", "ocr_encode_qwen", f"clip={clip_idx} budget={b} {repr(e)}"))


        out: Dict[str, Any] = {
            "video_id": video_id,
            "num_clips": int(T),
            "whisper": whisper_out,
            "whisper_mask": whisper_mask,
            "text_mask": text_mask,   
        }

        for b in budgets:
            out[f"vit_{b}"] = vit_out[b]
            out[f"vit_mask_{b}"] = vit_mask[b]

        for m in text_models:
            out[f"text_{m}"] = text_out[m]

        for b in budgets:
            out[f"ocr_mask_{b}"] = ocr_mask[b]  

        for m in text_models:
            for b in budgets:
                out[f"ocr_{m}_{b}"] = ocr_out[(m, b)]


        out.update(rationale_out)
        out.update(rationale_mask_out) 

        try:
            torch.save(out, str(out_path))
            ok_cnt += 1
        except Exception as e:
            failed_rows.append((video_id, "N/A", "save_pt", repr(e)))
            fail_cnt += 1

        pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)

    if failed_rows:
        write_failed_tsv(failed_tsv, failed_rows)
        print(f"[WARN] failures={len(failed_rows)} written to: {failed_tsv}")
    else:
        print("[INFO] no failures")

    print(f"[DONE] elapsed_sec={time.time() - t0:.2f}")
    print(f"[OUT] out_root={out_root}")
    if args.num_shards > 1:
        print(f"[SHARD] shard_id={args.shard_id} / num_shards={args.num_shards}")


if __name__ == "__main__":
    main()

