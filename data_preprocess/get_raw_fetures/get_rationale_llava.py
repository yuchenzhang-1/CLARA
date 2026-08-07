import argparse
import ast
import json
import os
import re
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration


def build_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--frame_index_root",
        type=str,
        help="Root dir containing per-video frame index json: <video_id>.json",
    )

    ap.add_argument(
        "--transcripts_root",
        type=str,
        help="Directory containing per-video transcript JSON: <video_id>.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        help="Output directory for rationale JSON: <video_id>_rationale.json",
    )

    ap.add_argument("--video_id", type=str, default="", help="If set, only process this one video_id.")

    ap.add_argument("--num_shards", type=int, default=1, help="Split videos into shards by index mod num_shards.")
    ap.add_argument("--shard_id", type=int, default=0, help="Which shard to process [0..num_shards-1].")

    ap.add_argument("--limit", type=int, default=-1, help="Process at most N videos (after sharding). -1 means no limit.")
    ap.add_argument("--num_frames", type=int, default=20)


    ap.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

  
    ap.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Try to load with this attention implementation (default flash_attention_2).",
    )

  
    ap.add_argument(
        "--image_mode",
        type=str,
        default="stitch",
        choices=["separate", "stitch"],
        help="separate=pass N images; stitch=grid into one image.",
    )
    ap.add_argument("--grid_cols", type=int, default=5)
    ap.add_argument("--grid_pad", type=int, default=2)
    ap.add_argument("--grid_bg", type=int, default=0)

    ap.add_argument(
        "--send_images_each_step",
        action="store_true",
        default=False,
        help="If set, Step A/B both send images. Default: only Step A sends images.",
    )

  
    ap.add_argument("--max_new_tokens_stepa", type=int, default=2048)
    ap.add_argument("--max_new_tokens_stepb", type=int, default=2048)

    ap.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling (default off).")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--skip_existing", action="store_true", default=True, help="If output exists, skip (default True).")
    ap.add_argument("--no_skip_existing", action="store_true", default=False, help="Disable skipping existing outputs.")

    ap.add_argument(
        "--log_path",
        type=str,
        default="",
        help="Optional JSONL log path. If empty, no log.",
    )

    ap.add_argument("--dump_prompts", action="store_true", default=False,
                    help="If set, save step prompts into output JSON for debugging.")

    return ap.parse_args()



def read_text_fields(transcript_path: Path) -> Tuple[str, str, str]:
    data = json.loads(transcript_path.read_text(encoding="utf-8", errors="ignore"))
    video_title = (data.get("title") or "").strip() or "N/A"
    video_description = (data.get("description") or "").strip() or "N/A"
    transcription = (data.get("text") or "").strip() or "N/A"
    return video_title, video_description, transcription

def append_jsonl(path_str: str, obj: Dict[str, Any]) -> None:
    if not path_str:
        return
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

def get_torch_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def frame_index_path(frame_index_root: Path, video_id: str) -> Path:
    return frame_index_root / f"{video_id}.json"

def load_frame_paths(frame_index_root: Path, video_id: str) -> Optional[List[str]]:
    p = frame_index_path(frame_index_root, video_id)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        frames = obj.get("frames", [])
        if isinstance(frames, list):
            frames = [str(x).strip() for x in frames if str(x).strip()]
            return frames if frames else None
        return None
    except Exception:
        return None

def load_images_from_paths(paths: List[str]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for fp in paths:
        try:
            imgs.append(Image.open(fp).convert("RGB"))
        except Exception as e:
            tqdm.write(f"[WARN] cannot open frame: {fp} | {e}")
    return imgs


def stitch_images_grid(images: List[Image.Image], cols: int = 5, pad: int = 2, bg: int = 0) -> Image.Image:
    if not images:
        raise ValueError("No images to stitch.")
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols

    ws = [im.size[0] for im in images]
    hs = [im.size[1] for im in images]
    tile_w = min(ws)
    tile_h = min(hs)

    resized = [im.resize((tile_w, tile_h), resample=Image.BICUBIC) for im in images]

    out_w = cols * tile_w + (cols - 1) * pad
    out_h = rows * tile_h + (rows - 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), color=(bg, bg, bg))

    for idx, im in enumerate(resized):
        r = idx // cols
        c = idx % cols
        x = c * (tile_w + pad)
        y = r * (tile_h + pad)
        canvas.paste(im, (x, y))

    return canvas


STEP_A_TARGET_JSON = {
    "objective_summary": "",
    "visual_description": "",
    "textual_description": "",
    "cross_modal_relation": "",
    "cross_modal_explanation": "",
    "contextually_important_elements": [],
}

def prompt_step_a_tagged(video_title: str, video_description: str, transcription: str) -> str:
    return f"""
Role: You are a professional video content verifier. Your task is to accurately summarize what the video shows and what the accompanying text says, in a objective and neutral way.

You may use:
1) 20 frames sampled uniformly in temporal order.
2) Video title (if available) and full transcription.

You need to:
(1) Describe what is visible in the frames.
(2) Summarize the key messages conveyed by the text.
(3) Provide an overall objective summary combining visual and textual evidence.
(4) Describe how the visuals and the text relate.

Return ONLY the following tagged format (no extra text before or after). Use the tag names exactly:

VISUAL_DESCRIPTION: Neutral description of the visible scenes, objects, actions, and symbols. Avoid guessing details that are not supported by the frames.
TEXTUAL_DESCRIPTION: Neutral summary of what the title and transcription. Avoid guessing details that are not supported by the text.
OBJECTIVE_SUMMARY: Overall summary combining visual and textual evidence. 
CROSS_MODAL_RELATION: One label from {{aligned, complementary, conflicting, unclear}}.
CROSS_MODAL_EXPLANATION: Briefly explain why that label fits, focusing on whether the visuals support, add context to, contradict, or do not clarify the text evidence.
CONTEXT_ELEMENTS:List of key contextual elements that help interpret the content (e.g., entities, groups, symbols).

Textual information:
VIDEO_TITLE: {video_title}
VIDEO_DESCRIPTION: {video_description}
TRANSCRIPTION: {transcription}
""".strip()


STEP_B_TARGET_JSON = {
    "final_decision": {"label": "", "explicitness": ""},
    "reasons": "",
    "confidence": "",
    "notes": "",
}

def prompt_step_b_tagged(step_a: dict, video_title: str, video_description: str, transcription: str) -> str:
    step_a_txt = json.dumps(step_a, ensure_ascii=False)
    return f"""
Role: You are a professional content verifier conducting hateful content analysis. Your task is to make a careful, evidence-based judgment using the information provided.

You may use:
1) 20 frames sampled uniformly in temporal order.
2) video title, video description, and full transcription.
3) The objective analysis of the video provided below.

You need to:
(1)Decide whether the video content is hateful or non-hateful.
(2)If hateful, indicate whether it is explicit or implicit.
(3)List the main reasons that support your decision, focusing on observable or stated evidence rather than speculation.

Return ONLY the following tagged format (no extra text before or after). Use the tag names exactly:

LABEL: <hate|non-hate>
EXPLICITNESS: <explicit|implicit|na>
REASONS:<reason 1>, <reason 2>, ...
CONFIDENCE: <high|medium|low>
NOTES: <any brief clarifications or assumptions, if needed>

Textual information:
VIDEO_TITLE: {video_title}
VIDEO_DESCRIPTION: {video_description}
TRANSCRIPTION: {transcription}

The objective analysis of the video:
{step_a_txt}
""".strip()


def _normalize_escaped_tags(text: str) -> str:
    return (text or "").replace("\\_", "_").replace("\\*", "*").replace("\\#", "#")

def _extract_block(text: str, headers: List[str], next_headers: List[str]) -> str:
    text = _normalize_escaped_tags(text)
    text = text.replace("```", "").strip()

    hdr_re = r"(?:%s)" % "|".join(re.escape(h) for h in headers)
    if next_headers:
        nxt_re = r"(?=^(?:%s)\s*$|\Z)" % "|".join(re.escape(h) for h in next_headers)
    else:
        nxt_re = r"(?=\Z)"

    pattern = rf"^\s*{hdr_re}\s*(?:\n|$)(.*?){nxt_re}"
    m = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if m:
        return m.group(1).strip()

    pattern2 = rf"^\s*{hdr_re}\s*(.*?){nxt_re}"
    m2 = re.search(pattern2, text, flags=re.MULTILINE | re.DOTALL)
    return m2.group(1).strip() if m2 else ""

def _normalize_label(x: str) -> str:
    x = (x or "").strip().lower().strip("<>").strip()
    mapping = {
        "non-hateful": "non-hate",
        "nonhateful": "non-hate",
        "not hateful": "non-hate",
        "non hate": "non-hate",
        "hateful": "hate",
    }
    return mapping.get(x, x)

def parse_tagged_step_a(text: str) -> dict:
    out = dict(STEP_A_TARGET_JSON)
    text = _normalize_escaped_tags(text).replace("```", "").strip()

    headers_map = {
        "VISUAL_DESCRIPTION": ["VISUAL_DESCRIPTION:", "**VISUAL_DESCRIPTION**"],
        "TEXTUAL_DESCRIPTION": ["TEXTUAL_DESCRIPTION:", "**TEXTUAL_DESCRIPTION**"],
        "OBJECTIVE_SUMMARY": ["OBJECTIVE_SUMMARY:", "**OBJECTIVE_SUMMARY**"],
        "CROSS_MODAL_RELATION": ["CROSS_MODAL_RELATION:", "**CROSS_MODAL_RELATION**"],
        "CROSS_MODAL_EXPLANATION": ["CROSS_MODAL_EXPLANATION:", "**CROSS_MODAL_EXPLANATION**"],
        "CONTEXT_ELEMENTS": ["CONTEXT_ELEMENTS:", "**CONTEXT_ELEMENTS**"],
    }
    order = ["VISUAL_DESCRIPTION", "TEXTUAL_DESCRIPTION", "OBJECTIVE_SUMMARY",
             "CROSS_MODAL_RELATION", "CROSS_MODAL_EXPLANATION", "CONTEXT_ELEMENTS"]

    def later_headers(i: int) -> List[str]:
        hs: List[str] = []
        for j in range(i + 1, len(order)):
            hs.extend(headers_map[order[j]])
        return hs

    out["visual_description"] = _extract_block(text, headers_map["VISUAL_DESCRIPTION"], later_headers(0))
    out["textual_description"] = _extract_block(text, headers_map["TEXTUAL_DESCRIPTION"], later_headers(1))
    out["objective_summary"] = _extract_block(text, headers_map["OBJECTIVE_SUMMARY"], later_headers(2))
    out["cross_modal_relation"] = _extract_block(text, headers_map["CROSS_MODAL_RELATION"], later_headers(3)).strip().lower()
    out["cross_modal_explanation"] = _extract_block(text, headers_map["CROSS_MODAL_EXPLANATION"], later_headers(4))
    ctx_raw = _extract_block(text, headers_map["CONTEXT_ELEMENTS"], [])

    ctx_list: List[str] = []
    if ctx_raw:
        s = ctx_raw.strip()
        if s.startswith("["):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    ctx_list = [str(x).strip() for x in val if str(x).strip()]
            except Exception:
                ctx_list = []
        else:
            lines = [ln.strip("-•* \t") for ln in s.splitlines()]
            ctx_list = [ln for ln in lines if ln]
    out["contextually_important_elements"] = ctx_list

    allowed = {"aligned", "complementary", "conflicting", "unclear"}
    if out["cross_modal_relation"] not in allowed:
        out["cross_modal_relation"] = "unclear"
    return out

def parse_tagged_step_b(text: str) -> dict:
    out = json.loads(json.dumps(STEP_B_TARGET_JSON))
    text = _normalize_escaped_tags(text).replace("```", "").strip()

    headers = {
        "LABEL": ["LABEL:", "**LABEL**"],
        "EXPLICITNESS": ["EXPLICITNESS:", "**EXPLICITNESS**"],
        "REASONS": ["REASONS:", "**REASONS**"],
        "CONFIDENCE": ["CONFIDENCE:", "**CONFIDENCE**"],
        "NOTES": ["NOTES:", "**NOTES**"],
    }
    order = ["LABEL", "EXPLICITNESS", "REASONS", "CONFIDENCE", "NOTES"]

    def later(i: int) -> List[str]:
        hs: List[str] = []
        for j in range(i + 1, len(order)):
            hs.extend(headers[order[j]])
        return hs

    label_blk = _extract_block(text, headers["LABEL"], later(0))
    exp_blk = _extract_block(text, headers["EXPLICITNESS"], later(1))
    reasons = _extract_block(text, headers["REASONS"], later(2))
    conf_blk = _extract_block(text, headers["CONFIDENCE"], later(3))
    notes = _extract_block(text, headers["NOTES"], [])

    label = _normalize_label(label_blk.splitlines()[0].strip() if label_blk else "")
    exp = (exp_blk.splitlines()[0].strip().lower().strip("<>").strip() if exp_blk else "")
    conf = (conf_blk.splitlines()[0].strip().lower().strip("<>").strip() if conf_blk else "")

    if label not in {"hate", "non-hate"}:
        label = ""
    if exp not in {"explicit", "implicit", "na"}:
        exp = "na" if label == "non-hate" else ""
    if conf not in {"high", "medium", "low"}:
        conf = ""

    if label == "non-hate":
        exp = "na"

    out["final_decision"]["label"] = label
    out["final_decision"]["explicitness"] = exp
    out["reasons"] = reasons.strip()
    out["confidence"] = conf
    out["notes"] = notes.strip()
    return out


@torch.inference_mode()
def llava_generate_with_images(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    images: List[Image.Image],
    prompt_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    if not images:
        raise ValueError("llava_generate_with_images() got empty images list.")

    content = [{"type": "image"} for _ in range(len(images))]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    img_in = images[0] if len(images) == 1 else images
    inputs = processor(images=img_in, text=prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    prefix_len = inputs["input_ids"].shape[1]

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample)}
    if do_sample:
        gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})

    output = model.generate(**inputs, **gen_kwargs)
    gen_only = output[0][prefix_len:]
    out = processor.decode(gen_only, skip_special_tokens=True)
    return (out or "").strip()


@torch.inference_mode()
def llava_generate_text_only(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    prompt_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prefix_len = inputs["input_ids"].shape[1]

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample)}
    if do_sample:
        gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})

    output = model.generate(**inputs, **gen_kwargs)
    gen_only = output[0][prefix_len:]
    out = processor.decode(gen_only, skip_special_tokens=True)
    return (out or "").strip()


def main():
    args = build_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards-1]")

    frame_index_root = Path(args.frame_index_root)
    transcripts_root = Path(args.transcripts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_existing = args.skip_existing and (not args.no_skip_existing)

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch_dtype = get_torch_dtype(args.dtype)

    append_jsonl(args.log_path, {
        "event": "run_start",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_index_root": str(frame_index_root),
        "transcripts_root": str(transcripts_root),
        "out_dir": str(out_dir),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "device": str(device),
        "attn_impl_requested": args.attn_impl,
        "num_frames": args.num_frames,
        "image_mode": args.image_mode,
        "send_images_each_step": args.send_images_each_step,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "limit": args.limit,
        "skip_existing": skip_existing,
        "dump_prompts": args.dump_prompts,
    })

    processor = AutoProcessor.from_pretrained(args.model_id)

    model: Optional[LlavaForConditionalGeneration] = None
    load_err: Optional[str] = None
    used_attn: Optional[str] = None
    for attn_impl in [args.attn_impl, "sdpa", "eager"]:
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            )
            used_attn = attn_impl
            if attn_impl != args.attn_impl:
                append_jsonl(args.log_path, {"event": "attn_fallback", "from": args.attn_impl, "to": attn_impl})
            load_err = None
            break
        except Exception as e:
            load_err = str(e)
            model = None
            used_attn = None

    if model is None:
        raise RuntimeError(f"Failed to load model with attn_impl={args.attn_impl} (and fallbacks). Last error: {load_err}")

    model.to(device)
    model.eval()


    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = sorted([p.stem for p in transcripts_root.glob("*.json")])
        if args.num_shards > 1:
            video_ids = [vid for i, vid in enumerate(video_ids) if (i % args.num_shards) == args.shard_id]
        if args.limit and args.limit > 0:
            video_ids = video_ids[:args.limit]

    ok = skip = fail = 0
    pbar = tqdm(video_ids, total=len(video_ids), desc="Processing", dynamic_ncols=True)

    for vid in pbar:
        pbar.set_postfix(ok=ok, skip=skip, fail=fail)

        transcript_path = transcripts_root / f"{vid}.json"
        out_path = out_dir / f"{vid}_rationale.json"

        if skip_existing and out_path.exists():
            skip += 1
            continue
        if not transcript_path.exists():
            skip += 1
            continue

        try:

            frame_paths = load_frame_paths(frame_index_root, vid)
            if not frame_paths:
                skip += 1
                append_jsonl(args.log_path, {"event": "skip", "video_id": vid, "reason": "missing_or_empty_frame_index",
                                             "index_path": str(frame_index_path(frame_index_root, vid))})
                continue

            frame_paths_k = frame_paths[: int(args.num_frames)]
            images = load_images_from_paths(frame_paths_k)
            if not images:
                skip += 1
                append_jsonl(args.log_path, {"event": "skip", "video_id": vid, "reason": "no_images_loaded",
                                             "n_paths": len(frame_paths_k)})
                continue

     
            if args.image_mode == "stitch":
                grid = stitch_images_grid(images, cols=args.grid_cols, pad=args.grid_pad, bg=args.grid_bg)
                model_images = [grid]
            else:
                model_images = images

            title, desc, transcription = read_text_fields(transcript_path)

            # Step A (always images)
            prompt_a = prompt_step_a_tagged(title, desc, transcription)
            raw_a = llava_generate_with_images(
                model=model,
                processor=processor,
                images=model_images,
                prompt_text=prompt_a,
                max_new_tokens=args.max_new_tokens_stepa,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                dtype=torch_dtype,
            )
            step_a = parse_tagged_step_a(raw_a)

            prompt_b = prompt_step_b_tagged(step_a, title, desc, transcription)
            if args.send_images_each_step:
                raw_b = llava_generate_with_images(
                    model=model,
                    processor=processor,
                    images=model_images,
                    prompt_text=prompt_b,
                    max_new_tokens=args.max_new_tokens_stepb,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=torch_dtype,
                )
            else:
                raw_b = llava_generate_text_only(
                    model=model,
                    processor=processor,
                    prompt_text=prompt_b,
                    max_new_tokens=args.max_new_tokens_stepb,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )

            step_b_parsed = parse_tagged_step_b(raw_b)
            final_decision = {
                "label": step_b_parsed["final_decision"]["label"],
                "explicitness": step_b_parsed["final_decision"]["explicitness"],
                "reasons": step_b_parsed.get("reasons", ""),
                "confidence": step_b_parsed.get("confidence", ""),
                "notes": step_b_parsed.get("notes", ""),
            }

            record: Dict[str, Any] = {
                "video_id": vid,
                "objective_description": step_a,
                "final_decision": final_decision,
                "raw": {"step_a": raw_a, "step_b": raw_b},
                "model_name": args.model_id,
                "attn_impl_requested": args.attn_impl,
                "attn_impl_loaded": used_attn,
                "image_mode": args.image_mode,
                "send_images_each_step": args.send_images_each_step,
                "num_frames": args.num_frames,
                "frame_index_root": str(frame_index_root),
            }
            if args.dump_prompts:
                record["debug_prompts"] = {"step_a": prompt_a, "step_b": prompt_b}

            atomic_write_json(out_path, record)
            ok += 1

        except Exception as e:
            fail += 1
            tb = traceback.format_exc(limit=80)
            atomic_write_json(out_path, {"video_id": vid, "error": str(e), "traceback": tb})
            append_jsonl(args.log_path, {"event": "fail", "video_id": vid, "error": str(e)})

    append_jsonl(args.log_path, {
        "event": "run_end",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {"ok": ok, "skip": skip, "fail": fail, "total": len(video_ids)},
    })
    print(f"[DONE] total={len(video_ids)} ok={ok} skip={skip} fail={fail}")
    print(f"[DONE] out_dir={out_dir}")


if __name__ == "__main__":
    main()

