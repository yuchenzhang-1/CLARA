import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def build_args():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--frame_index_root", type=str, 
                    help="Root dir containing per-video frame index json: <video_id>.json")

    ap.add_argument("--transcripts_root", type=str,  help="Directory containing per-video transcript JSON: <video_id>.json")

    ap.add_argument("--out_dir", type=str, help="Output directory for rationale JSON: <video_id>_rationale.json")

    ap.add_argument("--video_id", type=str, default="",
                    help="If set, only process this one video_id.")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Process at most N videos (for testing). -1 means no limit.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="If >1, split the videos into several shards.")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Process which shard of the videos.")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2") 
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--num_frames", type=int, default=20)

    ap.add_argument("--max_new_tokens_stepa", type=int, default=2048)
    ap.add_argument("--max_new_tokens_stepb", type=int, default=2048)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="If output exists, skip processing (default True).")
    ap.add_argument("--no_skip_existing", action="store_true", default=False,
                    help="Disable skipping existing outputs.")

    ap.add_argument("--log_path", type=str, default="",
                    help="Path to JSONL log file. If empty, save to out_dir/run_*.jsonl")

    return ap.parse_args()



def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def get_torch_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32

def read_text_fields(transcript_path: Path):
    data = json.loads(transcript_path.read_text(encoding="utf-8", errors="ignore"))
    video_title = (data.get("title") or "").strip() or "N/A"
    video_description = (data.get("description") or "").strip() or "N/A"
    transcription = (data.get("text") or "").strip() or "N/A"
    return video_title, video_description, transcription

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
Role: You are a professional video content verifier. Your task is to accurately document what the video shows and what the accompanying text says, in a objective and neutral way.

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

import ast
import re as _re_stepa

def parse_tagged_step_a(text: str) -> dict:
    out = dict(STEP_A_TARGET_JSON)
    text = text.replace("```", "").strip()

    def get_line(prefix: str) -> str:
        m = _re_stepa.search(rf"^{_re_stepa.escape(prefix)}\s*(.*)$", text, flags=_re_stepa.MULTILINE)
        return m.group(1).strip() if m else ""

    out["visual_description"] = get_line("VISUAL_DESCRIPTION:")
    out["textual_description"] = get_line("TEXTUAL_DESCRIPTION:")
    out["objective_summary"] = get_line("OBJECTIVE_SUMMARY:")
    out["cross_modal_relation"] = get_line("CROSS_MODAL_RELATION:").lower().strip()
    out["cross_modal_explanation"] = get_line("CROSS_MODAL_EXPLANATION:")

    def parse_list(prefix: str) -> list:
        raw = get_line(prefix).strip()
        if not raw:
            return []
        try:
            val = ast.literal_eval(raw)
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            return []
        except Exception:
            raw2 = raw.strip("[]")
            items = [x.strip().strip('"').strip("'") for x in raw2.split(",")]
            return [x for x in items if x]

    out["contextually_important_elements"] = parse_list("CONTEXT_ELEMENTS:")

    allowed = {"aligned", "complementary", "conflicting", "unclear"}
    if out["cross_modal_relation"] not in allowed:
        out["cross_modal_relation"] = "unclear"

    return out


STEP_B_TARGET_JSON = {
  "final_decision": {
    "label": "",
    "explicitness": ""
  },
  "reasons": "",
  "confidence": "",
  "notes": ""
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

import json as _json_stepb

def parse_tagged_step_b(text: str) -> dict:
    out = _json_stepb.loads(_json_stepb.dumps(STEP_B_TARGET_JSON))
    text = text.replace("```", "").strip()
    lines = [ln.rstrip() for ln in text.splitlines()]

    def get_single_line(prefix: str) -> str:
        for ln in lines:
            if ln.strip().startswith(prefix):
                return ln.split(":", 1)[1].strip() if ":" in ln else ""
        return ""

    def get_block_after(prefix: str, stop_prefixes: set) -> str:
        collecting = False
        buf = []
        for ln in lines:
            s = ln.strip()
            if not collecting:
                if s.startswith(prefix):
                    collecting = True
                    after = s.split(":", 1)[1].strip() if ":" in s else ""
                    if after:
                        buf.append(after)
                continue
            else:
                for sp in stop_prefixes:
                    if s.startswith(sp):
                        collecting = False
                        break
                if not collecting:
                    break
                buf.append(ln)
        return "\n".join([b.rstrip() for b in buf]).strip()

    label = get_single_line("LABEL:").lower().strip()
    explicitness = get_single_line("EXPLICITNESS:").lower().strip()
    confidence = get_single_line("CONFIDENCE:").lower().strip()
    notes = get_block_after("NOTES:", stop_prefixes={"LABEL:", "EXPLICITNESS:", "REASONS:", "CONFIDENCE:"})
    reasons_text = get_block_after("REASONS:", stop_prefixes={"LABEL:", "EXPLICITNESS:", "NOTES:", "CONFIDENCE:"})

    if label not in {"hate", "non-hate"}:
        label = ""
    if explicitness not in {"explicit", "implicit", "na"}:
        explicitness = "na" if label == "non-hate" else ""
    if confidence not in {"high", "medium", "low"}:
        confidence = ""

    if label == "non-hate":
        explicitness = "na"

    out["final_decision"]["label"] = label
    out["final_decision"]["explicitness"] = explicitness
    out["reasons"] = reasons_text
    out["confidence"] = confidence
    out["notes"] = notes
    return out



@torch.inference_mode()
def generate_text(model, processor, images: List[Image.Image], prompt: str,
                  max_new_tokens: int, temperature: float, top_p: float) -> str:
    messages = [{
        "role": "user",
        "content": ([{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}])
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
    )

    gen_trim = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
    text = processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()



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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_path) if args.log_path else (out_dir / f"rationale_run_{ts}.jsonl")

    append_jsonl(log_path, {
        "event": "run_start",
        "timestamp": ts,
        "frame_index_root": str(frame_index_root),
        "transcripts_root": str(transcripts_root),
        "out_dir": str(out_dir),
        "model_name": args.model_name,
        "attn_impl": args.attn_impl,
        "dtype": args.dtype,
        "num_frames": args.num_frames,
        "max_new_tokens": {"stepa": args.max_new_tokens_stepa, "stepb": args.max_new_tokens_stepb},
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "skip_existing": skip_existing,
    })
    print(f"[INFO] Log: {log_path}")

    torch_dtype = get_torch_dtype(args.dtype)
    print(f"[INFO] Loading model: {args.model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.attn_impl,
    )
    processor = AutoProcessor.from_pretrained(args.model_name)


    if args.video_id:
        video_ids = [args.video_id]
        tpath = transcripts_root / f"{args.video_id}.json"
        if not tpath.exists():
            raise FileNotFoundError(f"transcript json not found: {tpath}")
    else:
        video_ids = sorted([p.stem for p in transcripts_root.glob("*.json")])
        if args.limit and args.limit > 0:
            video_ids = video_ids[:args.limit]
        if args.num_shards > 1:
            video_ids = [vid for i, vid in enumerate(video_ids) if (i % args.num_shards) == args.shard_id]

    total = len(video_ids)
    ok = 0
    skipped = 0
    failed = 0

    pbar = tqdm(video_ids, total=total, desc="Processing", dynamic_ncols=True)

    for idx, vid in enumerate(pbar, start=1):
        out_path = out_dir / f"{vid}_rationale.json"
        tpath = transcripts_root / f"{vid}.json"

        pbar.set_postfix(ok=ok, skip=skipped, fail=failed)

        def log_line(msg: str):
            tqdm.write(msg)

        if skip_existing and out_path.exists():
            skipped += 1
            append_jsonl(log_path, {"event": "skip", "video_id": vid, "reason": "output_exists", "out_path": str(out_path)})
            log_line(f"[{idx}/{total}] [SKIP] {vid}: output exists | ok={ok} skip={skipped} fail={failed}")
            continue

        if not tpath.exists():
            skipped += 1
            append_jsonl(log_path, {"event": "skip", "video_id": vid, "reason": "missing_transcript", "transcript_path": str(tpath)})
            log_line(f"[{idx}/{total}] [SKIP] {vid}: missing transcript | ok={ok} skip={skipped} fail={failed}")
            continue

        try:
            frame_paths = load_frame_paths(frame_index_root, vid)
            if not frame_paths:
                skipped += 1
                ipath = frame_index_path(frame_index_root, vid)
                append_jsonl(log_path, {"event": "skip", "video_id": vid, "reason": "missing_or_empty_frame_index", "index_path": str(ipath)})
                log_line(f"[{idx}/{total}] [SKIP] {vid}: missing/empty frame index | ok={ok} skip={skipped} fail={failed}")
                continue

            frame_paths_k = frame_paths[: int(args.num_frames)]
            images = load_images_from_paths(frame_paths_k)
            if not images:
                skipped += 1
                append_jsonl(log_path, {"event": "skip", "video_id": vid, "reason": "no_images_loaded", "n_paths": len(frame_paths_k)})
                log_line(f"[{idx}/{total}] [SKIP] {vid}: cannot load images | ok={ok} skip={skipped} fail={failed}")
                continue

            video_title, video_description, transcription = read_text_fields(tpath)

        except Exception as e:
            failed += 1
            append_jsonl(log_path, {"event": "fail", "video_id": vid, "stage": "load_inputs", "error": str(e)})
            log_line(f"[{idx}/{total}] [FAIL] {vid}: load_inputs error={e} | ok={ok} skip={skipped} fail={failed}")
            continue

        log_line(f"[{idx}/{total}] [RUN] {vid} | ok={ok} skip={skipped} fail={failed}")

        try:
            raw_a = generate_text(
                model, processor, images,
                prompt_step_a_tagged(video_title, video_description, transcription),
                max_new_tokens=args.max_new_tokens_stepa,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            step_a = parse_tagged_step_a(raw_a)

            raw_b = generate_text(
                model, processor, images,
                prompt_step_b_tagged(step_a, video_title, video_description, transcription),
                max_new_tokens=args.max_new_tokens_stepb,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            step_b_parsed = parse_tagged_step_b(raw_b)

            final_decision = {
                "label": step_b_parsed["final_decision"]["label"],
                "explicitness": step_b_parsed["final_decision"]["explicitness"],
                "reasons": step_b_parsed.get("reasons", ""),
                "confidence": step_b_parsed.get("confidence", ""),
                "notes": step_b_parsed.get("notes", ""),
            }

            record = {
                "video_id": vid,
                "objective_description": step_a,
                "final_decision": final_decision,
                "raw": {"step_a": raw_a, "step_b": raw_b},
            }

            write_json(out_path, record)
            ok += 1
            append_jsonl(log_path, {"event": "ok", "video_id": vid, "out_path": str(out_path)})
            log_line(f"[{idx}/{total}] [OK] {vid}: wrote {out_path.name} | ok={ok} skip={skipped} fail={failed}")

        except Exception as e:
            failed += 1
            tb = traceback.format_exc(limit=20)
            append_jsonl(log_path, {"event": "fail", "video_id": vid, "stage": "runtime", "error": str(e), "traceback": tb})
            write_json(out_path, {"video_id": vid, "error": str(e), "traceback": tb})
            log_line(f"[{idx}/{total}] [FAIL] {vid}: runtime error={e} | ok={ok} skip={skipped} fail={failed}")

    pbar.close()

    summary = {"total": total, "ok": ok, "skipped": skipped, "failed": failed}
    append_jsonl(log_path, {"event": "run_end", "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "summary": summary})

    print(f"[DONE] {summary}")
    print(f"[DONE] log saved to {log_path}")


if __name__ == "__main__":
    main()


