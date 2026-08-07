import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import torchaudio
from tqdm import tqdm



SILENCE_THRESHOLD = 1.0
SILENCE_TRANSCRIPT = "N/A"
FRAME_BUDGETS = [40, 60, 80, 100, 120]




def wav_duration_sec(wav_path: Path) -> Optional[float]:
    if not wav_path.exists():
        return None
    try:
        info = torchaudio.info(str(wav_path))
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        try:
            wav, sr = torchaudio.load(str(wav_path))
            return float(wav.shape[-1]) / float(sr)
        except Exception:
            return None



def clip_segments_to_duration(
    segments: List[Dict[str, Any]],
    total_duration: float
) -> Tuple[List[Dict[str, Any]], int, int]:
    kept = []
    truncated = 0
    dropped = 0

    for s in segments:
        if "start" not in s or "end" not in s:
            continue
        st = float(s["start"])
        ed = float(s["end"])
        if ed <= st:
            continue

        if st >= total_duration:
            dropped += 1
            continue

        if ed > total_duration:
            s2 = dict(s)
            s2["end"] = float(total_duration)
            kept.append(s2)
            truncated += 1
        else:
            kept.append(s)

    return kept, truncated, dropped




def build_updated_clips_no_merge_speech(
    segments: List[Dict[str, Any]],
    total_duration: float,
    silence_merge_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    clean = []
    for idx, s in enumerate(segments):
        if "start" not in s or "end" not in s:
            continue
        st, ed = float(s["start"]), float(s["end"])
        if ed <= st:
            continue

        seg_id = int(s["id"]) if "id" in s and isinstance(s["id"], (int, float)) else idx
        txt = str(s.get("text", "")).strip()
        clean.append({"seg_id": seg_id, "start": st, "end": ed, "text": txt})

    clean.sort(key=lambda x: x["start"])

    clips: List[Dict[str, Any]] = []
    prev_end = 0.0
    have_prev_nonsilent = False

    def add_silent(start: float, end: float):
        clips.append({
            "start": start,
            "end": end,
            "is_silent": True,
            "texts": [],
            "source_segment_ids": [],
        })

    def add_speech(start: float, end: float, text: str, seg_id: int):
        clips.append({
            "start": start,
            "end": end,
            "is_silent": False,
            "texts": [text] if text else [],
            "source_segment_ids": [seg_id],
        })

    for seg in clean:
        st, ed, txt, sid = seg["start"], seg["end"], seg["text"], seg["seg_id"]


        if st > prev_end:
            gap = st - prev_end
            if gap < silence_merge_threshold:
                if have_prev_nonsilent:
                    clips[-1]["end"] = st
                else:
                    st = prev_end
            else:
                add_silent(prev_end, st)

        add_speech(st, ed, txt, sid)
        have_prev_nonsilent = True
        prev_end = max(prev_end, ed)

  
    if total_duration > prev_end:
        gap = total_duration - prev_end
        if gap < silence_merge_threshold and have_prev_nonsilent:
            clips[-1]["end"] = total_duration
        else:
            add_silent(prev_end, total_duration)

  
    for i, c in enumerate(clips):
        c["clip_idx"] = i
        c["duration"] = round(float(c["end"]) - float(c["start"]), 6)
        c["transcript"] = SILENCE_TRANSCRIPT if c["is_silent"] else " ".join(c["texts"]).strip()

    return clips



def allocate_frames_one_budget(
    clips: List[Dict[str, Any]],
    total_frames: int
) -> List[int]:
    n = len(clips)
    if n == 0:
        return []
    if total_frames <= 0:
        return [0] * n

    durations = [max(0.0, float(c.get("duration", 0.0))) for c in clips]
    tot = sum(durations)


    if tot <= 0.0:
        alloc = [0] * n
        alloc[0] = total_frames
        return alloc


    raw = [total_frames * d / tot for d in durations]

    base = [int(x // 1) for x in raw]  # floor
    cur = sum(base)
    remain = total_frames - cur


    frac = [(raw[i] - base[i], i) for i in range(n)]
    frac.sort(reverse=True, key=lambda x: x[0])

    k = 0
    while remain > 0 and k < n:
        _, i = frac[k]
        base[i] += 1
        remain -= 1
        k += 1

    if sum(base) > total_frames:
        overflow = sum(base) - total_frames
        frac.sort(key=lambda x: x[0]) 
        k = 0
        while overflow > 0 and k < n:
            _, i = frac[k]
            if base[i] > 0:
                base[i] -= 1
                overflow -= 1
            k += 1

    return base



def allocate_frames_multi_budgets(
    clips: List[Dict[str, Any]],
    budgets: List[int],
    out_field: str = "num_frames_allocated"
) -> None:
    for c in clips:
        c[out_field] = {}

    for b in budgets:
        alloc = allocate_frames_one_budget(clips, total_frames=int(b))
        for i, n in enumerate(alloc):
            clips[i][out_field][str(b)] = int(n)


def process_one_json(in_json: Path, wav_dir: Path, out_root: Path) -> Tuple[bool, str]: 
    video_id = in_json.stem
    out_dir = out_root / video_id                      
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clipinfo.json"

    data = json.loads(in_json.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    wav_path = wav_dir / f"{video_id}.wav"             
    dur = wav_duration_sec(wav_path)

    if dur is None:
        if segments:
            dur = max(float(s["end"]) for s in segments if "end" in s)
        else:
            dur = 0.0
        duration_source = "segments_end_fallback"
        clipped_segments = segments
        n_trunc = 0
        n_drop = 0
    else:
        duration_source = "wav_duration"
        clipped_segments, n_trunc, n_drop = clip_segments_to_duration(segments, float(dur))

    seg_end_before = None
    if segments:
        seg_end_before = max(float(s["end"]) for s in segments if "end" in s)
    seg_end_after = None
    if clipped_segments:
        seg_end_after = max(float(s["end"]) for s in clipped_segments if "end" in s)

    clips = build_updated_clips_no_merge_speech(
        segments=clipped_segments,
        total_duration=float(dur),
        silence_merge_threshold=SILENCE_THRESHOLD,
    )

    allocate_frames_multi_budgets(clips, FRAME_BUDGETS, out_field="num_frames_allocated")

    zero_stats = {}
    for b in FRAME_BUDGETS:
        z = sum(1 for c in clips if int(c["num_frames_allocated"][str(b)]) == 0)
        zero_stats[str(b)] = int(z)

    payload = {
        "video_id": video_id,
        "source": {
            "whisper_json": str(in_json),
            "audio_wav": str(wav_path) if wav_path.exists() else None,
            "duration_source": duration_source,
        },
        "policy": {
            "silence_merge_threshold_sec": SILENCE_THRESHOLD,
            "silence_transcript": SILENCE_TRANSCRIPT,
            "frame_budgets": FRAME_BUDGETS,
            "speech_merge": "disabled",
            "leading_short_silence_merge": True,
            "wav_duration_is_authoritative": True,
            "segment_truncation_if_exceed_wav": True,
            "segment_drop_if_start_after_wav": True,
            "frame_allocation_policy": "proportional_to_duration_allow_zero",
            "require_each_clip_at_least_one_frame": False,
        },
        "diagnostics": {
            "wav_duration_sec": round(float(dur), 6),
            "last_segment_end_sec_before_clip": None if seg_end_before is None else round(float(seg_end_before), 6),
            "last_segment_end_sec_after_clip": None if seg_end_after is None else round(float(seg_end_after), 6),
            "segments_truncated": int(n_trunc),
            "segments_dropped": int(n_drop),
            "num_zero_frame_clips_by_budget": zero_stats,
        },
        "total_duration_sec": round(float(dur), 6),
        "num_clips": len(clips),
        "clips": clips,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, video_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str)
    ap.add_argument("--wav_dir", type=str)
    ap.add_argument("--out_root", type=str)
    ap.add_argument("--video_id", type=str, default=None, help="only process one <video_id>.json in in_dir")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    wav_dir = Path(args.wav_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.video_id:
        jf = in_dir / f"{args.video_id}.json"
        if not jf.exists():
            raise FileNotFoundError(f"--video_id {args.video_id} not found: {jf}")
        json_files = [jf]
    else:
        json_files = sorted(in_dir.glob("*.json"))

    print(f"[INFO] Found {len(json_files)} json files in {in_dir}")

    ok = 0
    for jf in tqdm(json_files, desc="Building clipinfo", unit="video", dynamic_ncols=True):
        try:
            process_one_json(jf, wav_dir=wav_dir, out_root=out_root) 
            ok += 1
        except Exception as e:
            vid = jf.stem
            out_dir = out_root / vid
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "clipinfo.ERROR.txt").write_text(str(e), encoding="utf-8")

    print(f"[DONE] ok={ok}/{len(json_files)} saved under {out_root}")


if __name__ == "__main__":
    main()