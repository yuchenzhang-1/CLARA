import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Manager
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
import whisper

AUDIO_EXTS = {".mp4", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


def list_audio_files(audio_dir: Path) -> List[Path]:
    return sorted([p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def sec_to_ts_srt(x: float) -> str:
    ms = int(round(x * 1000.0))
    hh = ms // 3600000
    ms %= 3600000
    mm = ms // 60000
    ms %= 60000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def sec_to_ts_vtt(x: float) -> str:
    ms = int(round(x * 1000.0))
    hh = ms // 3600000
    ms %= 3600000
    mm = ms // 60000
    ms %= 60000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def write_srt(segments: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    idx = 1
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{sec_to_ts_srt(start)} --> {sec_to_ts_srt(end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: List[Dict[str, Any]], out_path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"{sec_to_ts_vtt(start)} --> {sec_to_ts_vtt(end)}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def safe_json_load(path: Path) -> Tuple[bool, str]:

    try:
        with open(path, "r", encoding="utf-8") as f:
            _ = json.load(f)
        return True, "json_ok"
    except Exception as e:
        return False, f"json_bad:{repr(e)}"


def check_already_done(
    out_json: Path,
    out_srt: Path,
    out_vtt: Path,
    need_srt: bool,
    need_vtt: bool,
) -> Tuple[bool, str]:

    if not out_json.exists():
        missing = ["json"]
        if need_srt and not out_srt.exists():
            missing.append("srt")
        if need_vtt and not out_vtt.exists():
            missing.append("vtt")
        return False, "missing:" + ",".join(missing)

    ok, reason = safe_json_load(out_json)
    if not ok:
  
        missing = ["json_invalid"]
        if need_srt and not out_srt.exists():
            missing.append("srt")
        if need_vtt and not out_vtt.exists():
            missing.append("vtt")
        return False, "missing:" + ",".join(missing) + f" ({reason})"


    missing = []
    if need_srt and not out_srt.exists():
        missing.append("srt")
    if need_vtt and not out_vtt.exists():
        missing.append("vtt")

    if missing:
        return False, "missing:" + ",".join(missing)

    return True, "done"


def log_print(lock, msg: str) -> None:
    with lock:
        print(msg, flush=True)


def worker(
    rank: int,
    gpu_id: int,
    files: List[str],
    out_dir: str,
    model_name: str,
    language: str | None,
    task: str,
    fp16: bool,
    word_timestamps: bool,
    write_srt_flag: bool,
    write_vtt_flag: bool,
    overwrite: bool,
    verbose: bool,
    # shared
    total_done,
    total_ok,
    total_skip,
    total_fail,
    lock,
) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_print(lock, f"[RANK {rank}] Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

    model = whisper.load_model(model_name, device="cuda")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    failed_list: List[Tuple[str, str]] = []

    for f in files:
        p = Path(f)
        vid = p.stem

        out_json = out_path / f"{vid}.json"
        out_srt = out_path / f"{vid}.srt"
        out_vtt = out_path / f"{vid}.vtt"

        if not overwrite:
            done, why = check_already_done(out_json, out_srt, out_vtt, write_srt_flag, write_vtt_flag)
            if done:
                with lock:
                    total_skip.value += 1
                    total_done.value += 1
                log_print(lock, f"[RANK {rank}][SKIP] {vid} -> {why}")
                continue
            else:
                log_print(lock, f"[RANK {rank}][TODO] {vid} -> {why} (will process)")
        else:
            log_print(lock, f"[RANK {rank}][OVERWRITE] {vid} -> will re-process")

        try:
            result = model.transcribe(
                str(p),
                task=task,
                language=language,
                fp16=fp16,
                word_timestamps=word_timestamps,
                verbose=verbose,
            )

            payload = {
                "video_id": vid,
                "audio_path": str(p),
                "language": result.get("language", None),
                "task": task,
                "text": (result.get("text") or "").strip(),
                "segments": result.get("segments", []),
            }

            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            if write_srt_flag:
                write_srt(payload["segments"], out_srt)
            if write_vtt_flag:
                write_vtt(payload["segments"], out_vtt)

            with lock:
                total_ok.value += 1
                total_done.value += 1

            log_print(lock, f"[RANK {rank}][OK] {vid} -> wrote json"
                              f"{', srt' if write_srt_flag else ''}"
                              f"{', vtt' if write_vtt_flag else ''}")

        except Exception as e:
            err = repr(e)
            failed_list.append((p.name, err))
            with lock:
                total_fail.value += 1
                total_done.value += 1
            log_print(lock, f"[RANK {rank}][FAIL] {vid} -> {err}")

  
    log_dir = out_path / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if failed_list:
        with open(log_dir / f"rank{rank}_failed.txt", "w", encoding="utf-8") as f:
            for name, err in failed_list:
                f.write(f"{name}\t{err}\n")

    log_print(lock, f"[RANK {rank}] DONE.")


def split_round_robin(paths: List[Path], n: int) -> List[List[Path]]:
    buckets = [[] for _ in range(n)]
    for i, p in enumerate(paths):
        buckets[i % n].append(p)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--gpus", help="GPU ids to use, e.g. 0,1")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default=None, help="e.g. en, zh. default auto-detect")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--word_timestamps", action="store_true")
    ap.add_argument("--write_srt", action="store_true")
    ap.add_argument("--write_vtt", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpu_ids) < 1:
        raise ValueError("No GPUs specified")

    files = list_audio_files(audio_dir)
    if not files:
        print("[INFO] No audio files found.")
        return

    total = len(files)
    print(f"[INFO] Found {total} files. Using GPUs: {gpu_ids}")
    splits = split_round_robin(files, len(gpu_ids))

    out_dir.mkdir(parents=True, exist_ok=True)

    with Manager() as manager:
        # shared counters
        total_done = manager.Value("i", 0)
        total_ok = manager.Value("i", 0)
        total_skip = manager.Value("i", 0)
        total_fail = manager.Value("i", 0)
        lock = manager.Lock()

        procs: List[Process] = []
        for rank, (gpu_id, subset) in enumerate(zip(gpu_ids, splits)):
            subset_str = [str(p) for p in subset]
            p = Process(
                target=worker,
                args=(
                    rank,
                    gpu_id,
                    subset_str,
                    str(out_dir),
                    args.model,
                    args.language,
                    args.task,
                    bool(args.fp16),
                    bool(args.word_timestamps),
                    bool(args.write_srt),
                    bool(args.write_vtt),
                    bool(args.overwrite),
                    bool(args.verbose),
                    total_done,
                    total_ok,
                    total_skip,
                    total_fail,
                    lock,
                ),
            )
            p.start()
            procs.append(p)

        with tqdm(total=total, desc="Overall progress", unit="video") as pbar:
            last = 0
            while any(p.is_alive() for p in procs):
                cur = total_done.value
                if cur > last:
                    pbar.update(cur - last)
                    last = cur
                pbar.set_postfix({
                    "ok": total_ok.value,
                    "skip": total_skip.value,
                    "fail": total_fail.value
                })
                time.sleep(0.2)


            cur = total_done.value
            if cur > last:
                pbar.update(cur - last)
            pbar.set_postfix({
                "ok": total_ok.value,
                "skip": total_skip.value,
                "fail": total_fail.value
            })

        for p in procs:
            p.join()

        print("\n========== GLOBAL SUMMARY ==========")
        print(f"Total files : {total}")
        print(f"OK          : {total_ok.value}")
        print(f"Skipped     : {total_skip.value}")
        print(f"Failed      : {total_fail.value}")
        print(f"Output dir  : {out_dir}")
        print(f"Logs in     : {out_dir / '_logs'}")


if __name__ == "__main__":
    main()