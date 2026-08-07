import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

CLIP_RE = re.compile(r"clip_(\d+)")
FRAME_RE = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)


def parse_int(pattern: re.Pattern, s: str, default: int = 10**9) -> int:
    m = pattern.search(s)
    return int(m.group(1)) if m else default


def list_clips(video_dir: Path) -> List[Path]:
    clips = [p for p in video_dir.iterdir() if p.is_dir() and p.name.startswith("clip_")]
    return sorted(clips, key=lambda p: parse_int(CLIP_RE, p.name))


def list_frames_in_budget(clip_dir: Path, budget_dir: str) -> List[Path]:
    fdir = clip_dir / "frames" / budget_dir
    if not fdir.exists():
        return []
    frames = sorted(
        fdir.glob("frame_*.jpg"),
        key=lambda p: parse_int(FRAME_RE, p.name)
    )
    return frames


def collect_all_frames(
    video_dir: Path,
    budget_dir: str,
    fallback_budgets: Optional[List[str]] = None,
) -> Tuple[List[Path], Optional[str]]:

    clips = list_clips(video_dir)
    if not clips:
        return [], None

    budgets_to_try = [budget_dir]
    if fallback_budgets:
        budgets_to_try += [b for b in fallback_budgets if b != budget_dir]

    for b in budgets_to_try:
        all_frames: List[Path] = []
        for c in clips:
            all_frames.extend(list_frames_in_budget(c, b))
        if all_frames:
            
            def global_key(p: Path):
                clip_name = p.parents[2].name  
                clip_id = parse_int(CLIP_RE, clip_name)
                frame_id = parse_int(FRAME_RE, p.name)
                return (clip_id, frame_id)

            all_frames = sorted(all_frames, key=global_key)
            return all_frames, b

    return [], None


def uniform_subsample(paths: List[Path], k: int) -> List[Path]:
    n = len(paths)
    if n == 0:
        return []
    if n <= k:
        return paths
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    return [paths[i] for i in idx]


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)



def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_features_dir", type=str, help="Path to raw_features (contains per-video folders)")
    ap.add_argument("--out_index_dir", type=str, help="Output dir to store per-video json index: <out>/<video_id>.json")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--budget_dir", type=str, default="frame_120",
                    help="Which budget folder under clip_xxx/frames/ , frame_100 for my model, frame_120 for baselines")
    ap.add_argument("--fallback", action="store_true",
                    help="If set, fallback to other budgets if budget_dir missing/empty")

    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing per-video json (default: skip existing)")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Process at most N videos (debug). -1 means all.")
    ap.add_argument("--video_id", type=str, default="",
                    help="If set, only process this one video_id")

    ap.add_argument("--num_shards", type=int, default=1,
                    help="If >1, split videos by index mod num_shards")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Which shard to process [0..num_shards-1]")

    return ap.parse_args()


def main():
    args = build_args()

    raw_features = Path(args.raw_features_dir)
    out_root = Path(args.out_index_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards-1]")

    if args.fallback:
        fallback_budgets = ["frame_120", "frame_100", "frame_80", "frame_60", "frame_40"]
    else:
        fallback_budgets = None

    if args.video_id:
        video_dirs = [raw_features / args.video_id]
        if not video_dirs[0].exists():
            raise FileNotFoundError(f"video_id not found: {video_dirs[0]}")
    else:
        video_dirs = [p for p in raw_features.iterdir() if p.is_dir()]
        video_dirs = sorted(video_dirs, key=lambda p: p.name)

  
        if args.num_shards > 1:
            video_dirs = [v for i, v in enumerate(video_dirs) if (i % args.num_shards) == args.shard_id]


        if args.limit and args.limit > 0:
            video_dirs = video_dirs[:args.limit]

    ok = 0
    skipped = 0
    failed = 0

    pbar = tqdm(video_dirs, total=len(video_dirs), desc="Indexing frames", dynamic_ncols=True)

    for idx, vdir in enumerate(pbar, start=1):
        vid = vdir.name
        pbar.set_postfix(ok=ok, skip=skipped, fail=failed)

        out_path = out_root / f"{vid}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            pbar.set_postfix(ok=ok, skip=skipped, fail=failed)
            continue

        try:
            all_frames, used_budget = collect_all_frames(
                vdir,
                budget_dir=args.budget_dir,
                fallback_budgets=fallback_budgets
            )
            if not all_frames:
                skipped += 1
                pbar.set_postfix(ok=ok, skip=skipped, fail=failed)
                continue

            selected = uniform_subsample(all_frames, args.k)

            record = {
                "video_id": vid,
                "budget_dir": used_budget,
                "k": int(args.k),
                "sampling": "uniform",
                "frames": [str(p) for p in selected],  
            }

            atomic_write_json(out_path, record)
            ok += 1
            pbar.set_postfix(ok=ok, skip=skipped, fail=failed)

        except Exception as e:
            failed += 1
            pbar.set_postfix(ok=ok, skip=skipped, fail=failed)
            tqdm.write(f"[FAIL] {vid}: {e}")

    pbar.close()
    print(f"[DONE] total={len(video_dirs)} ok={ok} skip={skipped} fail={failed}")
    print(f"[DONE] out_index_dir={out_root}")


if __name__ == "__main__":
    main()