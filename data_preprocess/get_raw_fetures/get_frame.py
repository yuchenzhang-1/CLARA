import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import Pool

import cv2
from tqdm import tqdm




def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))



def center_of_bin_frame_indices(
    start: float,
    end: float,
    k: int,
    fps: float,
    max_frame_idx: int,
) -> List[int]:

    if k <= 0:
        return []
    dur = max(1e-9, end - start)
    idxs = []
    for i in range(k):
        t = start + (i + 0.5) / k * dur
        fi = int(round(t * fps))
        fi = clamp(fi, 0, max_frame_idx)
        idxs.append(fi)
    return idxs




def marker_ok(marker_path: Path, budgets: List[int]) -> bool:
    if not marker_path.exists():
        return False
    try:
        m = read_json(marker_path)
        done = m.get("budgets_done", {})
        for b in budgets:
            if str(b) not in done or done[str(b)].get("status") != "ok":
                return False
        return True
    except Exception:
        return False



def write_no_frames_placeholder(
    out_video_dir: Path,
    clip_idx: int,
    budget: int,
    reason: str,
    overwrite: bool,
) -> None:

    out_dir = out_video_dir / f"clip_{clip_idx:03d}" / "frames" / f"frame_{budget}"
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholder = out_dir / "_NO_FRAMES.json"

    if (not overwrite) and placeholder.exists():
        return

    write_json(placeholder, {
        "budget": int(budget),
        "clip_idx": int(clip_idx),
        "allocated_frames": 0,
        "reason": str(reason),
        "done": True,
    })




def process_one_video(args_tuple) -> Tuple[str, str, Optional[str]]:
    (
        video_id,
        mp4_path,
        clipinfo_path,
        out_root,
        budgets,
        overwrite,
        jpeg_quality,
    ) = args_tuple

    mp4_path = Path(mp4_path)
    clipinfo_path = Path(clipinfo_path)
    out_root = Path(out_root)

    out_video_dir = out_root / video_id
    out_video_dir.mkdir(parents=True, exist_ok=True)
    marker_path = out_video_dir / "_FRAME_DONE.json"

    if (not overwrite) and marker_ok(marker_path, budgets):
        return video_id, "skipped", None


    if not mp4_path.exists():
        return video_id, "failed", f"missing mp4: {mp4_path}"
    if not clipinfo_path.exists():
        return video_id, "failed", f"missing clipinfo: {clipinfo_path}"

    clipinfo = read_json(clipinfo_path)
    clips = clipinfo.get("clips", [])
    if not clips:
        return video_id, "failed", "no clips in clipinfo"

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return video_id, "failed", f"cannot open video: {mp4_path}"

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 30.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frame_idx = frame_count - 1 if frame_count > 0 else 10**12


        frame_to_tasks: Dict[int, List[Tuple[int, int, int]]] = {}
        expected_by_budget = {b: 0 for b in budgets}

        zero_frame_cases: Dict[int, List[int]] = {b: [] for b in budgets} 


        for b in budgets:
            b_str = str(b)
            for c in clips:
                clip_idx = int(c["clip_idx"])
                start = float(c["start"])
                end = float(c["end"])
                k = int(c.get("num_frames_allocated", {}).get(b_str, 0))


                if k <= 0:
                    zero_frame_cases[b].append(clip_idx)
                    continue

                expected_by_budget[b] += k

                idxs = center_of_bin_frame_indices(start, end, k, fps, max_frame_idx)
                for local_i, fi in enumerate(idxs):
                    frame_to_tasks.setdefault(fi, []).append((b, clip_idx, local_i))

        if overwrite:
            for c in clips:
                clip_idx = int(c["clip_idx"])
                for b in budgets:
                    d = out_video_dir / f"clip_{clip_idx:03d}" / "frames" / f"frame_{b}"
                    if d.exists():
                        
                        for fp in d.glob("frame_*.jpg"):
                            fp.unlink(missing_ok=True)
                       
                        nf = d / "_NO_FRAMES.json"
                        if nf.exists():
                            nf.unlink(missing_ok=True)

        for b in budgets:
            for clip_idx in zero_frame_cases[b]:
                write_no_frames_placeholder(
                    out_video_dir=out_video_dir,
                    clip_idx=clip_idx,
                    budget=b,
                    reason="allocated_0_frames",
                    overwrite=overwrite,
                )

     
        saved_by_budget = {b: 0 for b in budgets}
        imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        wanted = set(frame_to_tasks.keys())
        last_needed = max(wanted) if wanted else -1

        cur_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            tasks = frame_to_tasks.get(cur_idx)
            if tasks:
                for (b, clip_idx, local_i) in tasks:
                    out_dir = (
                        out_video_dir
                        / f"clip_{clip_idx:03d}"
                        / "frames"
                        / f"frame_{b}"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    nf = out_dir / "_NO_FRAMES.json"
                    if nf.exists():
                        nf.unlink(missing_ok=True)

                    out_path = out_dir / f"frame_{local_i:03d}.jpg"

                    if (not overwrite) and out_path.exists():
                        continue

                    cv2.imwrite(str(out_path), frame, imwrite_params)
                    saved_by_budget[b] += 1

            if cur_idx >= last_needed:
                break
            cur_idx += 1


       
        budgets_done = {}
        for b in budgets:
            budgets_done[str(b)] = {
                "status": "ok",
                "expected_frames": int(expected_by_budget[b]),
                "saved_frames": int(saved_by_budget[b]),
                "fps_used": float(fps),
                "num_zero_frame_clips": int(len(zero_frame_cases[b])), 
            }

        write_json(marker_path, {
            "video_id": video_id,
            "mp4": str(mp4_path),
            "clipinfo": str(clipinfo_path),
            "budgets": budgets,
            "budgets_done": budgets_done,
        })

        return video_id, "ok", None

    except Exception as e:
        err = repr(e)
        (out_video_dir / "_FRAME_ERROR.txt").write_text(err, encoding="utf-8")
        return video_id, "failed", err

    finally:
        cap.release()


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4_dir")
    ap.add_argument("--clipinfo_root")
    ap.add_argument("--video_id", type=str)
    ap.add_argument("--out_root")
    ap.add_argument("--budgets", default="40,60,80,100,120")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()

    mp4_dir = Path(args.mp4_dir)
    clipinfo_root = Path(args.clipinfo_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    budgets = [int(x) for x in args.budgets.split(",")]


    if args.video_id is not None:
        vid = args.video_id.strip()
        mp4_path = mp4_dir / f"{vid}.mp4"
        clipinfo_path = clipinfo_root / vid / "clipinfo.json"

        if not mp4_path.exists():
            raise FileNotFoundError(f"--video_id {vid} not found: {mp4_path}")
        if not clipinfo_path.exists():
            raise FileNotFoundError(f"--video_id {vid} clipinfo missing: {clipinfo_path}")

        video_ids = [vid]
    else:
        mp4_files = sorted(mp4_dir.glob("*.mp4"))
        video_ids = [p.stem for p in mp4_files]


    tasks = []
    for vid in video_ids:
        tasks.append((
            vid,
            str(mp4_dir / f"{vid}.mp4"),
            str(clipinfo_root / vid / "clipinfo.json"),
            str(out_root),
            budgets,
            bool(args.overwrite),
            int(args.jpeg_quality),
        ))

    failed_log: Dict[str, str] = {}
    ok = skipped = failed = 0

    with Pool(processes=args.num_workers) as pool:
        for vid, status, err in tqdm(
            pool.imap_unordered(process_one_video, tasks, chunksize=4),
            total=len(tasks),
            desc="Extracting frames",
            unit="video",
            dynamic_ncols=True,
        ):
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failed_log[vid] = err or "unknown error"

    if failed_log:
        write_json(out_root / "failed_videos.json", failed_log)

    print("\n========== SUMMARY ==========")
    print(f"Total  : {len(tasks)}")
    print(f"OK     : {ok}")
    print(f"Skipped: {skipped}")
    print(f"Failed : {failed}")
    print(f"Out    : {out_root}")


if __name__ == "__main__":
    main()