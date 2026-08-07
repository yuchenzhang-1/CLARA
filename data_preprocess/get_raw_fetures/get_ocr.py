import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from paddleocr import PaddleOCR




def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def list_video_dirs(raw_root: Path) -> List[Path]:
    return sorted([d for d in raw_root.iterdir() if d.is_dir() and (d / "clipinfo.json").exists()])

def list_clip_dirs(video_dir: Path) -> List[Path]:
    clips = []
    for d in video_dir.iterdir():
        if d.is_dir() and re.match(r"clip_\d{3}$", d.name):
            clips.append(d)
    return sorted(clips, key=lambda x: int(x.name.split("_")[1]))

def list_frame_paths(frame_dir: Path) -> List[Path]:
    def key(p: Path) -> int:
        m = re.findall(r"\d+", p.stem)
        return int(m[0]) if m else 0
    return sorted(frame_dir.glob("frame_*.jpg"), key=key)



def parse_paddle_result_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    rec_texts = d.get("rec_texts", []) or []
    rec_scores = d.get("rec_scores", []) or []
    rec_polys = d.get("rec_polys", []) or []
    rec_boxes = d.get("rec_boxes", []) or []
    dt_polys = d.get("dt_polys", []) or []

    n = min(len(rec_texts), len(rec_scores))
    rec_texts = list(rec_texts)[:n]
    rec_scores = list(rec_scores)[:n]

    return {
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys,
        "rec_boxes": rec_boxes,
        "dt_polys": dt_polys,
    }

def filter_by_conf(texts: List[str], scores: List[float], conf_th: float):
    keep_t, keep_s = [], []
    for t, s in zip(texts, scores):
        try:
            sf = float(s)
        except Exception:
            continue
        if sf >= conf_th:
            keep_t.append(str(t))
            keep_s.append(sf)
    return keep_t, keep_s




def build_ocr() -> PaddleOCR:
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )



def process_one_clip(
    ocr: PaddleOCR,
    clip_dir: Path,
    conf_th: float,
    budgets: List[int],
    overwrite: bool,
) -> None:
    frames_root = clip_dir / "frames"
    out_dir = clip_dir / "ocr_text"
    ensure_dir(out_dir)
    out_path = out_dir / "ocr_clip.json"

    if nonempty(out_path) and (not overwrite):
        return

    clip_idx = int(clip_dir.name.split("_")[1])

    payload: Dict[str, Any] = {
        "clip_idx": clip_idx,
        "conf_th": float(conf_th),
        "budgets": {},
    }

    for b in budgets:
        bdir = frames_root / f"frame_{b}"
        if not bdir.exists():
            payload["budgets"][str(b)] = {
                "num_frames": 0,
                "frames": [],
                "clip_words_keep": [],
                "clip_scores_keep": [],
                "clip_words_all": [],
                "clip_scores_all": [],
                "note": f"missing {str(bdir)}",
            }
            continue

        frame_paths = list_frame_paths(bdir)
        img_list = [str(p) for p in frame_paths]

        frames_out = []
        clip_words_keep, clip_scores_keep = [], []
        clip_words_all, clip_scores_all = [], []

        if len(img_list) == 0:
            payload["budgets"][str(b)] = {
                "num_frames": 0,
                "frames": [],
                "clip_words_keep": [],
                "clip_scores_keep": [],
                "clip_words_all": [],
                "clip_scores_all": [],
            }
            continue

        results = ocr.predict(input=img_list)
        if len(results) != len(img_list):
            raise RuntimeError(f"predict returned {len(results)} results for {len(img_list)} images in {bdir}")

        for fp, res in zip(frame_paths, results):
            d = None
            if hasattr(res, "json") and callable(getattr(res, "json")):
                try:
                    d = res.json()
                except Exception:
                    d = None

            if d is None:
                tmp_dir = out_dir / "_tmp_json"
                ensure_dir(tmp_dir)
                res.save_to_json(save_path=str(tmp_dir))
                js = sorted(tmp_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                d = load_json(js[0]) if js else {}

            d = parse_paddle_result_dict(d)

            words_all = d["rec_texts"]
            scores_all = d["rec_scores"]
            words_keep, scores_keep = filter_by_conf(words_all, scores_all, conf_th)

            clip_words_all.extend(words_all)
            clip_scores_all.extend([float(x) for x in scores_all])
            clip_words_keep.extend(words_keep)
            clip_scores_keep.extend(scores_keep)

            frames_out.append({
                "frame_file": fp.name,
                "words_all": words_all,
                "scores_all": [float(x) for x in scores_all],
                "words_keep": words_keep,
                "scores_keep": [float(x) for x in scores_keep],
                "rec_polys": d.get("rec_polys", []),
                "rec_boxes": d.get("rec_boxes", []),
                "dt_polys": d.get("dt_polys", []),
            })

        payload["budgets"][str(b)] = {
            "num_frames": len(frame_paths),
            "frames": frames_out,
            "clip_words_keep": clip_words_keep,
            "clip_scores_keep": [float(x) for x in clip_scores_keep],
            "clip_words_all": clip_words_all,
            "clip_scores_all": [float(x) for x in clip_scores_all],
        }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# Main (dual tqdm)
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str)
    ap.add_argument("--conf_th", type=float, default=0.5)
    ap.add_argument("--frame_budgets", type=str, default="40,60,80,100,120")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log_dir", type=str, default=None)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    budgets = [int(x) for x in args.frame_budgets.split(",") if x.strip()]

    log_dir = Path(args.log_dir) if args.log_dir else (raw_root / "_logs_ocr")
    ensure_dir(log_dir)
    fail_tsv = log_dir / "ocr_failed.tsv"
    summary_json = log_dir / "ocr_summary.json"

    ocr = build_ocr()

    video_dirs = list_video_dirs(raw_root)
    if not video_dirs:
        print(f"[WARN] no videos found under {raw_root}")
        return

    failed: List[Tuple[str, str, str]] = []
    stats = {
        "videos_total": len(video_dirs),
        "clips_total": 0,
        "clips_ok": 0,
        "clips_skipped": 0,
        "clips_failed": 0,
        "conf_th": float(args.conf_th),
        "budgets": budgets,
    }

    pbar_v = tqdm(video_dirs, desc="OCR videos", unit="video", dynamic_ncols=True, mininterval=0.3)

    for vdir in pbar_v:
        vid = vdir.name
        clip_dirs = list_clip_dirs(vdir)
        stats["clips_total"] += len(clip_dirs)

        pbar_c = tqdm(
            clip_dirs,
            desc=f"OCR clips [{vid}]",
            unit="clip",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.3
        )

        for cdir in pbar_c:
            clip_name = cdir.name
            out_path = cdir / "ocr_text" / "ocr_clip.json"

            pbar_c.set_postfix({"cur": clip_name, "ok": stats["clips_ok"], "skip": stats["clips_skipped"], "fail": stats["clips_failed"]})

            if nonempty(out_path) and (not args.overwrite):
                stats["clips_skipped"] += 1
                continue

            try:
                process_one_clip(
                    ocr=ocr,
                    clip_dir=cdir,
                    conf_th=float(args.conf_th),
                    budgets=budgets,
                    overwrite=bool(args.overwrite),
                )
                stats["clips_ok"] += 1
            except Exception as e:
                stats["clips_failed"] += 1
                failed.append((vid, clip_name, repr(e)))

        pbar_v.set_postfix({"vid": vid, "ok": stats["clips_ok"], "skip": stats["clips_skipped"], "fail": stats["clips_failed"]})

    if failed:
        with fail_tsv.open("w", encoding="utf-8") as f:
            f.write("video_id\tclip\terror\n")
            for vid, clip, err in failed:
                f.write(f"{vid}\t{clip}\t{err}\n")

    summary_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[DONE] OCR finished. summary={summary_json}")
    if failed:
        print(f"[WARN] failures={len(failed)} logged to {fail_tsv}")


if __name__ == "__main__":
    main()
