import os
import glob
import subprocess
import shutil
from tqdm import tqdm



SRC_DIR = "..."
OUT_DIR = "..."

SR = 16000
MONO = True
MIN_BYTES = 10_000
LOG_PATH = os.path.join(OUT_DIR, "ffmpeg_errors.log")
MAX_PRINT_ERRORS = 5

def extract_wav(mp4_path: str, wav_path: str, sr: int = 16000, mono: bool = True) -> None:
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    
    tmp_path = wav_path + ".tmp.wav"  

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i", mp4_path,
        "-vn",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
    ]
    if mono:
        cmd += ["-ac", "1"]

    
    cmd += ["-f", "wav", tmp_path]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())

    os.replace(tmp_path, wav_path)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    print("[INFO] ffmpeg in python PATH =", ffmpeg_path)
    if ffmpeg_path is None:
        print("[FATAL] ffmpeg not found in PATH for this python environment.")
        return

    mp4_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.mp4")))
    print(f"[INFO] found {len(mp4_files)} mp4 files")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    ok = skipped = failed = redone = 0
    printed = 0

    pbar = tqdm(mp4_files, desc="Extracting WAV", unit="video", dynamic_ncols=True)
    for mp4_path in pbar:
        video_id = os.path.splitext(os.path.basename(mp4_path))[0]
        wav_path = os.path.join(OUT_DIR, f"{video_id}.wav")

        if os.path.exists(wav_path) and os.path.getsize(wav_path) >= MIN_BYTES:
            skipped += 1
            pbar.set_postfix(ok=ok, skipped=skipped, redone=redone, failed=failed)
            continue

        if os.path.exists(wav_path) and os.path.getsize(wav_path) < MIN_BYTES:
            redone += 1

        try:
            extract_wav(mp4_path, wav_path, sr=SR, mono=MONO)
            ok += 1
        except Exception as e:
            failed += 1
            err = str(e)

            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n===== {mp4_path} =====\n{err}\n")

            if printed < MAX_PRINT_ERRORS:
                printed += 1
                print(f"\n[ERROR] {mp4_path}\n{err[:2000]}\n")

            tmp_path = wav_path + ".tmp.wav"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        pbar.set_postfix(ok=ok, skipped=skipped, redone=redone, failed=failed)

    print(f"\n[DONE] ok={ok}, skipped={skipped}, redone={redone}, failed={failed}")
    print(f"[OUT]  {OUT_DIR}")
    print(f"[LOG]  {LOG_PATH}")

if __name__ == "__main__":
    main()