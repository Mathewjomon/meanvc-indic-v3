"""
Stage 1 - Prepare IndicVoices-R (Malayalam, Tamil, Hindi) for MeanVC-Indic v3.

WHAT THIS SCRIPT DOES
  Step A: --stage select   (about 1 minute, reads only the metadata table)
    - Reads data/ivr_metadata.parquet (made by the statistics cell in Colab).
    - Keeps good clips: 2-15 s long, transcript CER <= 0.10, SNR >= 15 dB.
    - For each language it picks:
        train : 10,000 clips, spread evenly over as many speakers as possible
        val   : up to 200 clips from 8 speakers that are NOT in train (for checking training)
        test  : up to 100 clips from the official test split (speakers never seen in training)
    - To avoid downloading all 165 GB, it uses only as many Parquet files as needed.
    - Saves the chosen clips to data/selection/selection.csv.

  Step B: --stage build    (about 30-60 minutes on Colab, downloads and converts audio)
    - Downloads only the Parquet files that contain chosen clips, one language at a time.
    - Converts each chosen clip: 44.1 kHz stereo  ->  16 kHz mono 16-bit WAV.
    - Packs train + val audio into a few large .tar files  -> data/audio_train/
      (a few big files are much faster on Drive than 30,000 small ones)
    - Saves test clips as normal .wav files you can play     -> data/audio_test/<ml|ta|hi>/
    - Writes the clip lists used by training                 -> data/manifests/{train,val,test}.tsv
    - If Colab disconnects, run it again: finished languages are skipped.

HOW TO RUN (in Colab, after logging in to Hugging Face)
  !python /content/drive/MyDrive/meanvc_indic_v3/code/data/prepare_ivr.py --stage select
  !python /content/drive/MyDrive/meanvc_indic_v3/code/data/prepare_ivr.py --stage build
"""

import argparse
import io
import re
import shutil
import tarfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

# --------------------------------------------------------------------------------------
# Paths: everything is found relative to this file, so the same script works on Colab
# (/content/drive/MyDrive/meanvc_indic_v3) and on the laptop (G:\My Drive\meanvc_indic_v3).
# --------------------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]      # .../meanvc_indic_v3
DATA = PROJECT / "data"
SELECTION_CSV = DATA / "selection" / "selection.csv"

# --------------------------------------------------------------------------------------
# Settings - change values here, not inside the functions.
# --------------------------------------------------------------------------------------
LANGS = {"Malayalam": "ml", "Tamil": "ta", "Hindi": "hi"}   # dataset folder -> short code
HF_REPO = "ai4bharat/indicvoices_r"

TARGET_SR = 16000            # MeanVC works at 16 kHz
MIN_SEC, MAX_SEC = 2.0, 15.0 # too short = little content, too long = wastes GPU memory
MAX_CER = 0.10               # transcript quality score from the dataset (lower is better)
MIN_SNR = 15.0               # signal-to-noise ratio in dB (higher is cleaner)

VAL_SPEAKERS = 8             # validation speakers per language (kept out of train)
VAL_PER_SPEAKER = 25         # -> up to 200 validation clips per language
TEST_CLIPS = 100             # test clips per language, from the official test split
FILE_MARGIN = 1.3            # use files until we have 1.3x the clips needed (room to balance speakers)
CLIPS_PER_TAR = 2000         # about 600 MB per .tar file


# ======================================================================================
# Small helpers
# ======================================================================================
def clean_text(s):
    """Tidy a transcript without changing the words:
    - Unicode NFC, so the same character is always stored the same way
    - every kind of space (normal, thin, non-breaking, tab) -> one normal space
    - remove invisible junk (zero-width space, byte-order mark)
    Zero-width joiner/non-joiner are KEPT: they matter for Malayalam/Hindi spelling."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def utt_id_for(code, file_name, row):
    """Unique, readable clip name, e.g. ml_tr00012_0153 = Malayalam, train file 12, row 153."""
    kind = "tr" if file_name.startswith("train") else "te"
    num = file_name.split("-")[1]                    # 'train-00012-of-00115.parquet' -> '00012'
    return f"{code}_{kind}{num}_{row:04d}"


def balanced_pick(df, n, rng):
    """Pick n clips spread evenly over speakers (round-robin):
    every speaker gives 1 clip, then a 2nd, then a 3rd ... until we have n.
    This gives the most different voices for a fixed number of clips."""
    queues = []
    for _, g in df.groupby("speaker_id"):
        idx = g.index.to_list()
        rng.shuffle(idx)
        queues.append(idx)
    rng.shuffle(queues)
    picked = []
    while len(picked) < n and queues:
        still_left = []
        for q in queues:
            if len(picked) >= n:
                break
            picked.append(q.pop())
            if q:
                still_left.append(q)
        queues = still_left
    return df.loc[picked]


def to_16k_mono(audio_bytes):
    """Decode one clip and convert it: any rate / stereo  ->  16 kHz mono 16-bit."""
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)                                   # stereo -> mono
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, sr // g)    # 44100 -> 16000 (x160 / 441)
    wav = np.clip(wav, -1.0, 1.0)                            # resampling can overshoot slightly
    return (wav * 32767).astype(np.int16)


# ======================================================================================
# Step A - choose the clips (metadata only, no audio)
# ======================================================================================
def select_language(meta, lang, n_train, rng):
    m = meta[meta.lang == lang]
    good = m[m.duration.between(MIN_SEC, MAX_SEC) & (m.cer <= MAX_CER) & (m.snr >= MIN_SNR)]
    print(f"\n{lang}: {len(m):,} clips, {len(good):,} pass the quality filters")

    # Test: official test split. Safety check: those speakers must never be in train/val.
    test_speakers = set(m[m.split == "test"].speaker_id)
    train_pool = good[(good.split == "train") & ~good.speaker_id.isin(test_speakers)]
    test = balanced_pick(good[good.split == "test"], TEST_CLIPS, rng)

    # Use files in random order until there are enough clips (+ room for validation speakers).
    per_file = train_pool.groupby("file").size()
    files = per_file.index.to_list()
    rng.shuffle(files)
    need = FILE_MARGIN * n_train + VAL_SPEAKERS * VAL_PER_SPEAKER
    chosen_files, total = [], 0
    for f in files:
        chosen_files.append(f)
        total += per_file[f]
        if total >= need:
            break
    pool = train_pool[train_pool.file.isin(chosen_files)]

    # Validation: whole speakers held out from training, so validation tells us how the
    # model does on voices it has not trained on.
    # Prefer random speakers with at least 10 clips; if there are too few, take the biggest ones.
    counts = pool.speaker_id.value_counts()
    eligible = counts[counts >= 10].index.to_list()
    rng.shuffle(eligible)
    if len(eligible) < VAL_SPEAKERS:
        eligible = counts.index.to_list()           # already sorted, most clips first
    val_speakers = eligible[:VAL_SPEAKERS]
    val = balanced_pick(pool[pool.speaker_id.isin(val_speakers)], VAL_SPEAKERS * VAL_PER_SPEAKER, rng)

    # Train: everything else, spread evenly over speakers.
    train_candidates = pool[~pool.speaker_id.isin(val_speakers)]
    if len(train_candidates) < n_train:
        print(f"  WARNING: only {len(train_candidates):,} clips available, using all of them")
    train = balanced_pick(train_candidates, min(n_train, len(train_candidates)), rng)

    parts = []
    for name, d in (("train", train), ("val", val), ("test", test)):
        d = d.copy()
        d["set"] = name
        parts.append(d)
        print(f"  {name:5s}: {len(d):6,} clips | {d.duration.sum() / 3600:5.1f} h | "
              f"{d.speaker_id.nunique():4d} speakers | "
              f"{(d.scenario != 'Read').mean() * 100 if len(d) else 0:3.0f}% spontaneous")
    sel = pd.concat(parts)
    print(f"  files to download: {sel.file.nunique()} "
          f"(of {m.file.nunique()} for this language)")
    return sel


def stage_select(args):
    meta_path = DATA / "ivr_metadata.parquet"
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path} - run the statistics cell in Colab first.")
    meta = pd.read_parquet(meta_path)
    rng = np.random.default_rng(args.seed)       # fixed seed = same selection every run

    sel = pd.concat([select_language(meta, lang, args.per_lang, rng) for lang in LANGS])
    sel["code"] = sel.lang.map(LANGS)
    sel["utt_id"] = [utt_id_for(c, f, r) for c, f, r in zip(sel.code, sel.file, sel.row)]
    assert sel.utt_id.is_unique, "clip names must be unique"

    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(SELECTION_CSV, index=False)
    print(f"\nSaved {len(sel):,} chosen clips -> {SELECTION_CSV}")


# ======================================================================================
# Step B - download, convert and pack the chosen clips
# ======================================================================================
def fetch_parquet(lang, file_name, raw_dir, source_dir):
    """Return a local path to one Parquet file (download it unless --source-dir is given)."""
    if source_dir:
        return Path(source_dir) / lang / file_name, False
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF_REPO, f"{lang}/{file_name}", repo_type="dataset", local_dir=raw_dir)
    return Path(p), True


def process_file(lang, file_name, rows, wav_dir, raw_dir, source_dir):
    """Download one Parquet file, convert its chosen clips to 16 kHz WAV, delete the file."""
    path, downloaded = fetch_parquet(lang, file_name, raw_dir, source_dir)
    table = pq.read_table(path, columns=["audio", "verbatim", "normalized"])
    out = []
    for r in rows.itertuples():
        rec = table.slice(int(r.row), 1).to_pylist()[0]
        pcm = to_16k_mono(rec["audio"]["bytes"])
        sf.write(wav_dir / f"{r.utt_id}.wav", pcm, TARGET_SR, subtype="PCM_16")
        out.append({"utt_id": r.utt_id,
                    "duration": round(len(pcm) / TARGET_SR, 3),
                    "text_verbatim": clean_text(rec["verbatim"]),
                    "text_normalized": clean_text(rec["normalized"])})
    del table
    if downloaded:
        path.unlink(missing_ok=True)       # free Colab disk space straight away
    return out


def build_language(lang, code, sel, args):
    part_file = DATA / "manifests" / "parts" / f"{code}.tsv"
    if part_file.exists():
        print(f"\n{lang}: already done (found {part_file.name}) - skipping")
        return

    t0 = time.time()
    work = Path(args.work)
    wav_dir = work / "wav16k" / code
    raw_dir = work / "raw"
    wav_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    s = sel[sel.lang == lang]
    groups = list(s.groupby("file"))
    print(f"\n{lang}: {len(s):,} clips from {len(groups)} files")

    # Several files at once: downloading is the slow part, so overlapping it saves time.
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process_file, lang, f, rows, wav_dir, raw_dir, args.source_dir)
                for f, rows in groups]
        for job in as_completed(jobs):
            results.extend(job.result())
            done += 1
            print(f"  {done}/{len(groups)} files | {len(results):,} clips converted | "
                  f"{(time.time() - t0) / 60:.1f} min")

    info = s.merge(pd.DataFrame(results), on="utt_id", suffixes=("_meta", ""))
    info["path"] = code + "/" + info.utt_id + ".wav"   # path relative to the audio folder

    # Pack train + val clips into a few large .tar files (fast on Drive and in Colab).
    tar_out = DATA / "audio_train"
    tar_out.mkdir(parents=True, exist_ok=True)
    trainval = info[info.set != "test"].sort_values("utt_id")
    info["shard"] = ""
    for k, start in enumerate(range(0, len(trainval), CLIPS_PER_TAR)):
        chunk = trainval.iloc[start:start + CLIPS_PER_TAR]
        name = f"{code}_{k:03d}.tar"
        local_tar = work / name
        with tarfile.open(local_tar, "w") as tar:
            for u in chunk.utt_id:
                tar.add(wav_dir / f"{u}.wav", arcname=f"{code}/{u}.wav")
        shutil.copy(local_tar, tar_out / name)            # build locally, then copy to Drive
        local_tar.unlink()
        info.loc[info.utt_id.isin(chunk.utt_id), "shard"] = name
        print(f"  packed {name} ({len(chunk)} clips)")

    # Test clips as normal WAV files, so they can be played in VS Code.
    test_out = DATA / "audio_test" / code
    test_out.mkdir(parents=True, exist_ok=True)
    for u in info[info.set == "test"].utt_id:
        shutil.copy(wav_dir / f"{u}.wav", test_out / f"{u}.wav")

    # Save this language's part of the manifest LAST: its existence means "language finished".
    part_file.parent.mkdir(parents=True, exist_ok=True)
    cols = ["utt_id", "set", "lang", "code", "speaker_id", "gender", "scenario",
            "duration", "path", "shard", "text_verbatim", "text_normalized"]
    info[cols].to_csv(part_file, sep="\t", index=False)
    print(f"  {lang} finished in {(time.time() - t0) / 60:.1f} min")


def stage_build(args):
    if not SELECTION_CSV.exists():
        raise SystemExit("Missing selection.csv - run with --stage select first.")
    sel = pd.read_csv(SELECTION_CSV, dtype={"speaker_id": str})
    for lang, code in LANGS.items():
        build_language(lang, code, sel, args)

    # Combine the per-language parts into the final train / val / test lists.
    parts = pd.concat([pd.read_csv(DATA / "manifests" / "parts" / f"{c}.tsv", sep="\t",
                                   dtype={"speaker_id": str}, keep_default_na=False)
                       for c in LANGS.values()])
    print("\nFinal manifests:")
    for name in ("train", "val", "test"):
        d = parts[parts.set == name]
        d.to_csv(DATA / "manifests" / f"{name}.tsv", sep="\t", index=False)
        print(f"  {name:5s}: {len(d):6,} clips, {d.duration.sum() / 3600:5.1f} h -> "
              f"data/manifests/{name}.tsv")
    print("Done. Train/val audio: data/audio_train/*.tar | Test audio: data/audio_test/")


# ======================================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare IndicVoices-R for MeanVC-Indic v3")
    ap.add_argument("--stage", choices=["select", "build"], required=True)
    ap.add_argument("--per-lang", type=int, default=10000, help="training clips per language")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", default="/content/work", help="fast local scratch folder (not Drive)")
    ap.add_argument("--workers", type=int, default=4, help="files processed at the same time")
    ap.add_argument("--source-dir", default=None,
                    help="read Parquet files from this folder instead of downloading (for testing)")
    args = ap.parse_args()
    stage_select(args) if args.stage == "select" else stage_build(args)