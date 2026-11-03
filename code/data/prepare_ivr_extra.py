"""
Stage 1b - Add MORE training clips for continued encoder training (round 2).

WHAT IT DOES
  Picks only clips that were NOT used before:
    - not in data/selection/selection.csv (round 1: train, val and test)
    - never from a validation or test speaker (so validation/test stay "unseen speakers")
    - same quality filters as round 1 (2-15 s, CER <= 0.10, SNR >= 15 dB)
  How many per language is set in EXTRA below ("all" = every remaining good clip).
  Then downloads, converts to 16 kHz mono and packs them exactly like prepare_ivr.py.

OUTPUT
  data/selection/selection_extra.csv          the chosen new clips
  data/audio_train/<ml|ta|hi>_extra_<k>.tar   their audio (next to the round-1 tar files)
  data/manifests/train_extra.tsv              list used by: train_encoder.py --train-manifest train_extra.tsv

HOW TO RUN (Colab, logged in to Hugging Face)
  !python /content/drive/MyDrive/meanvc_indic_v3/code/data/prepare_ivr_extra.py --stage select
  !python /content/drive/MyDrive/meanvc_indic_v3/code/data/prepare_ivr_extra.py --stage build
"""

import argparse
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_ivr as base  # noqa: E402  (same filters, naming, download + conversion)

DATA = base.DATA
EXTRA = {"Malayalam": "all", "Tamil": "all", "Hindi": 5000}     # new clips per language
EXTRA_CSV = DATA / "selection" / "selection_extra.csv"


# ======================================================================================
# Step A - choose new clips (metadata only)
# ======================================================================================
def select_language(meta, old, lang, n, rng):
    m = meta[meta.lang == lang]
    o = old[old.lang == lang]
    good = m[m.duration.between(base.MIN_SEC, base.MAX_SEC) & (m.cer <= base.MAX_CER)
             & (m.snr >= base.MIN_SNR)]

    used = set(zip(o.file, o.row))                                     # every round-1 clip
    blocked = set(o[o.set.isin(["val", "test"])].speaker_id) | set(m[m.split == "test"].speaker_id)
    pool = good[(good.split == "train") & ~good.speaker_id.isin(blocked)]
    pool = pool[[k not in used for k in zip(pool.file, pool.row)]]

    if n == "all" or n >= len(pool):
        pick = pool
    else:                                              # use only as many files as needed
        per_file = pool.groupby("file").size()
        files = per_file.index.to_list()
        rng.shuffle(files)
        chosen, total = [], 0
        for f in files:
            chosen.append(f)
            total += per_file[f]
            if total >= base.FILE_MARGIN * n:
                break
        pick = base.balanced_pick(pool[pool.file.isin(chosen)], n, rng)

    old_train_spk = set(o[o.set == "train"].speaker_id)
    new_spk = set(pick.speaker_id) - old_train_spk
    print(f"{lang}: {len(pool):,} unused good clips available -> picked {len(pick):,} | "
          f"{pick.duration.sum() / 3600:.1f} h | {pick.speaker_id.nunique()} speakers "
          f"({len(new_spk)} never heard in round 1) | "
          f"{(pick.scenario != 'Read').mean() * 100:.0f}% spontaneous | "
          f"files to download: {pick.file.nunique()}")
    return pick.assign(set="train")


def stage_select(args):
    meta = pd.read_parquet(DATA / "ivr_metadata.parquet")
    old = pd.read_csv(base.SELECTION_CSV, dtype={"speaker_id": str})
    rng = np.random.default_rng(args.seed + 1)            # different seed from round 1
    sel = pd.concat([select_language(meta, old, lang, EXTRA[lang], rng) for lang in base.LANGS])
    sel["code"] = sel.lang.map(base.LANGS)
    sel["utt_id"] = [base.utt_id_for(c, f, r) for c, f, r in zip(sel.code, sel.file, sel.row)]
    assert sel.utt_id.is_unique and not set(sel.utt_id) & set(old.utt_id), "overlap with round 1"
    sel.to_csv(EXTRA_CSV, index=False)
    print(f"\nTotal new clips: {len(sel):,} ({sel.duration.sum() / 3600:.1f} h) -> {EXTRA_CSV}")


# ======================================================================================
# Step B - download, convert, pack (same conversion as round 1)
# ======================================================================================
def build_language(lang, code, sel, args):
    part = DATA / "manifests" / "parts" / f"{code}_extra.tsv"
    if part.exists():
        print(f"\n{lang}: already done ({part.name}) - skipping")
        return
    t0 = time.time()
    work = Path(args.work)
    wav_dir, raw_dir = work / "wav16k" / code, work / "raw"
    wav_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    s = sel[sel.lang == lang]
    if s.empty:
        print(f"\n{lang}: no new clips selected - skipping")
        return
    groups = list(s.groupby("file"))
    print(f"\n{lang}: {len(s):,} new clips from {len(groups)} files")
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(base.process_file, lang, f, rows, wav_dir, raw_dir, args.source_dir)
                for f, rows in groups]
        for job in as_completed(jobs):
            results.extend(job.result())
            done += 1
            if done % 10 == 0 or done == len(groups):
                print(f"  {done}/{len(groups)} files | {len(results):,} clips | "
                      f"{(time.time() - t0) / 60:.1f} min")

    info = s.merge(pd.DataFrame(results), on="utt_id", suffixes=("_meta", ""))
    info["path"] = code + "/" + info.utt_id + ".wav"
    info = info.sort_values("utt_id").reset_index(drop=True)
    info["shard"] = ""
    out = DATA / "audio_train"
    for k, start in enumerate(range(0, len(info), base.CLIPS_PER_TAR)):
        chunk = info.iloc[start:start + base.CLIPS_PER_TAR]
        name = f"{code}_extra_{k:03d}.tar"
        local = work / name
        with tarfile.open(local, "w") as tar:
            for u in chunk.utt_id:
                tar.add(wav_dir / f"{u}.wav", arcname=f"{code}/{u}.wav")
        shutil.copy(local, out / name)
        local.unlink()
        info.loc[chunk.index, "shard"] = name
        print(f"  packed {name} ({len(chunk)} clips)")

    cols = ["utt_id", "set", "lang", "code", "speaker_id", "gender", "scenario",
            "duration", "path", "shard", "text_verbatim", "text_normalized"]
    part.parent.mkdir(parents=True, exist_ok=True)
    info[cols].to_csv(part, sep="\t", index=False)        # written last = language finished
    print(f"  {lang} finished in {(time.time() - t0) / 60:.1f} min")


def stage_build(args):
    sel = pd.read_csv(EXTRA_CSV, dtype={"speaker_id": str})
    for lang, code in base.LANGS.items():
        build_language(lang, code, sel, args)
    files = [DATA / "manifests" / "parts" / f"{c}_extra.tsv" for c in base.LANGS.values()]
    parts = pd.concat([pd.read_csv(f, sep="\t", dtype={"speaker_id": str}, keep_default_na=False)
                       for f in files if f.exists()])
    parts.to_csv(DATA / "manifests" / "train_extra.tsv", sep="\t", index=False)
    print(f"\ntrain_extra: {len(parts):,} clips, {parts.duration.sum() / 3600:.1f} h "
          f"-> data/manifests/train_extra.tsv")
    for code, g in parts.groupby("code"):
        print(f"  {code}: {len(g):,} clips, {g.duration.sum() / 3600:.1f} h, {g.speaker_id.nunique()} speakers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add new (unused) IndicVoices-R clips for round 2")
    ap.add_argument("--stage", choices=["select", "build"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source-dir", default=None, help="local Parquet folder instead of downloading (testing)")
    a = ap.parse_args()
    stage_select(a) if a.stage == "select" else stage_build(a)